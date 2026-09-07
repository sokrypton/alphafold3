'''Running AlphaFold 2 through the same inference pipeline as everything else.

`AF2ModelRunner` presents the interface `run_alphafold.ModelRunner` presents --
`model_name`, `model_dir`, `model_params`, `run_inference`,
`extract_inference_results`, `extract_embeddings`, `extract_distogram` -- so
`predict_structure` and everything downstream of it (the mmCIF writer, the
confidence JSONs, the ranking CSV, the terms-of-use file) run unchanged. AF2 is
selected by `spec.engine`, not by a branch threaded through the pipeline.

It takes AF3's featurised batch, exactly as an AF3-family model does. There is no
second featurisation: `alphafold3.af2.features.from_af3_batch` re-reads that
batch, and `alphafold3.af2.output` puts the coordinates back into its atom
layout. So the same JSON in gives comparable artifacts out on either engine.
'''

from __future__ import annotations

import concurrent.futures
import functools

import numpy as np

from alphafold3.af2 import features as af2_features
from alphafold3.af2 import output as af2_output
from alphafold3.af2.common import confidence as af2_confidence

_NAN = float('nan')


class AF2ModelRunner:
  """Runs AlphaFold 2 behind run_alphafold's ModelRunner interface."""

  def __init__(self, spec, device, model_dir, *, num_recycles=3,
               use_bfloat16=True, num_msa=512, num_extra_msa=1024,
               model_names=None, use_cluster_profile=True):
    self._spec = spec
    self._device = device
    self._model_dir = str(model_dir)
    self._num_recycles = num_recycles
    self._use_bfloat16 = use_bfloat16
    self._num_msa = num_msa
    self._num_extra_msa = num_extra_msa
    self._model_names = model_names
    self._use_cluster_profile = use_cluster_profile

  @property
  def model_dir(self):
    return self._model_dir

  @property
  def model_name(self) -> str:
    return self._spec.name

  @functools.cached_property
  def _runner(self):
    from alphafold3.af2.runner import AF2Runner

    return AF2Runner(
        model_type=self._spec.model_type,
        data_dir=self._model_dir,
        model_names=self._model_names,
        num_recycle=self._num_recycles,
        use_bfloat16=self._use_bfloat16,
        num_msa=self._num_msa,
        num_extra_msa=self._num_extra_msa,
        use_cluster_profile=self._use_cluster_profile,
    )

  @functools.cached_property
  def model_params(self):
    """The DeepMind AlphaFold 2 parameter sets found in model_dir.

    Loaded eagerly by run_alphafold before it launches anything, so a missing
    params directory is reported before a fold starts rather than after.
    """
    return self._runner.model_params

  def forward(self, batch, *, soft_seq=None, design_mask=None, key=None,
              opt=None, model_params=None):
    """batch (+ an optional soft sequence) -> AF2 outputs. DIFFERENTIABLE.

    `soft_seq` is a distribution over the 20 standard amino acids, shaped
    (num_tokens, 20) or (num_seq, num_tokens, 20) -- the same thing
    `alphafold3.model.Model.__call__` takes, so ONE design loop drives either
    engine. `design_mask` selects which tokens it replaces; the rest keep the
    batch's own aatype, which is what a binder target or a scaffolded motif
    needs.

    The relaxation stays with the CALLER, deliberately. AF2's own `soft_seq`
    turns parameters into a distribution with its own alpha/temp/soft/hard
    schedule, and AF3 has no equivalent -- so driving both engines through AF2's
    schedule would mean two different meanings for one design loop. At
    `soft=0, hard=0` that transform is `pseudo = input`, an exact identity, so
    handing it a distribution passes it through untouched. That identity is what
    makes one convention possible; it is not an approximation.

    With `soft_seq=None` this is plain prediction: the batch's own sequence as a
    one-hot. `run_inference` is exactly that call, so prediction and design are
    not two code paths here either.
    """
    import jax
    import jax.numpy as jnp

    inputs, seq = af2_features.from_af3_batch(batch, use_msa=True)
    num_seq = self._runner.num_seq
    aatype = jnp.asarray(
        [af2_features.rc.restype_order.get(a, af2_features.rc.restype_num)
         for a in seq])
    # Clipped to the 20 standard types: an X would one-hot to index 20, which is
    # outside the alphabet AF2's sequence parameters span.
    wt = jax.nn.one_hot(jnp.clip(aatype, 0, 19), 20)

    if soft_seq is None:
      blended = wt
    else:
      soft_seq = jnp.asarray(soft_seq)
      if soft_seq.ndim == 3:
        soft_seq = soft_seq[0]
      if design_mask is None:
        blended = soft_seq
      else:
        blended = jnp.where(jnp.asarray(design_mask)[:, None], soft_seq, wt)
    params = {'seq': jnp.broadcast_to(blended, (num_seq,) + blended.shape)}

    # soft/hard 0 so AF2's transform is the identity described above; alpha and
    # temp are then irrelevant but pinned so a caller's opt cannot reintroduce a
    # schedule by accident.
    full_opt = {'alpha': 1.0, 'temp': 1.0, 'soft': 0.0, 'hard': 0.0,
                'weights': {}}
    if opt:
      full_opt.update(opt)
    if key is None:
      key = jax.random.PRNGKey(0)
    return self._runner.apply(
        params, {**inputs, 'opt': full_opt}, key, model_params=model_params)

  def run_inference(self, featurised_example, rng_key):
    """One forward pass, from the SAME featurised batch an AF3 model gets."""
    import jax

    from alphafold3.model import feat_batch

    batch = feat_batch.Batch.from_data_dict(featurised_example)
    outputs = self.forward(batch, key=rng_key)
    result = jax.tree.map(np.asarray, dict(outputs))
    result['__identifier__'] = self.model_name.encode()
    # The batch travels with the result: extract_inference_results is handed the
    # BatchDict, and rebuilding the Batch there would be the second place that
    # has to agree about padding.
    result['__batch__'] = batch
    return result

  def extract_inference_results(self, batch, result, target_name: str):
    """-> [InferenceResult], with AF2's own confidences in AF3's fields.

    A LIST, not the generator that builds it: run_alphafold indexes the result
    (`inference_results[0]`), and ModelRunner.extract_inference_results returns a
    list for the same reason.
    """
    return list(self._inference_results(batch, result, target_name))

  def _inference_results(self, batch, result, target_name: str):
    del target_name
    from alphafold3.model import confidences, model as af3_model

    fb = result['__batch__']
    num_tokens = af2_features.num_real_tokens(fb)
    asym_id = np.asarray(fb.token_features.asym_id)[:num_tokens]

    plddt = af2_confidence.compute_plddt(
        np.asarray(result['predicted_lddt']['logits']))
    pae_out = result.get('predicted_aligned_error')
    if pae_out is not None:
      logits = np.asarray(pae_out['logits'])
      breaks = np.asarray(pae_out['breaks'])
      pae = af2_confidence.compute_predicted_aligned_error(
          logits, breaks)['predicted_aligned_error']
      ptm = float(af2_confidence.predicted_tm_score(logits, breaks))
      multi_chain = len(np.unique(asym_id)) > 1
      iptm = (float(af2_confidence.predicted_tm_score(
          logits, breaks, asym_id=asym_id)) if multi_chain else _NAN)
    else:
      pae = np.full((num_tokens, num_tokens), _NAN, np.float32)
      ptm = iptm = _NAN
      multi_chain = False

    pred_structure = af2_output.predicted_structure(result, fb)
    pred_structures = pred_structure.unstack()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(pred_structures), 32)) as ex:
      has_clash = list(ex.map(confidences.has_clash, pred_structures))
      fraction_disordered = list(
          ex.map(confidences.fraction_disordered, pred_structures))

    # AlphaFold 2's own ranking: mean pLDDT for a monomer, and the multimer
    # paper's 0.8*ipTM + 0.2*pTM for a complex. Deliberately AF2's formula and
    # not AF3's, because it is AF2's confidence heads being ranked.
    ranking = (0.8 * iptm + 0.2 * ptm) if multi_chain else float(
        np.mean(plddt[:num_tokens])) / 100.0

    empty_pair = np.full((num_tokens, num_tokens), _NAN, np.float32)
    contact_probs = _contact_probs(result, num_tokens)
    chain_ids = [pred_structure.chains[a - 1] for a in asym_id]
    res_ids = np.asarray(fb.token_features.residue_index)[:num_tokens]

    for idx, one in enumerate(pred_structures):
      yield af3_model.InferenceResult(
          predicted_structure=one,
          numerical_data={
              # AF2 has no PDE head at all -- not a missing output but a part of
              # the architecture that does not exist -- so these stay NaN.
              'full_pde': empty_pair,
              'full_pae': pae[:num_tokens, :num_tokens],
              'contact_probs': contact_probs,
          },
          metadata={
              'predicted_distance_error': _NAN,
              'ranking_score': ranking,
              'fraction_disordered': fraction_disordered[idx],
              'has_clash': has_clash[idx],
              'predicted_tm_score': ptm,
              'interface_predicted_tm_score': iptm,
              'chain_pair_pde_mean': np.full((1, 1), _NAN),
              'chain_pair_pde_min': np.full((1, 1), _NAN),
              'chain_pair_pae_min': np.full((1, 1), _NAN),
              'ptm': ptm,
              'iptm': iptm,
              'ptm_iptm_average': ranking,
              'intra_chain_single_pde': _NAN,
              'cross_chain_single_pde': _NAN,
              'pae_ichain': _NAN,
              'pae_xchain': _NAN,
              'ranking_confidence': ranking,
              'ranking_confidence_pae': _NAN,
              'chain_pair_iptm': np.full((1, 1), iptm),
              'iptm_ichain': _NAN,
              'iptm_xchain': _NAN,
              'token_chain_ids': chain_ids,
              'token_res_ids': res_ids,
          },
          model_id=result['__identifier__'],
          debug_outputs={},
      )

  def extract_embeddings(self, result, num_tokens):
    reps = result.get('representations') or {}
    out = {}
    if 'single' in reps:
      out['single_embeddings'] = np.asarray(
          reps['single'])[:num_tokens].astype(np.float16)
    if 'pair' in reps:
      out['pair_embeddings'] = np.asarray(
          reps['pair'])[:num_tokens, :num_tokens].astype(np.float16)
    return out or None

  def extract_distogram(self, result, num_tokens):
    dgram = result.get('distogram')
    if not dgram or 'logits' not in dgram:
      return None
    return np.asarray(dgram['logits'])[:num_tokens, :num_tokens, :]


def _contact_probs(result, num_tokens):
  """AF2's distogram logits -> P(C-beta distance < 8 A), as AF3 reports."""
  dgram = result.get('distogram')
  if not dgram or 'logits' not in dgram:
    return np.full((num_tokens, num_tokens), _NAN, np.float32)
  logits = np.asarray(dgram['logits'])[:num_tokens, :num_tokens]
  breaks = np.asarray(dgram['bin_edges'])
  probs = np.exp(logits - logits.max(-1, keepdims=True))
  probs /= probs.sum(-1, keepdims=True)
  # bin i covers [breaks[i-1], breaks[i]); the last bin is open-ended.
  return probs[..., np.concatenate([breaks, [np.inf]]) < 8.0].sum(-1).astype(
      np.float32)
