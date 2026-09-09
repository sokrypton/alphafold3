"""Fold a target with a converted model and report CA-RMSD -- the pipeline gate.

Module-equivalence (dev/oracles/prot_parity.py) says our converted blocks
compute what native's do on synthetic activations. It says nothing about
featurisation. This runs the whole model.

  PYTHONPATH=src:. python dev/oracles/fold_check.py protenix2 [6MRR.pdb]
"""
import os
import sys

import numpy as np

# parse_ca moved into the package: a tracked module needs it too, and these
# harnesses are out of tree.
from converters.pdb import A3, parse_ca  # noqa: F401  (re-exported)


def kabsch_rmsd(a, b):
  n = min(len(a), len(b))
  a, b = a[:n] - a[:n].mean(0), b[:n] - b[:n].mean(0)
  u, _, vt = np.linalg.svd(a.T @ b)
  d = np.sign(np.linalg.det(u @ vt))
  return float(np.sqrt((((a @ (u @ np.diag([1, 1, d]) @ vt)) - b) ** 2).sum(1).mean()))


LAST_RAW_BATCH = None


def build_batch(model_name, seq, model_dir=None, templates=None):
  """Featurise one protein chain for `model_name`, applying its own conventions.

  Split out of fold() so a second harness -- dev/oracles/grad_check.py -- gets
  the SAME batch a fold gets, including the per-model featurisation conventions
  and the language-model inputs. A gradient measured on a differently-built
  batch would be a gradient of a different model.
  """
  return _fold_setup(model_name, seq, model_dir, templates)[0]


def _fold_setup(model_name, seq, model_dir=None, templates=None, seed=0,
                chains=None, bonded_atom_pairs=None):
  """-> (batch, cfg, model_dir). Everything the harnesses need, done once.

  `chains` replaces the single protein chain built from `seq`, so the same
  featurisation and the same per-model conventions serve RNA, DNA, ligands and
  complexes -- see modality_check.py. `seq` is ignored when it is given.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp
  from alphafold3.common import folding_input
  from alphafold3.constants import decoded_ccd
  from alphafold3.data import featurisation
  from alphafold3.model import model as af3_model, model_registry
  from alphafold3.model import params as afp
  from alphafold3.model.components import utils
  from alphafold3.model.pipeline import model_features

  model_dir = model_dir or os.path.expanduser('~/ported/%s' % model_name)
  spec = model_registry.get(model_name)
  fold_input = folding_input.Input(
      name=model_name,
      chains=chains or [folding_input.ProteinChain(
          id='A', sequence=seq, ptms=[], unpaired_msa='', paired_msa='',
          templates=templates or [])],
      rng_seeds=[seed],
      bonded_atom_pairs=bonded_atom_pairs)
  ccd = decoded_ccd.get_ccd()
  featurise = lambda **kw: featurisation.featurise_input(
      fold_input=fold_input, ccd=ccd, buckets=None, **kw)
  batch = featurise()[0]
  # ESMFold2 folds from ESM-C's hidden states. LM_PAIR names an npz written by
  # converters.esmfold2_lm; without it the model still folds, just without its
  # language model (natively 1.719 A on 6MRR against 1.7 with it).
  # chai-1's TOKEN stream is mostly ESM2, so folding it with esm=None is folding
  # a different model (5.70 A where chai reaches 0.642). ESM_EMB names an npz
  # from alphafold3.model.esm --family esm2.
  esm = None
  if os.environ.get('ESM_EMB'):
    e = np.load(os.path.expanduser(os.environ['ESM_EMB']), allow_pickle=True)
    esm = e['esm'] if hasattr(e, 'files') else e
  lm_pair = None
  lm_npz = os.environ.get('LM_PAIR')
  hidden = os.environ.get('ESMC_HIDDEN')
  if hidden:
    # Preferred over LM_PAIR: build the pair rep with THIS model's own shim.
    # Every ESMFold2 release trains its own, and handing one variant another's
    # reads corr 0.026 against native -- a mistake a precomputed lm_pair file
    # makes very easy to make.
    from alphafold3.model import esm as esmfold2_lm
    h = np.load(os.path.expanduser(hidden))
    h = h['lm_hidden'] if hasattr(h, 'files') else h
    lm_pair = np.asarray(esmfold2_lm.shim(
        h, esmfold2_lm.load_shim_params(model_dir, model_name)), np.float32)
  elif lm_npz:
    z = np.load(os.path.expanduser(lm_npz))
    lm_pair = z['lm_pair'] if hasattr(z, 'files') else z
  if spec.featurise:
    batch = model_features.apply(batch, spec, refeaturise=featurise,
                                 model_dir=model_dir, esm=esm, has_msa=False,
                                 fold_input=fold_input, lm_pair=lm_pair)
  cfg = af3_model.Model.Config()
  cfg.global_config.flash_attention_implementation = 'xla'
  spec.configure(cfg)
  # BF16=none runs the trunk in float32. Worth having as a knob rather than a
  # constant: a model whose trunk is already marginal can be pushed over by
  # bfloat16, and that is invisible next to a gate that was measured in fp32.
  if os.environ.get('BF16'):
    cfg.global_config.bfloat16 = os.environ['BF16']
  # STEPS overrides the sampler's step count. Needed to compare like with like:
  # esmfold2_native_variants.py runs native at num_sampling_steps=200 whatever
  # the release's config says, so a fold at the config's own count is not the
  # same experiment.
  # NUM_MSA caps the MSA depth the trunk consumes (evoformer truncates to
  # config.num_msa, default 1024). A knob because depth is a real variable for
  # the models that HAVE an MSA encoder. ESMFold2-Experimental (dropped
  # 2026-09-09) folded 1STP at 14.4 A on a
  # 2145-row MSA and 0.476 A on none, so the response to depth is the
  # measurement that localises it.
  if os.environ.get('RECYCLES'):
    # Every module gate runs ONE trunk pass. A fold recycles (10 by default), so
    # a difference that a gate calls exact can still be re-injected through
    # `z = z_init + z_recycle(z_norm(z))` on every pass -- which is exactly the
    # boltz2 relative-CHAIN question: the trunk z-init is exact against the
    # vendor with one convention and the FOLD prefers the other. Being able to
    # fold at zero recycles is what separates those.
    cfg.num_recycles = int(os.environ['RECYCLES'])
  if os.environ.get('MSA_STACK'):
    # How many MSA-stack blocks run INSIDE the recycle loop. Zero removes the
    # module (and, for boltz2/chai1, the double-add that goes with it), which is
    # a differential test rather than a realistic setting: it says whether a
    # difference that only appears with recycling lives in that module.
    cfg.evoformer.msa_stack.num_layer = int(os.environ['MSA_STACK'])
  if os.environ.get('NUM_MSA'):
    cfg.evoformer.num_msa = int(os.environ['NUM_MSA'])
  if os.environ.get('STEPS'):
    cfg.heads.diffusion.eval.steps = int(os.environ['STEPS'])

  # run_inference's preprocessing: a bare apply on the raw batch raises
  # TracerArrayConversionError on its non-array fields.
  #
  # KEEP THE UNSTRIPPED BATCH. The fields being removed are the string-typed
  # layout columns -- chain_id, res_name, atom_name -- and the mmCIF WRITER
  # needs exactly those: `get_inference_result` reads
  # `batch.convert_model_output.flat_output_layout.chain_id` and gets None
  # without them. modality_check.py's --write reads it from here rather than
  # re-featurising, which would be a second featurisation and therefore a second
  # thing that can differ.
  global LAST_RAW_BATCH
  LAST_RAW_BATCH = batch
  batch = jax.tree_util.tree_map(jnp.asarray,
                                 utils.remove_invalidly_typed_feats(batch))
  return batch, cfg, model_dir


def fold(model_name, seq, model_dir=None, seed=0, templates=None, chains=None,
         bonded_atom_pairs=None):
  import haiku as hk
  import jax
  from alphafold3.model import model as af3_model
  from alphafold3.model import params as afp

  batch, cfg, model_dir = _fold_setup(model_name, seq, model_dir, templates,
                                      seed=seed, chains=chains,
                                      bonded_atom_pairs=bonded_atom_pairs)

  @hk.transform
  def forward(b):
    return af3_model.Model(cfg)(b)

  out = forward.apply(afp.get_model_haiku_params(model_dir=model_dir),
                      jax.random.PRNGKey(seed), batch)
  return out, batch


def main(argv):
  sys.argv = sys.argv[:1]                       # tokamax parses argv lazily
  model_name = argv[0]
  pdb = argv[1] if len(argv) > 1 else os.path.expanduser('~/6MRR.pdb')
  seq, native = parse_ca(pdb)
  print('%s: %d residues from %s' % (model_name, len(seq), os.path.basename(pdb)))
  # SEEDS lets one run cover several seeds: best-of-5 from ONE seed is a noisy
  # statistic, and comparing it across a change that alters the trajectory
  # (rather than just the weights) reads as a regression when it is sampling.
  seeds = [int(x) for x in os.environ.get('SEEDS', '0').split(',')]
  # MODEL_DIR points at an alternative blob for the same model -- e.g. one
  # converted at a different storage dtype, which is how intellifold2's bf16
  # weights were priced against the same weights at fp32.
  mdir = os.environ.get('MODEL_DIR')
  outs = [fold(model_name, seq, model_dir=mdir, seed=s_) for s_ in seeds]
  out, batch = outs[0]
  pos = np.concatenate(
      [np.asarray(o['diffusion_samples']['atom_positions']) for o, _ in outs])
  # dense per-token atom layout is N, CA, C, O, ... so CA is slot 1
  ca = pos[:, :, 1, :]
  rs = [kabsch_rmsd(ca[i], native) for i in range(ca.shape[0])]
  print('  %d samples, CA-RMSD: %s   best %.3f  mean %.3f'
        % (len(rs), ' '.join('%.3f' % r for r in rs), min(rs), float(np.mean(rs))))
  return pos, native, batch


if __name__ == '__main__':
  main(sys.argv[1:])
