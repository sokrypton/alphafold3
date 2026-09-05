"""Derive the parameter-shape manifest that ships beside a converted blob.

Loading a model should never have to ask the graph what its parameters are.
`jax.eval_shape` answers that in ~5 seconds without executing anything, which is
5 seconds on every single run -- and the answer only changes when the graph or
the config does, both of which are fixed by the time we publish a blob. So we
derive it once, here, at conversion time, and write it next to the weights.

The manifest is also the honest record of what a conversion covers: a parameter
the graph wants and the converter never produced is listed under "missing", so a
run can say which head is running on zeros instead of discovering it as a fold
that is quietly bad. (chai-1 shipped a blob whose template and MSA parameters
were random init, in production, for exactly this want of a manifest.)

Derived from the graph rather than from the converted parameters, so it cannot
agree with a converter that is wrong -- and so it covers what the converter
missed.
"""

from __future__ import annotations

import json
import os

import numpy as np


# A canonical input to derive shapes from. The parameter tree does not depend on
# the sequence or on how many tokens there are -- only on the config and on which
# optional feature blocks the batch carries -- so this is deliberately tiny.
_CANONICAL_SEQUENCE = 'MQIFVKTLTGKTITLEVE'


def canonical_batch(model_name, model_dir=None):
  """A small featurised batch carrying every block `model_name`'s graph reads.

  model_dir is where this model's weights live; chai-1 reads its own standard
  residue conformers from there.
  """
  from alphafold3.common import folding_input
  from alphafold3.constants import decoded_ccd
  from alphafold3.data import featurisation
  from alphafold3.model import model_registry
  from alphafold3.model.pipeline import model_features

  spec = model_registry.get(model_name)
  fold_input = folding_input.Input(
      name='shapes',
      chains=[folding_input.ProteinChain(
          id='A', sequence=_CANONICAL_SEQUENCE, ptms=[],
          unpaired_msa='', paired_msa='', templates=[])],
      rng_seeds=[0],
  )
  ccd = decoded_ccd.get_ccd()
  featurise = lambda **kw: featurisation.featurise_input(
      fold_input=fold_input, ccd=ccd, buckets=None, **kw)
  batch = featurise()[0]
  if spec.featurise:
    esm = None
    if spec.featurise.get('esm'):
      # chai-1 reads a (num_protein_tokens, 2560) block; its CONTENT cannot
      # change the parameter tree, only its presence and width can.
      n = int(np.asarray(batch['is_protein']).sum())
      esm = np.zeros((n, 2560), np.float32)
    lm_pair = None
    if spec.featurise.get('lm_pair'):
      # ESMFold2's language-model pair representation. Like chai's ESM block,
      # only its PRESENCE and width matter here -- but presence matters a lot:
      # without it the graph never builds the lm_encoder and the audit reports
      # its 36 stacked parameters as extra.
      from alphafold3.model import model as _af3_model
      cfg = _af3_model.Model.Config()
      spec.configure(cfg)
      n = int(np.asarray(batch['token_index']).shape[-1])
      lm_pair = np.zeros((n, n, cfg.evoformer.pair_channel), np.float32)
    batch = model_features.apply(
        batch, spec, refeaturise=featurise, model_dir=model_dir, esm=esm,
        has_msa=False, fold_input=fold_input, lm_pair=lm_pair)
  return batch


def derive(model_name, batch=None, model_dir=None):
  """-> {scope: {name: (dtype, shape)}} for `model_name`'s full parameter tree."""
  import haiku as hk
  import jax
  from alphafold3.model import model as af3_model
  from alphafold3.model import model_registry
  from alphafold3.model.components import utils

  config = af3_model.Model.Config()
  # xla, not the device default: deriving shapes must not depend on which GPU
  # (or none) this happens to run on.
  config.global_config.flash_attention_implementation = 'xla'
  model_registry.get(model_name).configure(config)

  if batch is None:
    batch = canonical_batch(model_name, model_dir=model_dir)

  def forward(b):
    return af3_model.Model(config)(b)

  shapes = jax.eval_shape(hk.transform(forward).init, jax.random.PRNGKey(0),
                          utils.remove_invalidly_typed_feats(batch))
  return {scope: {name: (str(np.dtype(leaf.dtype)), list(leaf.shape))
                  for name, leaf in leaves.items()}
          for scope, leaves in shapes.items()}


def provenance(model_name, checkpoint=None):
  """What produced this blob, so a stale one can be spotted rather than trusted.

  A conversion's output looks the same however old the converter was. The
  published OpenDDE blob turned out to predate a residue-alphabet fix -- gaps
  embedded as RNA adenine, every RNA base read one slot high -- and nothing
  about the file said so; it was found only by reconverting and diffing. The
  converter's source hash makes that visible.
  """
  import datetime
  import hashlib

  here = os.path.dirname(os.path.abspath(__file__))
  digest = hashlib.sha256()
  for name in (f'{model_name}.py', 'common.py'):
    path = os.path.join(here, name)
    if os.path.exists(path):
      with open(path, 'rb') as fh:
        digest.update(fh.read())
  out = {
      'converted': datetime.datetime.now(datetime.timezone.utc).isoformat(
          timespec='seconds'),
      'converter_sha256': digest.hexdigest()[:16],
  }
  if checkpoint is not None:
    path = os.path.expanduser(str(checkpoint))
    out['source'] = os.path.basename(path.rstrip('/'))
    if os.path.isfile(path):
      out['source_bytes'] = os.path.getsize(path)
  return out


def converter_is_current(manifest):
  """True if `manifest` was written by the converter source as it stands now."""
  recorded = (manifest.get('provenance') or {}).get('converter_sha256')
  if recorded is None:
    return None                       # written before provenance was recorded
  return recorded == provenance(manifest['model'])['converter_sha256']


def write(model_name, out_dir, params=None, batch=None, checkpoint=None):
  """Write <model>.shapes.json into out_dir; returns the path.

  `params` is the converted tree, used only to record which parameters the
  conversion did NOT produce.
  """
  shapes = derive(model_name, batch=batch, model_dir=out_dir)
  supplied = set()
  if params is not None:
    supplied = {f'{scope}/{name}'
                for scope, leaves in params.items() for name in leaves}
  missing = sorted(f'{scope}/{name}'
                   for scope, leaves in shapes.items() for name in leaves
                   if f'{scope}/{name}' not in supplied) if params else []
  manifest = {
      'model': model_name,
      'provenance': provenance(model_name, checkpoint),
      'num_arrays': sum(len(v) for v in shapes.values()),
      'missing': missing,
      'shapes': shapes,
  }
  out_dir = os.path.expanduser(str(out_dir))
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f'{model_name}.shapes.json')
  with open(path, 'w') as fh:
    json.dump(manifest, fh, indent=1, sort_keys=True)
  return path
