"""Generate the checked-in parameter-shape tables that models.build loads.

Why generate rather than hand-write: the tables are a SECOND statement of what the
haiku graph's parameters are. Derived mechanically from jax.eval_shape they cannot
disagree with the graph -- regenerate and the diff shows exactly what a converter or
config change did. Hand-authored "add these entries when flag X is set" logic is a
reimplementation of the graph's shape rules, and when it drifts the failure is a
WRONG tree (a converter weight silently landing in the wrong slot), not a slow one.

So config-dependence is handled by generating one table per (model, config
signature), not by conditionals in Python.

Usage:
    python -m tools.module_trace.gen_param_shapes            # every available model
    python -m tools.module_trace.gen_param_shapes rosettafold3 alphafold3

Writes colabdesign2/af3/param_shapes/<model>.json -- inside the package, so a pip
install carries them (tools/ does not ship). models.build prefers these,
falls back to the user cache, then to eval_shape -- so a missing or stale table is
only ever a slowdown.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(HERE, '..', '..', 'colabdesign2', 'af3', 'param_shapes')

# The configs worth pre-deriving: what the tests and oracle scripts actually build.
# A config not listed here still works -- it just derives at runtime.
STANDARD = dict(num_recycles=0, diffusion_steps=1, num_msa=1, flash_attention='xla')


# The cases worth pre-deriving. The parameter tree depends on WHICH OPTIONAL
# FEATURES a batch carries, so one entry per shape of input, not one per model:
# a table generated from a ligand case does not serve a protein-only run, and the
# miss is silent -- the only symptom is a slow start on a fresh Colab VM.
CASES = ('lig_btn', '6mrr_single_seq', '1ehz_trna')


def generate(model, case_name='lig_btn', **cfg_kwargs):
  from tools.module_trace import cases as mt_cases, models
  kw = dict(STANDARD)
  kw.update(cfg_kwargs)
  batch = mt_cases.build(case_name, **models.featurise_kwargs(model))['batch']
  cfg, _params, _fwd, missing = models.build(model, batch, shape_cache=False, **kw)
  # re-derive the raw tree rather than reading it back off `params`, so the table is
  # the graph's answer and not a round-trip through the converter merge
  import haiku as hk, jax
  from colabdesign2.af3.alphafold3.model.components import utils  # noqa: F811
  from colabdesign2.af3.alphafold3.model import model as af3_model

  def fwd(b, soft_seq=None, design_mask=None, structure=True, use_dropout=False):
    return af3_model.Model(cfg)(b, soft_seq=soft_seq, design_mask=design_mask,
                                structure=structure, use_dropout=use_dropout)
  import numpy as np
  shapes = jax.eval_shape(hk.transform(fwd).init, jax.random.PRNGKey(0),
                          utils.remove_invalidly_typed_feats(batch))
  # Sign with the SAME fbatch the lookup will present, or the entry can never
  # match: _param_shapes calls _repo_table(model, cfg, fbatch) while this used to
  # sign without it, so every checked-in table was dead on arrival.
  table = {
      'signature': models._shape_cache_signature(
          model, cfg, utils.remove_invalidly_typed_feats(batch)),
      'config': kw,
      'case': case_name,
      'num_arrays': sum(len(v) for v in shapes.values()),
      'unported': sorted(missing),
      'shapes': {sc: {nm: [str(np.dtype(r.dtype)), list(r.shape)]
                      for nm, r in v.items()} for sc, v in shapes.items()},
  }
  return table, len(missing)


def write(model, tables):
  """Write one file holding every generated case for `model`."""
  os.makedirs(TABLE_DIR, exist_ok=True)
  path = os.path.join(TABLE_DIR, model + '.json')
  with open(path, 'w') as fh:
    json.dump({'entries': tables}, fh, indent=1, sort_keys=True)
  return path


def main(argv):
  from tools.module_trace import models
  wanted = argv or [m for m in models.CHECKPOINTS if models.available(m)[0]]
  for model in wanted:
    ok, why = models.available(model)
    if not ok:
      print(f'{model:14s} SKIP  {why}')
      continue
    tables = []
    for case in CASES:
      try:
        table, n_missing = generate(model, case_name=case)
      except Exception as err:            # a case a model cannot featurise
        print(f'{model:14s} {case:18s} SKIP  {type(err).__name__}: '
              f'{str(err)[:70]}')
        continue
      tables.append(table)
      print(f'{model:14s} {case:18s} {table["num_arrays"]:4d} arrays, '
            f'{n_missing:3d} unported')
    if tables:
      print(f'{model:14s} -> {os.path.relpath(write(model, tables))} '
            f'({len(tables)} cases)')


if __name__ == '__main__':
  sys.path.insert(0, os.path.join(HERE, '..', '..'))
  main(sys.argv[1:])
