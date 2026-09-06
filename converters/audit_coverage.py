# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Which tensors of a native checkpoint does our converter never read?

    python -m converters.audit_coverage protenix2 [--verbose]

The gap this closes: `fill_from_manifest` audits the GRAPH side -- parameters the
model wants that the blob lacks, filled with zeros and reported. Nothing audited
the CHECKPOINT side, so a trained tensor the graph has no slot for was simply
dropped in silence. That is how protenix2's `distogram_head.linear.bias` went
missing: `hm.Linear` defaults to `use_bias=False`, the converter never asked for
the key, and no gate could see it (the distogram feeds no structure).

Run BOTH directions on every new converter. An independent Protenix port
(ChoongHwanLee, chlee19990109-cloud/ColabFold, `colabfold2-protenix-proof`) asserts every one of
its 4174 source tensors is consumed exactly once; this is that check.

HOW IT WORKS, and its one caveat. The state dict is wrapped so `__getitem__` and
`__contains__ record` the key. Two converters instead SCAN with `sd.items()`
(`protenix2._p_atom_transformer`, `opendde` at its atom encoder), and a scan
cannot be attributed key-by-key -- so scanned keys are resolved by VALUE: a
checkpoint tensor whose sorted elements appear in some converted array is
counted as consumed. That second pass is why the report distinguishes
`never read` from `read`, and why it is exact rather than an over-count.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


class _Watched(dict):
  """A state dict that records which keys were addressed by name."""

  def __init__(self, d):
    super().__init__(d)
    self.seen: set[str] = set()
    self.scanned = False

  def __contains__(self, k):
    self.seen.add(k)
    return dict.__contains__(self, k)

  def __getitem__(self, k):
    self.seen.add(k)
    return dict.__getitem__(self, k)

  def items(self):
    self.scanned = True
    return dict.items(self)

  def keys(self):
    self.scanned = True
    return dict.keys(self)


def _value_index(params):
  """{n_elements: [sorted-value arrays]} over every converted leaf.

  Keyed by size so the comparison below is a handful of candidates, not a scan
  of the whole tree. Stacked leaves (a layer_stack holds all N blocks in one
  array) are indexed per slice as well, since that is the shape a scanned
  per-block tensor ends up in.
  """
  index: dict[int, list[np.ndarray]] = {}

  def add(a):
    a = np.asarray(a)
    if a.size:
      index.setdefault(a.size, []).append(np.sort(a.ravel()))

  def add_all(v):
    """Index a leaf and every FUSED piece a converter may have built it from.

    Three fusions in this codebase hide a source tensor inside a bigger array,
    and a whole-leaf comparison misses all three:
      * SwiGLU transitions concatenate operands a|b along the last axis
        (`common.concat_ab`), so index both halves;
      * triangle multiplication INTERLEAVES a and b as even/odd output columns
        (`interleave_ab`), so index both strides;
      * a layer_stack holds every block in one leading axis, so index slices.
    """
    add(v)
    if v.ndim >= 1 and v.shape[0] <= 64:            # layer_stack axis
      for i in range(v.shape[0]):
        add(v[i])
    if v.ndim >= 1 and v.shape[-1] % 2 == 0:
      h = v.shape[-1] // 2
      add(v[..., :h]); add(v[..., h:])              # concat_ab
      add(v[..., 0::2]); add(v[..., 1::2])          # interleave_ab

  def walk(v, depth):
    """Index a leaf and every leading-axis slice, to `depth` levels.

    TWO levels, not one: openfold3/opendde stack their diffusion transformer as
    (n_super, per_super, ...) via `stack_super`, so a single block's weight is
    v[i][j] and a one-level scan reports every one of them as missing.
    """
    add_all(v)
    if depth and v.ndim >= 2 and v.shape[0] <= 64:
      for i in range(v.shape[0]):
        walk(v[i], depth - 1)

  for scope, leaves in params.items():
    if scope == '__meta__':
      continue
    # Mappers come in two shapes and both are legitimate: NESTED
    # {scope: {name: array}}, which is what save_af3_params takes, and FLAT
    # {'scope/name': array}, which is what the ESMFold2 mapper emits because it
    # assembles one dict across four top-level scopes. Take either.
    if isinstance(leaves, dict):
      for _, v in leaves.items():
        walk(np.asarray(v), 2)
    else:
      walk(np.asarray(leaves), 2)
  return index


def audit(model: str, sd: dict, params: dict, verbose: bool = False) -> list[str]:
  """Return the checkpoint keys no converted parameter accounts for."""
  by_name = {k for k in sd if k in getattr(sd, 'seen', ())}
  candidates = sorted(k for k in sd if k not in by_name)
  if not candidates:
    return []

  index = _value_index(params)
  missing = []
  for k in candidates:
    v = np.sort(np.asarray(sd[k]).ravel())
    hits = index.get(v.size, ())
    if not any(np.array_equal(v, h) for h in hits):
      missing.append(k)
  return missing


# (module, loader, default checkpoint). A model absent here has no single-file
# loader -- chai-1 ships several TorchScript archives, not one state dict.
_LOADERS = {
    'protenix2': ('protenix2', 'load_protenix_checkpoint',
                  '/home/ubuntu/protenix_weights/protenix-v2.pt'),
    'openfold3': ('openfold3', 'load_of3_checkpoint',
                  '/home/ubuntu/of3-p2-155k.pt'),
    'openbind0': ('openfold3', 'load_of3_checkpoint',
                 '/home/ubuntu/of3-ob-174k.pt'),
    'rosettafold3': ('rosettafold3', 'load_rf3_checkpoint',
                     '/home/ubuntu/rf3_weights/'
                     'rf3_foundry_01_24_latest_remapped.ckpt'),
    'intellifold2': ('intellifold2', None,
                     '/home/ubuntu/model_v2/intellifold_v2.pt'),
    'boltz2': ('boltz2', None,
               '/home/ubuntu/boltz2_weights/boltz2_conf.ckpt'),
    'opendde': ('opendde', None, '/home/ubuntu/opendde_weights/opendde.pt'),
    # the ESMFold2 family: one loader, the variant chosen by the directory
    **{m: ('esmfold2', 'load_esmfold2_checkpoint',
           '/home/ubuntu/esmfold2_variants/%s' % hub)
       for m, hub in (
           ('esmfold2', 'ESMFold2'),
           ('esmfold2_fast', 'ESMFold2-Fast'),
           ('esmfold2_exp', 'ESMFold2-Experimental'),
           ('esmfold2_exp_fast', 'ESMFold2-Experimental-Fast'),
           ('esmfold2_exp_cutoff2025', 'ESMFold2-Experimental-Cutoff2025'),
           ('esmfold2_exp_fast_cutoff2025',
            'ESMFold2-Experimental-Fast-Cutoff2025'),
           ('esmfold2_lm600m',
            'ESMFold2-Experimental-Fast-base600M-step1500k'))},
}

_MAPPERS = {
    'protenix2': 'map_protenix2_to_af3',
    'openfold3': 'map_openfold3_to_af3',
    'openbind0': 'map_openfold3_to_af3',
    'rosettafold3': 'map_rosettafold3_to_af3',
    'intellifold2': 'map_intellifold2_to_af3',
    'boltz2': 'map_boltz2_to_af3',
    'opendde': 'convert_opendde',
    **{m: 'map_esmfold2_to_af3_graph' for m in (
        'esmfold2', 'esmfold2_fast', 'esmfold2_exp', 'esmfold2_exp_fast',
        'esmfold2_exp_cutoff2025', 'esmfold2_exp_fast_cutoff2025',
        'esmfold2_lm600m')},
}


def _plain_load(path):
  """torch.load for the converters that take a state dict directly."""
  import torch
  sd = torch.load(path, map_location='cpu', weights_only=False)
  for key in ('model', 'state_dict', 'ema_state_dict'):
    if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
      sd = sd[key]
      break
  # numpy, not torch: the converters index and transpose with numpy semantics
  # (`a.transpose(2, 0, 1)` is a TypeError on a torch tensor). Strip a DDP
  # `module.` prefix here too -- opendde's converter only strips it when handed
  # torch tensors, so converting first would hide every key behind it.
  out = {}
  for k, v in sd.items():
    if not hasattr(v, 'shape'):
      continue
    out[k[len('module.'):] if k.startswith('module.') else k] = (
        v.detach().cpu().float().numpy())
  return out


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--checkpoint', default=None)
  ap.add_argument('--verbose', action='store_true')
  args = ap.parse_args(argv)

  import importlib

  mod_name, loader_name, default_ckpt = _LOADERS[args.model]
  mod = importlib.import_module(f'converters.{mod_name}')
  path = args.checkpoint or default_ckpt
  sd_raw = (getattr(mod, loader_name)(path) if loader_name
            else _plain_load(path))
  sd = _Watched(sd_raw)
  params = getattr(mod, _MAPPERS[args.model])(sd)

  missing = audit(args.model, sd, params, args.verbose)

  # A converter may declare tensors it deliberately does not map, each with a
  # reason. Those are ACCOUNTED FOR, not missing: the number worth reporting is
  # the count nobody has thought about.
  import re
  dead = getattr(mod, 'DEAD_TENSORS', ())
  explained, unexplained = [], []
  for k in missing:
    reason = next((r for pat, r in dead if re.search(pat, k)), None)
    (explained if reason else unexplained).append((k, reason))

  print(f'{args.model}: {len(sd_raw)} checkpoint tensors, '
        f'{len(unexplained)} unaccounted for'
        + (f' ({len(explained)} deliberately unmapped)' if explained else ''))
  for k, _ in unexplained:
    print(f'  {k}  {tuple(np.asarray(sd_raw[k]).shape)}')
  if explained and args.verbose:
    print('  deliberately unmapped:')
    for k, reason in explained:
      print(f'    {k}\n        {reason}')
  return 1 if unexplained else 0


if __name__ == '__main__':
  sys.exit(main())
