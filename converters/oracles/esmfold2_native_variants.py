"""Fold 6MRR with NATIVE ESMFold2, one variant at a time.

Two of the six converted variants fold badly (esmfold2_exp_fast 11.7 A,
esmfold2_exp_cutoff2025 7.7 A) while siblings with the SAME architecture fold
well. That is not something our own numbers can settle: it separates "the port
is wrong for these two" from "these two checkpoints are weaker", and only the
reference implementation can say which.

Runs in ~/venv_esm (torch + transformers + ESM-C), NOT the jax venv:

    ~/venv_esm/bin/python converters/oracles/esmfold2_native_variants.py \
        [ESMFold2-Experimental-Fast ...]
"""
import os
import sys

import numpy as np
import torch

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel)
from transformers.models.esmfold2.protein_utils import prepare_protein_features

# Upstream's own confirmation that the experimental line is a different model,
# not a config of the released one: its checkpoints declare
# ESMFold2ExperimentalModel, which lives in its own module and does not take
# esmc_precision.
_CLASSES = {'ESMFold2Model': ESMFold2Model,
            'ESMFold2ExperimentalModel': ESMFold2ExperimentalModel}

A3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
      'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
      'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
      'TYR': 'Y', 'VAL': 'V'}

DEFAULT = ['ESMFold2', 'ESMFold2-Experimental', 'ESMFold2-Experimental-Fast',
           'ESMFold2-Experimental-Cutoff2025',
           'ESMFold2-Experimental-Fast-Cutoff2025', 'ESMFold2-Fast']


def parse_ca(path):
  """(sequence, CA coords), altloc-deduped -- 6MRR has three."""
  seq, xyz, seen = [], [], set()
  for line in open(path):
    if not line.startswith('ATOM') or line[12:16].strip() != 'CA':
      continue
    key = line[21] + line[22:27]
    if key in seen:
      continue
    seen.add(key)
    seq.append(A3.get(line[17:20].strip(), 'X'))
    xyz.append([float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)])
  return ''.join(seq), np.asarray(xyz)


def kabsch(a, b):
  a = a - a.mean(0)
  b = b - b.mean(0)
  u, _, vt = np.linalg.svd(a.T @ b)
  d = np.sign(np.linalg.det(u @ vt))
  r = u @ np.diag([1.0, 1.0, d]) @ vt
  return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def main(argv):
  names = argv or DEFAULT
  seq, native = parse_ca(os.path.expanduser('~/6MRR.pdb'))
  print('6MRR: %d residues' % len(seq))
  feats0 = prepare_protein_features(seq)
  a2t = feats0['atom_to_token'][0].numpy().astype(int)
  msk = feats0['atom_attention_mask'][0].numpy().astype(bool)
  ch = feats0['ref_atom_name_chars'][0].numpy().astype(int)
  nm = lambda i: ''.join(chr(c + 32) for c in ch[i]).strip()
  rep = np.full(int(a2t[msk].max()) + 1, -1)
  for i in range(len(a2t)):
    if msk[i] and nm(i) == 'CA':
      rep[a2t[i]] = i

  for name in names:
    local = os.path.expanduser('~/esmfold2_variants/%s' % name)
    src = local if os.path.exists(local + '/config.json') else 'biohub/%s' % name
    try:
      import json
      arch = json.load(open(os.path.join(local, 'config.json')))['architectures'][0] \
          if os.path.exists(local + '/config.json') else 'ESMFold2Model'
      cls = _CLASSES[arch]
      kw = {'esmc_precision': 'bf16'} if cls is ESMFold2Model else {}
      m = cls.from_pretrained(src, **kw).cuda().eval()
      feats = {k: v.cuda() for k, v in prepare_protein_features(seq).items()}
      rs = []
      for seed in range(3):
        torch.manual_seed(seed)
        with torch.no_grad():
          o = m(**feats, num_loops=3, num_diffusion_samples=1,
                num_sampling_steps=200)
        x = o['sample_atom_coords'].float().cpu().numpy()
        if x.ndim == 3:
          x = x[0]
        rs.append(kabsch(x[rep], native))
      print('  %-42s CA-RMSD %s   best %.3f  plddt %.3f'
            % (name, ' '.join('%.3f' % r for r in rs), min(rs),
               float(o['plddt'].mean())), flush=True)
      del m
      torch.cuda.empty_cache()
    except Exception as err:  # pylint: disable=broad-except
      print('  %-42s FAILED %s: %s' % (name, type(err).__name__, err), flush=True)


if __name__ == '__main__':
  main(sys.argv[1:])
