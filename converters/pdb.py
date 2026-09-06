"""Minimal PDB reading for the converters.

`parse_ca` lived in the oracle harnesses, which are dev-only and out of tree --
so a tracked module (esm_lm) importing it from there made the package depend
on files the repo does not ship. It is small and has one careful behaviour worth
keeping in one place, so it lives here instead.
"""

import numpy as np

A3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
      'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
      'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
      'TYR': 'Y', 'VAL': 'V'}


def parse_ca(path, chain=None):
  """(sequence, CA coords). Dedupes altloc records -- 6MRR has three (LYS14,
  GLU43, ASP64), and counting them twice yields a 71-residue 68-mer."""
  seq, xyz, seen = [], [], set()
  for line in open(path):
    if not line.startswith('ATOM') or line[12:16].strip() != 'CA':
      continue
    if chain and line[21] != chain:
      continue
    key = line[21] + line[22:27]
    if key in seen:
      continue
    seen.add(key)
    res = line[17:20].strip()
    if res not in A3:
      continue
    seq.append(A3[res])
    xyz.append([float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)])
  return ''.join(seq), np.array(xyz)
