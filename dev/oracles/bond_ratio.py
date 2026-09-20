"""Mean bond ratio of an atomised entity against the CCD ideals.

The measure ESMFOLD2_PTM.md / BOLTZ2_PTM.md use: for every bond the CCD defines
for a component, the folded distance over the ideal distance. 1.000 is perfect.

TWO CONTROLS ARE MANDATORY, and both come from the SAME structure:
  * the backbone bonds of the unmodified residues -- if those are not ~1.00 the
    fold is not worth reading and no conclusion about the modification follows;
  * a plain CCD ligand, when the job has one -- a ligand is atomised exactly as
    a modified residue is but sits in its own chain, so it separates "this port
    cannot place atomised things" from "this port cannot place an atomised
    residue INSIDE a polymer".

ATOM NAMES ARE MATCHED BY NAME, NEVER BY ORDER. The brief lost a day to
slicing the CCD's atom list to the first ten, which keeps OXT -- a leaving atom
a mid-chain residue drops -- shifting every later name by one and measuring
OG-P against the phosphorus's neighbour. That read 1.101 and looked like "the
vendor is imperfect too".

  PYTHONPATH=src:. python dev/oracles/bond_ratio.py <cif> [COMP ...]
"""
import collections
import sys

import numpy as np


def read_cif_atoms(path):
  """-> [(chain, resseq, resname, atomname, xyz)] from an mmCIF atom_site loop."""
  rows, cols, in_loop = [], [], False
  for line in open(path):
    s = line.strip()
    if s.startswith('_atom_site.'):
      cols.append(s.split('.', 1)[1]); in_loop = True; continue
    if in_loop:
      if s.startswith('#') or not s:
        in_loop = False; continue
      f = s.split()
      if len(f) < len(cols):
        continue
      r = dict(zip(cols, f))
      rows.append((r.get('label_asym_id') or r.get('auth_asym_id'),
                   r.get('label_seq_id') or r.get('auth_seq_id'),
                   r.get('label_comp_id'), r.get('label_atom_id'),
                   np.array([float(r['Cartn_x']), float(r['Cartn_y']),
                             float(r['Cartn_z'])])))
  return rows


def main(path, comps):
  from alphafold3.constants import decoded_ccd

  rows = read_cif_atoms(path)
  by_res = collections.OrderedDict()
  for chain, seq, comp, atom, xyz in rows:
    by_res.setdefault((chain, seq, comp), {})[atom] = xyz
  print(f'{len(rows)} atoms, {len(by_res)} residues/components')

  ccd = decoded_ccd.get_ccd()
  def ideals(comp):
    c = ccd.get(comp)
    if c is None:
      return None
    names = c['_chem_comp_atom.atom_id']
    xyz = {n: np.array([float(x), float(y), float(z)]) for n, x, y, z in zip(
        names, c['_chem_comp_atom.pdbx_model_Cartn_x_ideal'],
        c['_chem_comp_atom.pdbx_model_Cartn_y_ideal'],
        c['_chem_comp_atom.pdbx_model_Cartn_z_ideal'])}
    out = []
    for a, b in zip(c['_chem_comp_bond.atom_id_1'], c['_chem_comp_bond.atom_id_2']):
      if a in xyz and b in xyz:
        out.append((a, b, float(np.linalg.norm(xyz[a] - xyz[b]))))
    return out

  # the control: N-CA, CA-C, C-O of every standard residue in the structure
  ctrl = []
  for (chain, seq, comp), atoms in by_res.items():
    if comp in comps:
      continue
    for a, b in (('N', 'CA'), ('CA', 'C'), ('C', 'O')):
      bi = ideals(comp)
      if bi is None or a not in atoms or b not in atoms:
        continue
      ideal = next((d for x, y, d in bi if {x, y} == {a, b}), None)
      if ideal:
        ctrl.append(np.linalg.norm(atoms[a] - atoms[b]) / ideal)
  if ctrl:
    print(f'control  {len(ctrl):4d} backbone bonds   mean ratio {np.mean(ctrl):.3f}')

  for comp in comps:
    bi = ideals(comp)
    if bi is None:
      print(f'{comp}: not in the CCD here'); continue
    for (chain, seq, c), atoms in by_res.items():
      if c != comp:
        continue
      rs = [(a, b, np.linalg.norm(atoms[a] - atoms[b]) / d)
            for a, b, d in bi if a in atoms and b in atoms]
      if not rs:
        continue
      worst = max(rs, key=lambda r: abs(r[2] - 1))
      print(f'{comp} chain {chain} res {seq}: {len(rs)} bonds  '
            f'mean ratio {np.mean([r[2] for r in rs]):.3f}   '
            f'worst {worst[0]}-{worst[1]} {worst[2]:.3f}')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2:] or ['SEP', 'GOL'])
