"""Check a prediction's stereochemistry against the CCD ideal conformer.

RMSD cannot see a wrong stereocentre: inverting one barely moves an atom, so a
structure gate reads clean while the molecule is a different one. Inverting two
of biotin's three centres cost 0.81 A of ligand RMSD, which reads as a nudge.

    python scripts/chirality_check.py --codes BTN prediction.cif
    python scripts/chirality_check.py --codes A,G,C,U rna.cif

Three kinds of false positive have to be excluded, and each of them was found by
reading the numbers rather than trusting the count:

  * A stereocentre shows only THREE heavy neighbours here, not four -- these
    files carry no hydrogens. Requiring four finds nothing at all.
  * A planar atom spans no volume, so the sign of what it does span is noise.
    Biotin's three real centres come out at 0.69-0.80 normalised volume and its
    ureido carbonyl at 0.002, which flipped from run to run.
  * Symmetry-equivalent substituents are not stereochemistry: a phosphate's OP1
    and OP2 are the same atom, and 76 of a tRNA's phosphates read as inverted
    until they were excluded. That also needs the ideal conformer's hydrogens
    dropped, or OP2 looks substituted rather than terminal.
"""

import argparse
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from Bio.PDB import MMCIFParser


def ligand_atoms(path, resname):
  st = MMCIFParser(QUIET=True).get_structure('x', path)[0]
  for ch in st:
    for r in ch:
      if r.get_resname().strip() == resname:
        return {a.get_name().strip(): np.asarray(a.coord) for a in r}
  return {}


def centres(atoms, bond_cut=1.9):
  """Candidate stereocentres: atoms with three or more heavy neighbours.

  Three, not four: these structures carry no hydrogens, so a tetrahedral centre
  shows only its three heavy substituents. The sign of the volume they span
  still fixes which side the fourth (the hydrogen) is on, as long as the three
  are taken in the same order on both sides -- hence the sort by atom name.
  """
  names = sorted(atoms)
  neighbours = {n: sorted(m for m in names
                          if m != n and np.linalg.norm(atoms[n] - atoms[m]) < bond_cut)
                for n in names}
  out = {}
  for n, near in neighbours.items():
    if len(near) < 3:
      continue
    # Skip centres with two interchangeable substituents. A phosphate's OP1 and
    # OP2 are the same atom by symmetry, so "inverting" it only means the two
    # names swapped -- 76 of tRNA's phosphates read as flipped stereocentres
    # before this, which is a naming artefact and not chemistry.
    terminal = [m for m in near if len(neighbours[m]) == 1]
    elements = [_element(m) for m in terminal]
    if len(elements) != len(set(elements)):
      continue
    out[n] = near[:4]
  return out


def _element(name):
  """The element of a PDB atom name -- its leading alphabetic characters."""
  return ''.join(c for c in name if c.isalpha())[:1]


def signed_volume(atoms, centre, neighbours, normalise=False):
  a, b, c = (atoms[n] - atoms[centre] for n in neighbours[:3])
  v = float(np.dot(np.cross(a, b), c))
  if normalise:
    v /= (np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c)) or 1.0
  return v


# A planar centre spans no volume, so the sign of what it does span is noise.
# Biotin's three real stereocentres come out at 0.69-0.80 normalised; its ureido
# carbonyl at 0.002. Anything under this is not a stereocentre.
TETRAHEDRAL = 0.5


def compare(pred_path, ref_path, resname):
  pred, ref = ligand_atoms(pred_path, resname), ligand_atoms(ref_path, resname)
  shared = set(pred) & set(ref)
  if not shared:
    return None
  ref_centres = {n: [x for x in nb if x in shared]
                 for n, nb in centres(ref).items() if n in shared}
  agree, total, flipped = 0, 0, []
  for n, nb in ref_centres.items():
    if len(nb) < 3 or not set(nb) <= set(pred):
      continue
    if abs(signed_volume(ref, n, nb, normalise=True)) < TETRAHEDRAL:
      continue                      # planar: not a stereocentre
    total += 1
    same = np.sign(signed_volume(pred, n, nb)) == np.sign(signed_volume(ref, n, nb))
    agree += bool(same)
    if not same:
      flipped.append(n)
  return agree, total, flipped


def ccd_ideal(code):
  from alphafold3.constants.decoded_ccd import get_ccd

  entry = get_ccd().get(code)
  if entry is None:
    return {}
  names = entry['_chem_comp_atom.atom_id']
  cols = ('_chem_comp_atom.pdbx_model_Cartn_x_ideal',
          '_chem_comp_atom.pdbx_model_Cartn_y_ideal',
          '_chem_comp_atom.pdbx_model_Cartn_z_ideal')
  if not all(c in entry for c in cols):
    return {}
  # Heavy atoms only. The ideal conformer carries hydrogens and the predictions
  # do not, and leaving them in makes a phosphate's OP2 look like a substituted
  # oxygen rather than a terminal one -- so the symmetry filter never fires and
  # all 76 of a tRNA's phosphates read as inverted stereocentres.
  elements = entry.get('_chem_comp_atom.type_symbol', [''] * len(names))
  out = {}
  for i, n in enumerate(names):
    if elements[i].strip().upper() in ('H', 'D'):
      continue
    try:
      out[n.strip()] = np.array([float(entry[c][i]) for c in cols])
    except ValueError:
      pass
  return out


def residues(path, codes):
  st = MMCIFParser(QUIET=True).get_structure('x', path)[0]
  found = []
  for ch in st:
    for r in ch:
      name = r.get_resname().strip()
      if name in codes:
        found.append((name, f'{ch.id}{r.id[1]}',
                      {a.get_name().strip(): np.asarray(a.coord) for a in r}))
  return found


def check(path, codes):
  agree = total = 0
  flipped = []
  for name, tag, atoms in residues(path, codes):
    ref = ccd_ideal(name)
    if not ref:
      continue
    shared = set(atoms) & set(ref)
    for c, nb in centres(ref).items():
      nb = [x for x in nb if x in shared]
      if c not in shared or len(nb) < 3:
        continue
      if abs(signed_volume(ref, c, nb, normalise=True)) < TETRAHEDRAL:
        continue
      total += 1
      if np.sign(signed_volume(atoms, c, nb)) == np.sign(signed_volume(ref, c, nb)):
        agree += 1
      else:
        flipped.append(f'{name}{tag}.{c}')
  return agree, total, flipped

def main():
  p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
  p.add_argument('--codes', required=True,
                 help='comma-separated CCD codes to check (e.g. BTN or A,G,C,U)')
  p.add_argument('structures', nargs='+')
  args = p.parse_args()
  codes = set(args.codes.split(','))
  print(f'{"structure":48s} {"stereocentres correct":>21s}  flipped')
  for path in args.structures:
    agree, total, flipped = check(path, codes)
    print(f'{path[-48:]:48s} {f"{agree}/{total}":>21s}  {",".join(flipped[:6]) or "-"}')


if __name__ == '__main__':
  main()
