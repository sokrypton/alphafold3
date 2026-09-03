"""Did the model build the D-amino acids as D, and did the cyclic wrap help?

5KX0 is a de novo cyclic peptide with 11 D-residues among 26. The learned
L-prior is WRONG for eleven of them, which makes it the one target where a
chirality signal has to earn its keep -- and where a gap shows:

    native RoseTTAFold3   15/15 L kept, 11/11 D built as D, 1.5-1.8 A
    this port             15/15 L kept,  2/11 D built as D, 2.3 A

    python scripts/dl_chirality_score.py <tag> [<tag> ...]

Reads $S/dl_<tag>/5kx0/. The sign of the CA signed volume separates the two
configurations cleanly: L gives +0.77 in the CCD ideal, D gives -0.77.
"""
import json, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/ubuntu/alphafold3/src')
sys.path.insert(0, '/home/ubuntu/alphafold3/scripts')
import numpy as np
from Bio.PDB import MMCIFParser
from chirality_check import ccd_ideal, centres, signed_volume

S = '/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/run'
D_CODES = {'DAL','DAR','DAS','DCY','DGL','DGN','DHI','DIL','DLE','DLY','DPN',
           'DPR','DSN','DTH','DTR','DTY','DVA'}

def ca_signs(path):
  """-> {(resnum, resname): sign of the CA signed volume}."""
  st = MMCIFParser(QUIET=True).get_structure('x', path)[0]
  out = {}
  for ch in st:
    for r in ch:
      name = r.get_resname().strip()
      atoms = {a.get_name().strip(): np.asarray(a.coord) for a in r}
      if not {'N', 'CA', 'C', 'CB'} <= set(atoms):
        continue                       # glycine and anything incomplete
      out[(r.id[1], name)] = np.sign(signed_volume(atoms, 'CA', ['C', 'CB', 'N']))
  return out

def expected():
  want = {}
  for code in ('ALA','ARG','ASN','ASP','CYS','GLN','GLU','HIS','ILE','LEU','LYS',
               'MET','PHE','PRO','SER','THR','TRP','TYR','VAL') + tuple(D_CODES):
    ref = ccd_ideal(code)
    if {'N','CA','C','CB'} <= set(ref):
      want[code] = np.sign(signed_volume(ref, 'CA', ['C','CB','N']))
  return want

def rmsd(P, Q):
  P = P - P.mean(0); Q = Q - Q.mean(0)
  V, _, W = np.linalg.svd(P.T @ Q)
  d = np.sign(np.linalg.det(V @ W))
  return float(np.sqrt(((P @ (V @ np.diag([1,1,d]) @ W) - Q) ** 2).sum(1).mean()))

def ca_trace(path):
  st = MMCIFParser(QUIET=True).get_structure('x', path)[0]
  return np.array([r['CA'].coord for ch in st for r in ch if 'CA' in r])

want = expected()
ref_ca = ca_trace('/home/ubuntu/5KX0.cif')
print(f'{"run":16s} {"L kept":>8s} {"D built as D":>13s} {"CA-RMSD":>9s} {"ptm":>5s} {"pLDDT":>6s}')
for tag in sys.argv[1:]:
  base = f'{S}/dl_{tag}/5kx0'
  path = f'{base}/5kx0_model.cif'
  if not os.path.exists(path):
    print(f'{tag:16s} {"(pending)":>8s}'); continue
  signs = ca_signs(path)
  l_ok = l_tot = d_ok = d_tot = 0
  wrong = []
  for (num, name), got in sorted(signs.items()):
    if name not in want:
      continue
    if name in D_CODES:
      d_tot += 1; d_ok += got == want[name]
    else:
      l_tot += 1; l_ok += got == want[name]
    if got != want[name]:
      wrong.append(f'{name}{num}')
  x = ca_trace(path)
  n = min(len(x), len(ref_ca))
  s = json.load(open(f'{base}/5kx0_summary_confidences.json'))
  c = json.load(open(f'{base}/5kx0_confidences.json'))
  print(f'{tag:16s} {f"{l_ok}/{l_tot}":>8s} {f"{d_ok}/{d_tot}":>13s} '
        f'{rmsd(x[:n], ref_ca[:n]):9.2f} {s["ptm"]:5.2f} {np.mean(c["atom_plddts"]):6.1f}'
        + (f'   wrong: {",".join(wrong[:6])}' if wrong else ''))
