"""CA-RMSD of native OpenFold3/OpenBind mmCIF samples against a reference cif.

Reuses modality_check's own reader and kabsch so the number is directly
comparable to the L6 row it is meant to answer.
"""
import sys, glob, numpy as np
sys.path.insert(0, 'dev/oracles')
import modality_check as mc

ref_cif, pattern, chain = sys.argv[1], sys.argv[2], sys.argv[3]

def cas(path, ch=None):
  out = {}
  for r in mc.read_cif_atoms(path):
    if r.get('label_atom_id') != 'CA':
      continue
    if ch is not None and r.get('label_asym_id') != ch:
      continue
    if r.get('label_alt_id') not in ('.', '?', '', 'A'):
      continue
    sid = int(r['label_seq_id'])
    out.setdefault(sid, [float(r['Cartn_x']), float(r['Cartn_y']), float(r['Cartn_z'])])
  return out

ref = cas(ref_cif, chain)
best = None
for f in sorted(glob.glob(pattern)):
  pred = cas(f)
  common = sorted(set(ref) & set(pred))
  a = np.array([pred[i] for i in common]); b = np.array([ref[i] for i in common])
  r = mc.kabsch(a, b)[0]
  print('  %-22s %3d CA  CA-RMSD %.3f' % (f.rsplit('_', 2)[-2] + '_' + f.rsplit('_', 2)[-1].split('.')[0], len(common), r))
  best = r if best is None else min(best, r)
print('best %.3f' % best)
