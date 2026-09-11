"""C1'-RMSD of native protenix samples against 1EHZ, scored like modality_check."""
import sys, glob
import numpy as np
sys.path.insert(0, 'dev/oracles')
import modality_check as mc

def _name(r):
  # mmCIF DOUBLE-quotes an atom name containing a prime, so the token is
  # `"C1'"`. Strip ONLY the double quotes: stripping the single quote too takes
  # the prime with it and leaves C1, which matches nothing.
  return r.get('label_atom_id', '').strip('"')


ref_rows = [r for r in mc.read_cif_atoms(sys.argv[1])
            if _name(r) == "C1'" and r.get('label_asym_id') == 'A']
ref = {int(r['label_seq_id']): [float(r['Cartn_x']), float(r['Cartn_y']),
                               float(r['Cartn_z'])] for r in ref_rows}
best = None
for f in sorted(glob.glob(sys.argv[2])):
  pred = {}
  for r in mc.read_cif_atoms(f):
    if _name(r) == "C1'":
      pred.setdefault(int(r['label_seq_id']),
                      [float(r['Cartn_x']), float(r['Cartn_y']), float(r['Cartn_z'])])
  common = sorted(set(ref) & set(pred))
  v = mc.kabsch(np.array([pred[i] for i in common]),
                np.array([ref[i] for i in common]))[0]
  print('  %-30s %3d C1\'  %.3f' % (f.split('/')[-1], len(common), v))
  best = v if best is None else min(best, v)
print('best %.3f' % best)
