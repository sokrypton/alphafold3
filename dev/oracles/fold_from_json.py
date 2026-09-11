"""Fold from a fold-input JSON -- so an MSA can be supplied -- and score CA-RMSD.

    PYTHONPATH=src:.:dev/oracles python dev/oracles/fold_from_json.py \
      <model> <input>.json <reference>.cif        # SEED=n to repeat

`modality_check` reads an MSA json only on its ligand case, and that case
appends a ligand. This exists for the question "is a bistable target still
bistable with an MSA?" -- plain 5K9P flips basin per process from a single
sequence, and with a 4-row MSA it stops, on both sides.

It prints an OFFSET SWEEP for a reason: our `residue_index` is already the
reference's numbering, and a +1 in this mapping shifts every residue by one,
which on a compact 76-mer costs ~2 A after superposition and reads exactly like
a bad fold. That bug (mine, in this file's first version) made protenix1's
with-MSA fold look 2.3 A worse than native and cost a long detour through the
trunk and the denoiser looking for a gap that was not there. The tell was in the
output the whole time: 75 CA matched where the native scorer matched 76.
"""

import os, sys
import numpy as np
sys.path.insert(0, 'dev/oracles')
import fold_check
import modality_check as mc
from alphafold3.common import folding_input

model, jpath, cif = sys.argv[1], sys.argv[2], sys.argv[3]
fi = folding_input.Input.from_json(open(jpath).read())
chains = list(fi.chains)
seq = chains[0].sequence
n_msa = chains[0].unpaired_msa.count('>') if getattr(chains[0], 'unpaired_msa', '') else 0
print('%s: %d residues, %d MSA rows' % (model, len(seq), n_msa))
out, batch = fold_check.fold(model, seq, chains=chains, seed=int(os.environ.get('SEED', 0)))
pos = np.asarray(out['diffusion_samples']['atom_positions'])
pl = out.get('predicted_lddt')
if pl is not None:
  pl = np.asarray(pl['predicted_lddt'] if isinstance(pl, dict) else pl)
  print('  mean pLDDT %.1f' % pl.mean())
ref_rows = [r for r in mc.read_cif_atoms(cif)
            if r.get('label_atom_id') == 'CA' and r.get('label_asym_id') == 'A'
            and r.get('label_alt_id') in ('.', '?', '', 'A')]
ref = {int(r['label_seq_id']): [float(r['Cartn_x']), float(r['Cartn_y']),
                               float(r['Cartn_z'])] for r in ref_rows}
res = np.asarray(batch['residue_index'])
mask = np.asarray(batch['pred_dense_atom_mask'])
rs = []
for si in range(pos.shape[0]):
  a, b = [], []
  for t in range(pos.shape[1]):
    ri = int(res[t])
    if ri in ref and mask[t][1]:
      a.append(pos[si, t, 1]); b.append(ref[ri])
  rs.append(mc.kabsch(np.array(a), np.array(b))[0])
print('  %d CA, CA-RMSD %s  best %.3f'
      % (len(a), ' '.join('%.3f' % r for r in rs), min(rs)))
# WHERE the error is: a disordered tail and a wrong fold look nothing alike, and
# native's own 1.87 A here is almost all the C-terminal LRGG tail (median
# per-residue deviation 0.53 A).
# OFFSET SWEEP. Our residue_index -> reference numbering mapping is an
# assumption, and an off-by-one on a compact 76-mer costs ~3-4 A after
# superposition -- indistinguishable from a bad fold unless it is checked. The
# tell that prompted this: we matched 75 CA where the native scorer matched 76.
si0 = int(np.argmin(rs))
for off in (-1, 0, 1, 2):
  A2, B2 = [], []
  for t in range(pos.shape[1]):
    ri = int(res[t]) + off
    if ri in ref and mask[t][1]:
      A2.append(pos[si0, t, 1]); B2.append(ref[ri])
  if len(A2) > 10:
    print('  offset %+d: %d CA, CA-RMSD %.3f'
          % (off, len(A2), mc.kabsch(np.array(A2), np.array(B2))[0]))
si = si0
ids, A, B = [], [], []
for t in range(pos.shape[1]):
  ri = int(res[t])
  if ri in ref and mask[t][1]:
    ids.append(ri); A.append(pos[si, t, 1]); B.append(ref[ri])
A, B = np.array(A), np.array(B)
r, rot, ca, cb = mc.kabsch(A, B)
dev = np.sqrt((((A - ca) @ rot - (B - cb)) ** 2).sum(-1))
order = np.argsort(-dev)[:6]
print('  per-residue deviation: median %.2f, 90th %.2f, worst %s'
      % (np.median(dev), np.percentile(dev, 90),
         [(ids[i], round(float(dev[i]), 1)) for i in order]))
