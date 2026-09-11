"""Fold with NATIVE's trunk injected, to settle trunk-vs-downstream.

Our openbind0 folds ubiquitin to 10.4 A where native gets 2.4. Every module
comparison so far sits at or near native's own TF32 floor, so the module gates
cannot say whether the fault is in the trunk or after it. This substitutes
native's own (s_trunk, z_trunk) into our diffusion head and folds:

  * lands near 2.4 A  -> the fault is the TRUNK (and the module gates are
    blind to it, because openbind0's trunk is chaotic: native's own tf32 change
    moves z by max|d| 200 on rms 149).
  * stays near 10 A   -> the fault is DOWNSTREAM: diffusion, heads, or the atom
    layout they read.

INJECT=none reruns the same path untouched, as the control.
"""
import os, sys
import numpy as np
sys.path.insert(0, 'dev/oracles')

model, npz, cif = sys.argv[1], sys.argv[2], sys.argv[3]
inject = os.environ.get('INJECT', 'both')
d = np.load(npz)
nat_s = d['s_trunk'][0].astype(np.float32)
nat_z = d['z_trunk'][0].astype(np.float32)

import fold_check
import modality_check as mc
from alphafold3.model.network import diffusion_head

_orig = diffusion_head.DiffusionHead.__call__


def patched(self, positions_noisy, noise_level, batch, embeddings, *a, **kw):
  if inject != 'none':
    import jax.numpy as jnp
    e = dict(embeddings)
    if inject in ('both', 'pair'):
      e['pair'] = jnp.asarray(nat_z, e['pair'].dtype)
    if inject in ('both', 'single'):
      e['single'] = jnp.asarray(nat_s, e['single'].dtype)
    embeddings = e
  return _orig(self, positions_noisy, noise_level, batch, embeddings, *a, **kw)


diffusion_head.DiffusionHead.__call__ = patched

seq = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
out, batch = fold_check.fold(model, seq, seed=0)

# score CA against the reference cif, matching on the batch's own residue index
ref_rows = [r for r in mc.read_cif_atoms(cif)
            if r.get('label_atom_id') == 'CA' and r.get('label_asym_id') == 'A'
            and r.get('label_alt_id') in ('.', '?', '', 'A')]
ref = {int(r['label_seq_id']): [float(r['Cartn_x']), float(r['Cartn_y']),
                                float(r['Cartn_z'])] for r in ref_rows}
pos = np.asarray(out['diffusion_samples']['atom_positions'])
mask = np.asarray(batch['pred_dense_atom_mask'])
print('shapes: pos', pos.shape, 'mask', mask.shape, 'res', np.asarray(batch['residue_index']).shape)
names = batch.get('ref_atom_name_chars')
# CA is atom index 1 in the dense per-token layout for a standard residue
res = np.asarray(batch['residue_index'])
rmsds = []
for si in range(pos.shape[0]):
  a, b = [], []
  for t in range(pos.shape[1]):
    # NO +1. Our `residue_index` is already the reference's numbering for a
    # plain protein chain, and the +1 that stood here shifted every residue by
    # one: on a compact 76-mer that costs ~2-3 A after superposition, which is
    # indistinguishable from a bad fold. The tell was matching 75 CA where the
    # native scorer matched 76. It made protenix1's with-MSA fold read 4.02
    # against native's 1.87 when the truth is 1.82, and sent me through the
    # whole trunk-and-denoiser ladder looking for a gap that was not there.
    ri = int(res[t])
    if ri in ref and mask[t][1]:
      a.append(pos[si, t, 1]); b.append(ref[ri])
  rmsds.append(mc.kabsch(np.array(a), np.array(b))[0])
print('INJECT=%s  %d CA  CA-RMSD %s  best %.3f'
      % (inject, len(a), ' '.join('%.3f' % r for r in rmsds), min(rmsds)))
