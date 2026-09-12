"""L7: the OUTPUT side -- what we hand back, not what the network computes.

Everything above L6 stops at a coordinate array. Between that array and the file
a user reads sit two conversions nothing gates:

  * the gather from the model's per-token dense atom layout into the flat mmCIF
    layout, which SILENTLY writes (0, 0, 0) for any atom it cannot find and only
    logs a warning (model.py: `missing_atoms_indices`). An input convention that
    drops or renames an atom therefore reaches the user as an atom at the
    origin, not as an error -- and the drop conventions differ per model
    (opendde keeps the terminal OXT; four others drop it).
  * the confidence columns. pLDDT goes into the b-factor column, and whether it
    is 0-1 or 0-100, per-atom or per-token, is a per-vendor convention.

This checks both by ROUND-TRIPPING: fold, write the mmCIF, parse it back, and
compare the parsed file against the batch that produced it.

    PYTHONPATH=src:.:dev/oracles python dev/oracles/output_parity.py [models...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_check                                        # noqa: E402

from alphafold3.common import folding_input              # noqa: E402
from alphafold3.model import model as model_lib          # noqa: E402

CASES = {
    'monomer': '/home/ubuntu/5k9p_nomsa.json',
    'ligand': '/home/ubuntu/ligand_ours.json',
    'dimer': '/home/ubuntu/dimer_ours.json',
}


def check(model, case, path):
  from alphafold3 import structure
  from alphafold3.model import feat_batch

  chains = list(folding_input.Input.from_json(open(path).read()).chains)
  seq = getattr(chains[0], 'sequence', '')
  result, batch = fold_check.fold(model, seq, chains=chains)
  coords = np.asarray(result['diffusion_samples']['atom_positions'])[0]
  plddt = result.get('predicted_lddt')
  if plddt is not None:
    plddt = np.asarray(plddt)
    plddt = plddt[0] if plddt.ndim == 3 else plddt
  # The output layouts are OBJECT arrays, which _fold_setup strips via
  # remove_invalidly_typed_feats before handing the batch to jnp -- so the
  # output side is only reachable through the RAW batch it keeps beside it.
  raw = fold_check.LAST_RAW_BATCH
  fb = feat_batch.Batch.from_data_dict(raw)
  struc = model_lib.predicted_structure_from_coords(coords, fb,
                                                    predicted_lddt=plddt)

  fails = []
  xyz = np.stack([struc.atom_x, struc.atom_y, struc.atom_z], -1)
  n_zero = int((np.abs(xyz).sum(-1) == 0).sum())
  if n_zero:
    fails.append('%d atoms at the origin (the gather dropped them)' % n_zero)

  # the mmCIF must round-trip to the same atoms
  cif = struc.to_mmcif()
  back = structure.from_mmcif(cif)
  if back.num_atoms != struc.num_atoms:
    fails.append('round-trip atom count %d != %d'
                 % (back.num_atoms, struc.num_atoms))
  else:
    for f in ('atom_name', 'res_id', 'res_name', 'chain_id'):
      if not np.array_equal(np.asarray(getattr(struc, f)),
                            np.asarray(getattr(back, f))):
        fails.append('round-trip %s differs' % f)
    d = float(np.abs(xyz - np.stack(
        [back.atom_x, back.atom_y, back.atom_z], -1)).max())
    if d > 5e-3:                          # mmCIF writes 3 decimals
      fails.append('round-trip coords max|d| %.4g' % d)

  # the b-factor column has to be a pLDDT on ONE scale
  bf = np.asarray(struc.atom_b_factor)
  if plddt is not None and (bf.min() < 0 or bf.max() > 100 or bf.max() <= 1.0):
    fails.append('b-factor range [%.3f, %.3f] is not a 0-100 pLDDT'
                 % (bf.min(), bf.max()))

  # every real atom of the batch has to reach the file
  n_real = int((np.asarray(raw['pred_dense_atom_mask']) > 0).sum())
  if struc.num_atoms != n_real:
    fails.append('wrote %d atoms, batch has %d real' % (struc.num_atoms, n_real))

  print('  %-14s %-8s %-5s %s'
        % (model, case, struc.num_atoms,
           'OK' if not fails else 'FAIL: ' + '; '.join(fails)))
  return not fails


def main():
  models = sys.argv[1:] or ['alphafold3']
  ok = True
  for m in models:
    for case, path in CASES.items():
      try:
        ok &= check(m, case, path)
      except Exception as e:                              # noqa: BLE001
        print('  %-14s %-8s ERROR %s' % (m, case, str(e)[:200]))
        ok = False
  return 0 if ok else 1


if __name__ == '__main__':
  raise SystemExit(main())
