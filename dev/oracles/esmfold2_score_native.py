"""Score dev/oracles/esmfold2_native_msa.py's dump with the IN-REPO reference.

`modality_check.reference()` handles 1STP's numbering (its chain A does not
start at residue 1 and has gaps); a hand-rolled CA list does not, which is why
the same coordinates read 18.8 A there and are scored properly here.
"""
import os, sys
import numpy as np
for p in ('/home/ubuntu/alphafold3/dev/oracles', '/home/ubuntu/alphafold3/src',
          '/home/ubuntu/alphafold3'):
    sys.path.insert(0, p)
sys.argv = sys.argv[:1]
import modality_check as mc

z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dumps', 'esmfold2_native_1stp.npz'), allow_pickle=True)
case = mc.CASES['ligand_1stp']
ref_seq, ref_atoms = mc.reference(case)
# `reference` returns [(seq_id, letter)] and {(seq_id, atom_name): xyz}; the CA
# list has to be built through the seq_ids, which is exactly the registration a
# hand-rolled parse gets wrong.
nat = np.array([ref_atoms[(sid, 'CA')] for sid, _ in ref_seq
                if (sid, 'CA') in ref_atoms])
print('in-repo reference: %d residues, %d with a CA' % (len(ref_seq), len(nat)))
for k in z.files:
    if not k.startswith('ca|'):
        continue
    x = np.asarray(z[k])
    n = min(len(x), len(nat))
    print('  native %-22s CA-RMSD %.3f A (in-repo reference, %d atoms)'
          % (k[3:], mc.kabsch(x[:n], nat[:n])[0], n))
