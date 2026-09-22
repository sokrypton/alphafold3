"""Does a COVALENT ligand stay attached? The last unscreened capability.

`featurise_spec` had no way to express a covalent link at all until now: AF3's
Input carries `bonded_atom_pairs` and we never threaded it, so a covalently bound
inhibitor was featurised as a free ligand that merely happens to sit nearby.
Same gap shape as PTMs, and the same risk -- silent, and it looks like a
prediction.

CASE: 6LU7, SARS-CoV-2 main protease (306 aa) with the N3 inhibitor, which is a
SIX-COMPONENT ligand chain (02J, ALA, VAL, LEU, PJE, 010) covalently bonded to
CYS145 SG at 1.793 A. So it exercises multi-component ligands and covalency at
once. The links come from the structure's own `_struct_conn` records:

    A CYS 145 SG  --  B PJE 5 C20     1.793   (the protein-ligand bond)
    B 02J 1 C     --  B ALA 2 N       1.465
    B LEU 4 C     --  B PJE 5 N       1.415
    B PJE 5 C     --  B 010 6 O       1.448

The two peptide links inside the ALA-VAL-LEU stretch are NOT in _struct_conn
(the file leaves them to standard polymer connectivity, which does not apply once
the whole thing is a ligand chain), so they are declared here and flagged as an
inference rather than read off the file.

COLUMNS
  CA      protein CA-RMSD -- the anchor; a ligand number on a fold that did not
          happen means nothing
  lig     ligand all-atom RMSD in the protein's frame, i.e. is it in the pocket
  SG-C20  the covalent bond length. 1.79 A is chemistry; a plausible ligand RMSD
          with a 6 A "bond" is a ligand sitting near the site, not attached.
          This is the column the screen exists for.

  BONDS=0 drops every declared link, which is the control: if a row looks the
  same either way, this port is not reading bonded_atom_pairs.

  MODELS=openfold3 RECYCLES=10 PYTHONPATH=/home/ubuntu/ColabDesign2 \\
      ~/venv/bin/python tools/oracles/covalent_parity.py

RESULTS (2026-09-02, single sequence, protein self-templated, 10 recycles).
Reference SG-C20 1.79 A.

  model            CA     lig   SG-C20
  intellifold2   0.26    2.03     1.53
  rosettafold3   0.10     n/a      n/a     (atom names are elements -- see below)
  boltz2         0.30   18.99     1.60
  openfold3      0.97    5.90     1.95
  protenix2      2.67   46.56     1.58
  opendde         OOM                      (306 aa needs 1024 structural tokens)

  openfold3, BONDS=0 control:
                 0.96    9.20     9.79

THE CONTROL IS THE POINT. With the link declared the ligand is bonded to CYS145
at 1.95 A; without it, the same model puts it 9.79 A away -- in the neighbourhood
of the site, not attached. So bonded_atom_pairs is both plumbed and load-bearing,
and before this it could not be expressed at all.

EVERY MEASURABLE PORT FORMS THE BOND (1.53-1.95 A against 1.79). What separates
them is the ligand's POSE: intellifold2 places all 49 atoms at 2.03 A, openfold3
at 5.90, while boltz2 (18.99) and protenix2 (46.56) attach the right atom to the
right cysteine and splay the rest. That is exactly why `lig` and `SG-C20` are
separate columns -- a covalent ligand can be correctly bonded and still wrong.

MATCHING THE LIGAND ATOMS, and two ways that do NOT work (both tried, both gave
confident nonsense):
  * flat in-order pairing of our ligand tokens against the file's atom order is
    not a correspondence -- it moved openfold3's bond from 1.95 to 8.16 A on an
    unchanged prediction.
  * keying by (component, index-within-component) is wrong because a ligand
    component is ATOMISED: one token per ATOM, each with its own residue_index,
    so a "component" group holds one atom.
Names are the only identity that survives -- and they do not survive for
rosettafold3, whose featuriser renames atomised atoms to their ELEMENT ('C20' ->
'C'), so its ligand columns are reported unavailable rather than guessed. Its
protein still folds to 0.10 A.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/ColabDesign2')

ALL = ('openfold3', 'intellifold2', 'boltz2', 'opendde', 'protenix2',
       'rosettafold3')
LIG = ('02J', 'ALA', 'VAL', 'LEU', 'PJE', '010')
# (chain, 1-based residue, atom) pairs. Chain 'A' is the protein, 'B' the ligand.
BONDS = [(('A', 145, 'SG'), ('B', 5, 'C20')),      # _struct_conn covale1
         (('B', 1, 'C'), ('B', 2, 'N')),           # covale2
         (('B', 2, 'C'), ('B', 3, 'N')),           # inferred peptide link
         (('B', 3, 'C'), ('B', 4, 'N')),           # inferred peptide link
         (('B', 4, 'C'), ('B', 5, 'N')),           # covale3
         (('B', 5, 'C'), ('B', 6, 'O'))]           # covale4
ANCHOR = (('A', 145, 'SG'), ('B', 5, 'C20'), 1.793)


def kabsch(P, Q):
  pc, qc = P.mean(0), Q.mean(0)
  V, _, Wt = np.linalg.svd((P - pc).T @ (Q - qc))
  d = np.sign(np.linalg.det(V @ Wt))
  return V @ np.diag([1, 1, d]) @ Wt, pc, qc


def apply(U, pc, qc, P):
  return (P - pc) @ U + qc


def reference():
  """6LU7: protein sequence, protein CA, and {(comp_idx, atom): xyz} for N3."""
  import gemmi
  st = gemmi.read_structure(glob.glob('/home/ubuntu/**/6LU7.cif',
                                      recursive=True)[0])
  st.setup_entities()
  st.remove_waters()
  st.remove_alternative_conformations()
  st.remove_hydrogens()
  prot = st[0]['A']
  seq = ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                for r in prot)
  ca = np.array([[r['CA'][0].pos.x, r['CA'][0].pos.y, r['CA'][0].pos.z]
                 for r in prot], np.float32)
  lig = {}
  for i, r in enumerate(st[0]['C']):
    for a in r:
      lig[(i + 1, a.name)] = [a.pos.x, a.pos.y, a.pos.z]
  sg = prot[144]['SG'][0].pos
  return seq, ca, lig, np.array([sg.x, sg.y, sg.z], np.float32)


def main():
  import jax
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  from colabdesign2.af3.runner import AF3Runner
  from tools.module_trace import models

  seq, ref_ca, ref_lig, ref_sg = reference()
  L = len(seq)
  use_bonds = os.environ.get('BONDS', '1') == '1'
  recycles = int(os.environ.get('RECYCLES', '10'))
  contig = '%d %s' % (L, '/'.join('lig:' + c for c in LIG))
  print('6LU7 Mpro %d aa + N3 (%s), bonds %s, template %s, %d recycles'
        % (L, '-'.join(LIG), 'ON' if use_bonds else 'OFF (control)',
           os.environ.get('TEMPLATE', '1'), recycles))
  print('  reference SG-C20 %.2f A'
        % float(np.linalg.norm(ref_sg - np.asarray(ref_lig[(5, 'C20')]))))
  print('%-14s %7s %7s %8s' % ('model', 'CA', 'lig', 'SG-C20'))

  dec = lambda r: ''.join(chr(int(c) + 32) if 0 <= c < 64 else ''
                          for c in r).strip()
  for m in (os.environ.get('MODELS').split(',') if os.environ.get('MODELS')
            else list(ALL)):
    try:
      kw = models.featurise_kwargs(m)
      if 'struct_num_tokens' in kw:
        kw['struct_num_tokens'] = int(os.environ.get('STRUCT_TOKENS', '1024'))
      # TEMPLATE=1 self-templates the PROTEIN chain from the crystal. Mpro is
      # 306 aa and does not fold from a single sequence (CA 16.9 A), and a
      # ligand number on a fold that did not happen means nothing -- so the
      # ligand columns only become readable once the protein is held in place.
      # The template covers the protein only; the ligand still has to be placed.
      tmpl = None
      if os.environ.get('TEMPLATE', '1') == '1':
        from colabdesign.af.prep import prep_pdb
        from colabdesign2.af3.features import spec_to_templates
        p = prep_pdb(os.path.expanduser('~/6LU7.cif'), chain='A',
                     ignore_missing=True)
        tspec = parse_contigs('A').resolve(idx=p['idx'])
        tmpl = spec_to_templates(tspec, p['batch'])
      batch = f3.featurise_spec(parse_contigs(contig), sequences={0: seq},
                                msa_crop_size=1, templates=tmpl,
                                bonds=(BONDS if use_bonds else None), **kw)
      batch = batch[0] if isinstance(batch, tuple) else batch
      cfg, params, _f, _x = models.build(m, batch, num_recycles=recycles,
                                         diffusion_steps=200, num_msa=1)
      r = AF3Runner(cfg=cfg, model_params=params, diffusion='forward', num_msa=1)
      x = np.asarray(r.predict(batch, key=jax.random.PRNGKey(1))
                     ['diffusion_samples']['atom_positions'])
      while x.ndim > 3:
        x = x[0]
      if m == 'opendde':
        from colabdesign2.af3 import structural_features as sf
        x = np.asarray(sf.structural_to_residue_positions(
            x, np.asarray(batch['structbook/residue_atom_gather'])))
      chars = np.asarray(batch['ref_atom_name_chars'])
      mask = np.asarray(batch['ref_mask']).astype(bool)
      asym = np.asarray(batch['asym_id'])
      ri = np.asarray(batch['residue_index'])
      # protein CA, then the ligand's atoms keyed by (component, atom name)
      A = x[:L, 1]
      U = kabsch(A, ref_ca[:L])
      # Match ligand atoms BY NAME. Two wrong alternatives were tried first and
      # both produced confident nonsense, so they are recorded here:
      #   * flat in-order pairing (our ligand tokens vs the file's atom order)
      #     is NOT a correspondence -- it moved openfold3's bond from 1.95 A to
      #     8.16 A on an unchanged prediction.
      #   * keying by (component, index-within-component) is wrong because a
      #     ligand component is ATOMISED: one token per ATOM, each with its own
      #     residue_index, so a "component" group holds a single atom.
      # Names are the only identity that survives, and they do survive for every
      # port EXCEPT rosettafold3, whose featuriser (atomworks) renames atomised
      # atoms to their ELEMENT -- 'C20' decodes as 'C'. Its ligand columns are
      # therefore reported as unavailable rather than guessed at.
      got = {}
      for t in range(x.shape[0]):
        if asym[t] != 2:
          continue
        for a in range(chars.shape[1]):
          if mask[t, a]:
            got[(int(ri[t]), dec(chars[t, a]))] = x[t, a]
      keys = [k for k in got if k in ref_lig]
      anchor_key = (ANCHOR[1][1], ANCHOR[1][2])
      if anchor_key not in got or len(keys) < len(ref_lig) // 2:
        print('%-14s %7.2f %7s %8s   (atom names are elements here -- '
              'ligand columns unavailable)'
              % (m, float(np.sqrt(((apply(*U, A) - ref_ca[:L]) ** 2).sum(-1).mean())),
                 'n/a', 'n/a'), flush=True)
        continue
      P = apply(*U, np.array([got[k] for k in keys]))
      Q = np.array([ref_lig[k] for k in keys])
      lig_rmsd = float(np.sqrt(((P - Q) ** 2).sum(-1).mean()))
      (c1, r1, a1), _b, _d = ANCHOR
      sg = x[r1 - 1, [dec(chars[r1 - 1, i]) for i in range(chars.shape[1])].index(a1)]
      bond = float(np.linalg.norm(np.asarray(sg) - np.asarray(got[anchor_key])))
      print('%-14s %7.2f %7.2f %8.2f   (%d/%d ligand atoms matched)'
            % (m, float(np.sqrt(((apply(*U, A) - ref_ca[:L]) ** 2).sum(-1).mean())),
               lig_rmsd, bond, len(keys), len(ref_lig)), flush=True)
    except Exception as e:                                  # noqa: BLE001
      print('%-14s FAILED  %s: %s' % (m, type(e).__name__, str(e)[:90]),
            flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
