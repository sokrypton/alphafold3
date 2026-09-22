"""Can the ports model a POST-TRANSLATIONAL MODIFICATION? Phospho-Ser, all seven.

Until now `featurise_spec` hardcoded `ptms=[]` (and `modifications=[]` for DNA and
RNA), so a modified residue could not be expressed at all. The nucleic version of
that gap already cost real accuracy once: tRNA's modified bases were fed as the
unknown nucleotide, which also cost those tokens their representative atom
(openfold3 8.113 -> 1.538 A once resolved to the parent). This screen exists for
the protein half, now that `modifications=` threads through.

CASE: 5K9P -- ubiquitin phosphorylated at Ser20 (76 aa, single chain, no
heteroatoms, sub-2 A). Chosen because the FOLD is not in question: every port
folds ubiquitin, so the only thing that can move is the modification. Picking a
target the model already solves is usually the mistake -- it hides an inert
feature path -- but here it is the point, because the phosphate is measured
directly rather than through its effect on the backbone.

WHAT AF3 DOES WITH A PTM, which the screen has to account for: it ATOMISES the
modified residue -- one token per atom, exactly like a ligand -- and gives each of
those tokens the PARENT restype. So phospho-Ser20 turns a 76-residue chain into
85 tokens, token index no longer tracks residue index, and the residue's atoms
are spread one per token at dense slot 0. Verified: SEP contributes
N CA CB OG C O P O1P O2P O3P, and PTR 16 atoms.

COLUMNS
  CA        CA-RMSD over all 76 residues -- the control that says the fold happened
  PO4       RMSD of the phosphate (P, O1P, O2P, O3P) to the crystal, in the frame
            fitted on CA. This is the number the screen is for.
  OG-P      the phosphoester bond length. ~1.6 A is chemistry; a plausible PO4
            RMSD with a 3 A bond means the group is floating, not attached.
  noPTM     CA-RMSD of the same sequence folded with NO modification, as plain
            Ser. It has no phosphate to score at all, which is the baseline the
            PTM path has to beat by existing.

  MODELS=openfold3 RECYCLES=10 PYTHONPATH=/home/ubuntu/ColabDesign2 \\
      ~/venv/bin/python tools/oracles/ptm_parity.py

RESULTS (2026-09-02, single sequence, 10 recycles, seed 1). ALL SEVEN MODEL A PTM.
Reference OG-P bond 1.61 A.

  model            CA     PO4    OG-P     noPTM
  intellifold2   2.02    0.73    1.61      1.62
  openfold3      1.41    2.09    1.59      1.50
  rosettafold3   2.16    2.32    1.58      1.68
  opendde        2.16    2.36    1.59      1.92
  chai1          2.05    2.84    1.91      1.95
  boltz2         2.06    3.10    1.61      1.86
  protenix2      7.55    4.64    1.61      8.58

BOLTZ2 NEEDS ITS OWN CONVENTION, ALL OF IT AT ONCE (fixed; was PO4 5.95 with a
3.56 A OG-P bond, i.e. a phosphate floating off an inflated residue -- every
intra-residue bond ~2.4x too long while the peptide bonds in and out were fine).

Native boltz2 on the same input gets PO4 1.77 with correct bonds, so the
capability existed and this was a port bug. Its tokenizer
(data/tokenize/boltz2.py) has three branches, and a PTM takes the third:

    if res["is_standard"]:                       -> one token per residue
    elif chain["mol_type"] == NONPOLYMER:        -> one token per ATOM (ligands)
    else:                                        -> ONE token, ALL the atoms,
                                                    res_type = UNK, modified=True

Three differences from AF3's convention, and they only work TOGETHER. Measured,
each on top of the previous:

  layout      restype   modified flag    PO4    OG-P
  atomised    parent    0               5.95    3.56    (AF3's convention)
  one token   parent    0               6.40    4.27
  one token   UNK       0               6.42    4.70
  one token   parent    1               6.53    4.37
  one token   UNK       1               3.10    1.61    <- all three

So neither the tokenisation nor the restype nor the flag does anything on its
own -- an earlier commit called the tokenisation clash the root cause on the
strength of reading native source, and the experiment disproved it. (UNK, 1) is
the joint signature boltz2 was trained to read as "this token is a modified
residue, take its shape from the reference conformer"; either half alone is a
combination it never saw. The bond is now chemically correct (1.61 A against the
crystal's 1.61 and native's 1.64); the residual 3.10 vs native's 1.77 on the
phosphate is unexplained, and CA moves 1.65 -> 2.06 because the trunk loses "this
is a serine" (native is 1.76).

Delivered as featurise_spec(modified_as_one_token=True) plus the `is_modified`
token feature, both declared for boltz2 in module_trace.models.FEATURISE. Every
other family keeps AF3's convention and its numbers are unchanged to the digit.

BOND INFORMATION, both halves checked and neither closes the residual gap:
  * SYMMETRY is already right -- boltz2 is in model_config.OPENFOLD3_LINEAGE, so
    token_bond_matrix/token_bond_type_matrix are called with symmetrize=True,
    matching featurizerv2 writing both bonds[t1,t2] and bonds[t2,t1].
  * The token-bond DIAGONAL is a real difference from native and does nothing.
    Native keeps a non-standard residue's whole CCD bond list in bond_data
    (parse_ccd_residue sets bonds=bonds; a STANDARD residue gets bonds=[]), and
    when the residue is one token both atoms of every intra-residue bond map to
    that same token, so the list collapses onto bonds[t,t] -- an entry a standard
    residue never has. We set nothing there. Adding it, with either bond order,
    moves the phosphate 3.10 -> 3.16 A. Not the gap.
  * boltz2 has no atom-level intra-residue bond feature to be missing: the only
    other place structure.bonds is read is cyclic-polymer detection, which
    explicitly skips same-residue bonds.

ALSO RULED OUT, so they are not re-run: the reference space (all ten atoms share
one ref_space_uid in BOTH layouts) and the reference conformer geometry (N-CA
1.46, CB-OG 1.37, OG-P 1.67, P-O1P 1.52 -- correct in both), and the bonds (9,
with real CCD orders 1 and 2). Two intermediate readings of those were wrong
because the SEP's tokens were located by a guessed index; find them by
residue_index, and remember an atomised token's single atom sits at its
DENSE-LAYOUT slot, not slot 0.

protenix2's 7.55 A is NOT a PTM failure -- its bond is a clean 1.61 and its
unmodified control is 8.58, i.e. it simply does not fold ubiquitin from a single
sequence. Read the PO4 column only where the CA column says a fold happened.

TWO SCORING TRAPS this screen hit, both of which first looked like port bugs:
  * rosettafold3 read "only 1/4 phosphate atoms". atomworks names atomised atoms
    by ELEMENT, so O1P/O2P/O3P all decode to "O" and a name lookup finds one.
    Atoms are therefore identified POSITIONALLY, by index into the residue's
    CCD-ordered token run.
  * opendde crashed with an IndexError. attach_structural_batch re-featurises to
    build the structural layout and was not being passed `modifications`, so it
    tokenised the chain as its unmodified self -- 76 rows of
    residue_atom_gather against 85 residue-level tokens. Now threaded through.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/ColabDesign2')

ALL = ('openfold3', 'intellifold2', 'boltz2', 'opendde', 'protenix2',
       'rosettafold3', 'chai1')
MOD = 'SEP'
PHOSPHATE = ('P', 'O1P', 'O2P', 'O3P')
# SEP's atoms in CCD order, which is the token order AF3 atomises into.
# Taken from the crystal itself (5K9P's SEP20 lists exactly this), so it
# is not a guess about the CCD.
ORDER = ('N', 'CA', 'CB', 'OG', 'C', 'O', 'P', 'O1P', 'O2P', 'O3P')


def kabsch(P, Q):
  pc, qc = P.mean(0), Q.mean(0)
  V, _, Wt = np.linalg.svd((P - pc).T @ (Q - qc))
  d = np.sign(np.linalg.det(V @ Wt))
  return V @ np.diag([1, 1, d]) @ Wt, pc, qc


def apply(U, pc, qc, P):
  return (P - pc) @ U + qc


def reference():
  """5K9P chain A: sequence (SEP -> S), 1-based SEP position, {(res, atom): xyz}."""
  import gemmi
  st = gemmi.read_structure(glob.glob('/home/ubuntu/**/5K9P.cif',
                                      recursive=True)[0])
  st.setup_entities()
  st.remove_waters()
  st.remove_alternative_conformations()
  st.remove_hydrogens()
  seq, pos, coords = '', None, {}
  for i, r in enumerate(st[0]['A']):
    if r.name == MOD:
      seq += 'S'
      pos = i + 1
    else:
      seq += gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
    for a in r:
      coords[(i + 1, a.name)] = [a.pos.x, a.pos.y, a.pos.z]
  return seq, pos, coords


def predicted_atoms(batch, x):
  """(atoms, modified) -- {(1-based residue, name): xyz}, and per-residue lists.

  Two things make a name lookup insufficient on its own.

  Tokens cannot be indexed by residue: an atomised residue occupies several
  tokens. They are grouped the way AF3 numbers them -- a new
  (asym_id, residue_index) pair starts a new residue -- which is the same
  mapping _preserve_reference_gaps uses to spread per-residue numbering back
  over tokens.

  And rosettafold3 legitimately renames atomised atoms to their ELEMENT symbol
  (atomworks: "Using element type for atom names of atomized tokens", verified
  against a native rf3 batch), so SEP's O1P/O2P/O3P all decode to "O" and a
  name lookup finds one of four phosphate atoms. Hence `modified`: the residue's
  atoms in TOKEN order, which is CCD order for every port because they all go
  through AF3's featuriser, and rf3 renames without reordering. Position in that
  list identifies an atom without trusting its name.
  """
  asym = np.asarray(batch['asym_id']).reshape(-1)
  ri = np.asarray(batch['residue_index']).reshape(-1)
  keep = np.asarray(batch['seq_mask']).reshape(-1).astype(bool)
  chars = np.asarray(batch['ref_atom_name_chars'])
  mask = np.asarray(batch['ref_mask']).astype(bool)
  dec = lambda r: ''.join(chr(int(c) + 32) if 0 <= c < 64 else ''
                          for c in r).strip()
  out, per_res = {}, {}
  ordinal = 0
  for t in range(len(asym)):
    if not keep[t]:
      continue
    if t and (asym[t] != asym[t - 1] or ri[t] != ri[t - 1]):
      ordinal += 1
    for a in range(chars.shape[1]):
      if mask[t, a]:
        out[(ordinal + 1, dec(chars[t, a]))] = x[t, a]
        per_res.setdefault(ordinal + 1, []).append(x[t, a])
  return out, per_res


def main():
  import jax
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  from colabdesign2.af3.runner import AF3Runner
  from tools.module_trace import models

  seq, pos, ref = reference()
  L = len(seq)
  recycles = int(os.environ.get('RECYCLES', '10'))
  print('5K9P phospho-ubiquitin, %d aa, %s at %d, %d recycles'
        % (L, MOD, pos, recycles))
  print('  reference OG-P bond %.2f A'
        % float(np.linalg.norm(np.asarray(ref[(pos, 'OG')])
                               - np.asarray(ref[(pos, 'P')]))))
  print('%-14s %7s %7s %7s | %7s' % ('model', 'CA', 'PO4', 'OG-P', 'noPTM'))

  def fold(m, mods):
    kw = models.featurise_kwargs(m)
    if 'struct_num_tokens' in kw:
      kw['struct_num_tokens'] = int(os.environ.get('STRUCT_TOKENS', '256'))
    batch = f3.featurise_spec(parse_contigs('%d' % L).resolve(),
                              sequences={0: seq}, msa_crop_size=1,
                              modifications=mods, **kw)
    batch = batch[0] if isinstance(batch, tuple) else batch
    cfg, params, _f, _miss = models.build(m, batch, num_recycles=recycles,
                                          diffusion_steps=200, num_msa=1)
    runner = AF3Runner(cfg=cfg, model_params=params, diffusion='forward',
                       num_msa=1)
    out = runner.predict(batch, key=jax.random.PRNGKey(
        int(os.environ.get('SEED', '1'))))
    x = np.asarray(out['diffusion_samples']['atom_positions'])
    while x.ndim > 3:
      x = x[0]
    if m == 'opendde':
      from colabdesign2.af3 import structural_features as sf
      x = np.asarray(sf.structural_to_residue_positions(
          x, np.asarray(batch['structbook/residue_atom_gather'])))
    return predicted_atoms(batch, x)

  def ca_rmsd(got):
    keys = [(r, 'CA') for r in range(1, L + 1)
            if (r, 'CA') in got and (r, 'CA') in ref]
    A = np.array([got[k] for k in keys])
    B = np.array([ref[k] for k in keys])
    return A, B, float(np.sqrt(((apply(*kabsch(A, B), A) - B) ** 2).sum(-1).mean()))

  for m in (os.environ.get('MODELS').split(',') if os.environ.get('MODELS')
            else list(ALL)):
    try:
      got, mods = fold(m, {0: [(MOD, pos)]})
      A, B, ca = ca_rmsd(got)
      U = kabsch(A, B)
      # identify the modified residue's atoms POSITIONALLY (see predicted_atoms)
      atoms = mods.get(pos, [])
      note = ''
      if len(atoms) == len(ORDER):
        idx = [ORDER.index(a) for a in PHOSPHATE]
        P = apply(*U, np.array([atoms[i] for i in idx]))
        Q = np.array([ref[(pos, a)] for a in PHOSPHATE])
        po4 = float(np.sqrt(((P - Q) ** 2).sum(-1).mean()))
        bond = float(np.linalg.norm(
            np.asarray(atoms[ORDER.index('OG')])
            - np.asarray(atoms[ORDER.index('P')])))
      else:
        po4, bond = float('nan'), float('nan')
        note = '  (%d atoms on the modified residue, expected %d)' % (
            len(atoms), len(ORDER))
      _, _, plain = ca_rmsd(fold(m, None)[0])
      print('%-14s %7.2f %7.2f %7.2f | %7.2f%s'
            % (m, ca, po4, bond, plain, note), flush=True)
    except Exception as e:                                  # noqa: BLE001
      print('%-14s FAILED  %s: %s' % (m, type(e).__name__, str(e)[:80]),
            flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
