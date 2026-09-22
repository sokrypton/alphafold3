"""Cross-model RNA parity: fold a real tRNA through every port.

Ligands are now covered for all seven ports (ligand_parity.py). RNA is not
covered ANYWHERE: every fold test in the repo is protein, and the only RNA in
the tree is the 1EHZ trace case, which exercises featurisation rather than a
fold. So no port has ever been shown to fold a nucleic acid at all.

1EHZ is yeast tRNA-Phe (76 nt, 1.93 A) -- pure RNA, no protein, and small.
Scored on C1' (the nucleic-acid representative atom, as everywhere else here).

  MODELS=boltz2,protenix2 PYTHONPATH=/home/ubuntu/ColabDesign2 \
      ~/venv/bin/python tools/oracles/rna_parity.py

A de novo RNA fold is genuinely hard, so read this as a coarse screen: single
digits mean the nucleic-acid path works, tens of angstroms or a non-physical
backbone mean it does not. The C1'-C1' spacing is printed for exactly that
reason -- a plausible RMSD on an exploded chain is not a fold.

RESULTS (2026-09-02, single sequence, no MSA, RECYCLES=10, seed 1). ALL SEVEN
PORTS FOLD tRNA.

  model             RMSD   spacing   reading
  rosettafold3     1.143      6.01   best of the family (native rf3: 0.933-1.224)
  opendde          1.372      6.08   native opendde is 1.120 (C4')
  boltz2           1.419      6.06
  openfold3        1.441      5.86
  intellifold2     1.621      6.00
  chai1            1.648      6.11
  protenix2        2.062      6.04   native protenix is 2.058 -- parity

Seven bugs stood between the first run of this screen (which nothing but boltz2
and if2 survived) and the table above:

  1. of3/protenix2/rf3 never permuted msa_activations through the residue
     alphabet -- silent, 32 rows either way. of3 18.680 -> 8.113.
  2. modified bases were fed as the unknown nucleotide N (14 of 76 residues),
     which also cost those tokens their pseudo-beta atom. of3 8.113 -> 1.538.
  3. RECYCLES=0, where the family default is 10.
  4. rf3's alphabet is NOT of3's: G/C and DG/DC are transposed
     (converters/rosettafold3.py _AF3_TO_RF3_AATYPE). rf3 16.755 -> 1.143.
  5. opendde/protenix chain their two atom-attention LayerNorms -- the keys'
     adaptive LayerNorm reads the queries' ALREADY-NORMALISED activation -- where
     AF3 normalises the raw activation once per side. Gated on the diffusion atom
     encoder against native with every input injected: a_token corr 0.990195 ->
     0.999955, one-step denoise 0.0701 -> 0.0075 A. protenix2 2.253 -> 2.062,
     which lands on native protenix's own 2.058.
  6. opendde/protenix zero-PAD the atom key window where AF3 slides it back in
     bounds, so their edge blocks hold different atoms at different key slots.
     Our v_lm against native's, inside native's own mask_trunked: 98.23% ->
     100.00%.
  7. THIS SCREEN's own opendde gather was reading OP1, a phosphate oxygen, as the
     nucleotide representative -- see the opendde branch below. Under (5)+(6) the
     fold is 7.529 A by the wrong atom and 1.160 A by the right one, so the old
     5.584 was two errors partly cancelling.

The spacing column is the guard that made all of this readable: a plausible RMSD
on an exploded chain is not a fold, and a gather pointing at the wrong atom
shows up here first (opendde's three wrong gathers read 0.44, 7.57 and 5.04 A).

CAVEAT on reading these as generalisation: 1EHZ is a 1999 structure, inside every
one of these models' training windows. The table is a PARITY screen against each
native implementation, not evidence that any of them predicts new RNA.
"""
import os
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/ColabDesign2')

ALL = ('boltz2', 'openfold3', 'opendde', 'intellifold2', 'protenix2',
       'rosettafold3', 'chai1')


def rna_reference():
  """1EHZ chain A as (sequence, C1' coords), modified bases mapped to their PARENT.

  gemmi's one_letter_code gives X for anything non-standard, and 14 of tRNA-Phe's
  76 residues are modified (1MA, PSU, 7MG, YYG, ...). Feeding those as the unknown
  nucleotide N -- what the first version of this harness did -- throws away 18% of
  the sequence AND leaves those tokens without a pseudo-beta atom (AF3 picks C4/C2
  by base identity and falls back to the first atom, P, with a warning per token).
  Every CCD entry here carries _chem_comp.mon_nstd_parent_comp_id, so the parent is
  known exactly: 1MA->A, PSU->U, 7MG->G and so on. That is a fairer target for every
  model, and it is the sequence a user would actually fold.
  """
  import glob
  import gemmi
  from colabdesign2.af3.features import get_ccd
  path = glob.glob('/home/ubuntu/**/1EHZ.cif', recursive=True)[0]
  st = gemmi.read_structure(path)
  st.setup_entities()
  st.remove_ligands_and_waters()
  ccd = get_ccd()
  seq, coords = [], []
  for r in st[0]['A']:
    name = r.name
    if name not in ('A', 'G', 'C', 'U'):
      d = ccd.get(name)
      parent = d['_chem_comp.mon_nstd_parent_comp_id'][0] if d else '?'
      name = parent if parent in ('A', 'G', 'C', 'U') else 'N'
    seq.append(name if len(name) == 1 else 'N')
    a = r.find_atom("C1'", '*')
    coords.append([a.pos.x, a.pos.y, a.pos.z] if a else [np.nan] * 3)
  return ''.join(seq), np.asarray(coords, np.float32)


def rna_reference():
  """1EHZ chain A as (sequence, C1' coords), modified bases resolved to their parent.

  The first version of this harness took gemmi's one_letter_code, which returns X
  for anything non-standard, and mapped X -> N. That threw away 14 of tRNA-Phe's 76
  residues (1MA, PSU, 7MG, 5MC, YYG, ...) as "unknown nucleotide" -- 18% of the
  sequence -- and left those tokens with no pseudo-beta atom, since AF3 picks C4/C2
  by base identity and otherwise falls back to the first atom (P) with a warning per
  token. The target was harder than it needed to be, for everyone.

  The residue-typing approach is borrowed from py2Dmol's structure parser
  (py2Dmol/viewer.py, _extract_coords): classify with
  gemmi.find_tabulated_residue(...).is_nucleic_acid() rather than trusting
  one_letter_code. gemmi's table already knows the modified bases and returns their
  parent as a LOWERCASE letter -- PSU->'u', 1MA->'a', 7MG->'g' -- so the parent comes
  straight out of it. It does not know YYG (wybutosine), which falls through to the
  CCD's own _chem_comp.mon_nstd_parent_comp_id (YYG->G). Between them all 14 resolve.
  """
  import glob
  import gemmi
  from colabdesign2.af3.features import get_ccd
  path = glob.glob('/home/ubuntu/**/1EHZ.cif', recursive=True)[0]
  st = gemmi.read_structure(path)
  st.setup_entities()
  st.remove_ligands_and_waters()
  ccd, seq, coords, unresolved = get_ccd(), [], [], []
  for r in st[0]['A']:
    info = gemmi.find_tabulated_residue(r.name)
    letter = info.one_letter_code.upper() if info and info.is_nucleic_acid() else ' '
    if letter not in ('A', 'G', 'C', 'U'):
      d = ccd.get(r.name)
      parent = d['_chem_comp.mon_nstd_parent_comp_id'][0] if d else '?'
      letter = parent if parent in ('A', 'G', 'C', 'U') else 'N'
      if letter == 'N':
        unresolved.append(r.name)
    seq.append(letter)
    a = r.find_atom("C1'", '*')          # repo convention (cases._cif_chain)
    coords.append([a.pos.x, a.pos.y, a.pos.z] if a else [np.nan] * 3)
  if unresolved:
    print('  (%d residues left as N: %s)' % (len(unresolved), sorted(set(unresolved))))
  return ''.join(seq), np.asarray(coords, np.float32)


def rna_reference_c4():
  """1EHZ chain A C4' coordinates -- opendde's nucleotide representative atom."""
  import glob
  import gemmi
  st = gemmi.read_structure(glob.glob('/home/ubuntu/**/1EHZ.cif', recursive=True)[0])
  st.setup_entities()
  st.remove_ligands_and_waters()
  out = []
  for r in st[0]['A']:
    a = r.find_atom("C4'", '*')
    out.append([a.pos.x, a.pos.y, a.pos.z] if a else [np.nan] * 3)
  return np.asarray(out, np.float32)


def kabsch(P, Q):
  P, Q = P - P.mean(0), Q - Q.mean(0)
  V, _, Wt = np.linalg.svd(P.T @ Q)
  U = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ Wt))]) @ Wt
  return float(np.sqrt(((P @ U - Q) ** 2).sum(-1).mean()))


def main():
  import jax
  from colabdesign2.af3.runner import AF3Runner
  from tools.module_trace import cases, models

  want = os.environ.get('MODELS')
  seeds = [int(x) for x in os.environ.get('SEEDS', '1').split(',')]
  print('%-14s %10s %12s' % ('model', 'RMSD', 'spacing'))
  print("  (C1' for every model -- opendde's structural-token output is gathered\n   back to the residue layout by atom NAME, see the opendde branch below)")
  for m in (want.split(',') if want else list(ALL)):
    try:
      kw = models.featurise_kwargs(m)
      if os.environ.get('NO_CHIRAL') == '1' and kw.get('chirals'):
        # rf3 is the ONLY port that consumes chiral features, and its chiral
        # gate covered a protein (6MRR, 213/213 rows) and a ligand (biotin, 9/9)
        # -- never a nucleotide. Ribose has four stereocentres per residue, so a
        # wrong SIGN there is an actively wrong restraint on every residue.
        kw['chirals'] = False
      if 'struct_num_tokens' in kw:      # opendde: sized for 6MRR, see ligand_parity
        kw['struct_num_tokens'] = int(os.environ.get('STRUCT_TOKENS', '512'))
      # Build the batch directly rather than through cases.build: the 1ehz case
      # resolves a chain-REFERENCE contig ('A'), which needs an idx= this
      # harness has no reason to carry. A free 'rna:<L>' segment with the
      # sequence supplied is the same input to the model.
      from colabdesign2 import parse_contigs
      from colabdesign2.af3 import features as f3
      seq, ref = rna_reference()
      L = len(seq)
      spec = parse_contigs('rna:%d' % L).resolve()
      batch = f3.featurise_spec(spec, sequences={0: seq}, msa_crop_size=1, **kw)
      batch = batch[0] if isinstance(batch, tuple) else batch
      cfg, params, _f, missing = models.build(
          m, batch, num_recycles=int(os.environ.get('RECYCLES', '0')),
          diffusion_steps=200, num_msa=1)
      if missing:
        print('  (%d params at INIT)' % len(missing))
      runner = AF3Runner(cfg=cfg, model_params=params, diffusion='forward',
                         num_msa=1)
      for s in seeds:
        out = runner.predict(batch, key=jax.random.PRNGKey(s))
        x = np.asarray(out['diffusion_samples']['atom_positions'])
        while x.ndim > 3:
          x = x[0]
        if m == 'opendde':
          # opendde diffuses on STRUCTURAL TOKENS, so its output is not in the
          # residue-atom layout and the representative atom has to be gathered
          # back. Do it by DECODED ATOM NAME through structbook/
          # residue_atom_gather, which rebuilds the residue-level layout from the
          # structural one, and take C1' like every other model.
          #
          # This replaces a role-and-slot version (role 5, dense slot 2) that was
          # wrong and cost this row 4 A. Slot 2 had been picked by scanning for
          # the most physical consecutive spacing, and the procedure was
          # cross-checked on 6MRR protein -- where it does pick slot 1 = CA and
          # does recover 1.481 A. It still failed here, because a plausible
          # spacing is not identification: the role-5 (rna_bb) subtoken's dense
          # slots are
          #     OP3 P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' O2' C1'
          # so slot 2 is OP1, a phosphate oxygen, and C4' is slot 6. Scored
          # against C4' reference coordinates OP1 reads 5.042 A where the same
          # prediction's real C4' reads 1.160 A. A spacing heuristic cannot tell
          # a phosphate oxygen from a sugar carbon; an atom name can.
          gather = np.asarray(batch['structbook/residue_atom_gather'])
          chars = np.asarray(batch['ref_atom_name_chars'])
          rmask = np.asarray(batch['ref_mask']).astype(bool)
          dec = lambda r: ''.join(chr(int(c) + 32) if 0 <= c < 64 else ''
                                  for c in r).strip()
          flat = x.reshape(-1, 3)
          rep = np.full((L, 3), np.nan, np.float32)
          for t in range(L):
            for j in range(gather.shape[1]):
              if gather[t, j] >= 0 and rmask[t, j] and dec(chars[t, j]) == "C1'":
                rep[t] = flat[gather[t, j]]
        else:
          # C1' is dense-atom slot 0 for a nucleotide, the way CA is slot 1 for
          # an amino acid. Taken from the batch rather than assumed: the atom
          # name decides, so a layout change cannot silently move it.
          chars = np.asarray(batch['ref_atom_name_chars'])
          dec = lambda r: ''.join(chr(int(c) + 32) if 0 <= c < 64 else ''
                                  for c in r).strip()
          rep = np.full((L, 3), np.nan, np.float32)
          for t in range(min(L, chars.shape[0])):
            for a in range(chars.shape[1]):
              if dec(chars[t, a]) == "C1'":
                rep[t] = x[t, a]
                break
        keep = ~np.isnan(rep).any(-1) & ~np.isnan(ref[:L]).any(-1)
        d = np.sqrt(((rep[keep][1:] - rep[keep][:-1]) ** 2).sum(-1))
        print('%-14s %10.3f %12.2f' % (m, kabsch(rep[keep], ref[:L][keep]),
                                       d.mean()))
    except Exception as e:
      print('%-14s %10s %12s   %s: %s'
            % (m, 'ERR', '', type(e).__name__, str(e)[:80]))
  return 0


if __name__ == '__main__':
  sys.exit(main())
