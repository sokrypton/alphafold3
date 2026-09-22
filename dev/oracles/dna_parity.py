"""Do the ports fold DNA? The one nucleic case nothing had ever run.

RNA is screened (rna_parity.py) and DNA was not screened at all -- featurisation
tests only. That is the wrong way round for risk, because DNA is where the
residue-alphabet bug class actually lives: rosettafold3 orders its nucleotides
`21 A, 22 C, 23 G, 24 U, ... 26 DA, 27 DC, 28 DG, 29 DT` where OF3/protenix/
opendde order `22 G, 23 C, ... 27 DG, 28 DC`, so G/C AND DG/DC are transposed.
The RNA half of that was found and fixed (16.755 -> 1.143 A on 1EHZ); the DNA
half was fixed in the same array and has never been exercised.

CASE. The lambda-repressor operator duplex from 1LMB, chains 1 and 2:

    AATACCACTGGCGGTGATAT
    TATATCACCGCCAGTGGTAT

20 bp, protein stripped. Chosen because it is ASYMMETRIC. Almost every
crystallised short duplex is self-complementary -- the Drew-Dickerson dodecamer
CGCGAATTCGCG is the standard choice -- and a palindrome is exactly the wrong
target here: transposing G and C maps it to another valid palindrome, which
still folds to a fine B-DNA duplex, so the bug this screen exists to catch would
be invisible. Under the same swap this operator mispairs.

METRICS. Both strands' C1' together after superposition is the duplex geometry;
strand 2's RMSD in strand 1's FRAME is the register -- a mispaired duplex can
have two individually reasonable strands and the wrong relationship, exactly as
in multimer_parity.py. `spacing` is the same exploded-chain guard rna_parity.py
uses: a plausible RMSD on a chain that is not a chain is not a fold.

SWAP=1 is the SENSITIVITY CONTROL, and it is the point of the design. It feeds
the G<->C and DG<->DC transposed sequence. If a row barely moves under SWAP, this
screen cannot see the bug class and its clean numbers mean nothing; the swap must
degrade the fold for the target to be a valid probe.

  MODELS=rosettafold3 SWAP=1 RECYCLES=10 PYTHONPATH=/home/ubuntu/ColabDesign2 \\
      ~/venv/bin/python tools/oracles/dna_parity.py

RESULTS (2026-09-02, single sequence, 10 recycles, seed 1). ALL SEVEN PORTS FOLD
DNA. The reference's own intra-strand C1' spacing is 5.00 A, so every row is a
real duplex rather than an extended chain:

  model          C1 RMSD   register   spacing
  boltz2            1.59       1.64      4.91
  chai1             1.89       1.89      4.92
  intellifold2      1.95       2.18      5.01
  opendde           1.99       2.19      5.09
  protenix2         2.21       2.59      4.95
  openfold3         2.22       2.42      5.05
  rosettafold3      2.67       3.05      4.89

WHAT THE SWAP CONTROL SHOWED, AND WHY IT CHANGED THE PLAN. Under SWAP=1,
openfold3 goes 2.22 -> 2.85 and rosettafold3 2.67 -> 2.79. That is a real but
FEEBLE signal, and the reason is structural, not a flaw in the target: swapping
G<->C in both strands maps a G:C pair to C:G, so the swapped duplex is still
perfectly Watson-Crick complementary and still folds to good B-DNA. Choosing an
ASYMMETRIC operator instead of the usual palindrome does not rescue this, because
complementarity is preserved under the G<->C involution either way.

So a DNA duplex CANNOT gate the residue-alphabet bug class -- the thing that most
needed checking here. The RNA half of that bug was caught by a fold (rf3 16.755 ->
1.143 A on 1EHZ) only because tRNA's tertiary structure is sequence-specific in a
way a duplex is not. The alphabets are therefore gated STATICALLY instead, against
each native repo's own residue table, in tests/test_nucleotide_alphabets.py: all
six non-baseline converters are correct, protenix/opendde/boltz2 on A G C U /
DA DG DC DT and rosettafold3/chai1 on the alphabetical A C G U / DA DC DG DT.

This screen's remaining job is the honest one: DNA works end-to-end everywhere.
A structured DNA target whose fold depends on base identity -- a G-quadruplex, a
Holliday junction, an aptamer -- would be the sensitive fold probe, if these
models predict such things well enough for the number to mean anything.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/ColabDesign2')

ALL = ('boltz2', 'openfold3', 'opendde', 'intellifold2', 'protenix2',
       'rosettafold3', 'chai1')
_SWAP = str.maketrans('GCgc', 'CGcg')


def kabsch(P, Q):
  pc, qc = P.mean(0), Q.mean(0)
  V, _, Wt = np.linalg.svd((P - pc).T @ (Q - qc))
  d = np.sign(np.linalg.det(V @ Wt))
  return V @ np.diag([1, 1, d]) @ Wt, pc, qc


def rmsd_after(U, pc, qc, P, Q):
  return float(np.sqrt((((P - pc) @ U - (Q - qc)) ** 2).sum(-1).mean()))


def rmsd(P, Q):
  return rmsd_after(*kabsch(P, Q), P, Q)


def duplex():
  """1LMB chains 1 and 2 as [(sequence, C1' coords), ...], protein stripped."""
  import gemmi
  path = glob.glob('/home/ubuntu/**/1LMB.cif', recursive=True)[0]
  st = gemmi.read_structure(path)
  st.setup_entities()
  st.remove_ligands_and_waters()
  st.remove_alternative_conformations()
  out = []
  for ch in st[0]:
    rs = [r for r in ch if r.name in ('DA', 'DG', 'DC', 'DT')]
    if not rs:
      continue
    seq = ''.join(r.name[-1] for r in rs)
    c1 = np.array([[r["C1'"][0].pos.x, r["C1'"][0].pos.y, r["C1'"][0].pos.z]
                   for r in rs], np.float32)
    out.append((seq, c1))
  return out


def main():
  import jax
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  from colabdesign2.af3.runner import AF3Runner
  from tools.module_trace import models

  chains = duplex()
  (sa, ca), (sb, cb) = chains[0], chains[1]
  la, lb = len(sa), len(sb)
  swap = os.environ.get('SWAP') == '1'
  qa, qb = (sa.translate(_SWAP), sb.translate(_SWAP)) if swap else (sa, sb)
  recycles = int(os.environ.get('RECYCLES', '10'))
  print('1LMB operator duplex %d + %d bp%s, %d recycles'
        % (la, lb, '   [G<->C SWAPPED: sensitivity control]' if swap else '',
           recycles))
  print('  %s / %s' % (qa, qb))
  # the reference's OWN intra-strand C1'-C1' spacing, so the guard column has an
  # anchor instead of a remembered rule of thumb
  print('  reference C1 spacing %.2f A'
        % float(np.median(np.concatenate([
            np.linalg.norm(np.diff(ca, axis=0), axis=-1),
            np.linalg.norm(np.diff(cb, axis=0), axis=-1)]))))
  print('%-14s %8s %9s %9s' % ('model', 'C1 RMSD', 'register', 'spacing'))

  for m in (os.environ.get('MODELS').split(',') if os.environ.get('MODELS')
            else list(ALL)):
    try:
      kw = models.featurise_kwargs(m)
      if 'struct_num_tokens' in kw:
        kw['struct_num_tokens'] = int(os.environ.get('STRUCT_TOKENS', '256'))
      # Two SEPARATE dna chains. In a contig string ':' (or whitespace) is the
      # CHAIN separator and '/' the segment separator WITHIN a chain, so
      # 'dna:20/dna:20' is one 40-mer chain and 'dna:20 dna:20' is two 20-mers.
      # The '/0 ' spelling that works for protein ('146/0 74') also fails here:
      # the zero-length segment defaults to protein and AF3 chains are
      # single-type.
      spec = parse_contigs('dna:%d dna:%d' % (la, lb)).resolve()
      batch = f3.featurise_spec(spec, sequences={0: qa, 1: qb},
                                msa_crop_size=1, **kw)
      batch = batch[0] if isinstance(batch, tuple) else batch
      cfg, params, _f, missing = models.build(m, batch, num_recycles=recycles,
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
      # C1' by NAME, never by slot -- rna_parity.py's 4 A scoring bug was a slot
      # assumed to be the representative atom that was a phosphate oxygen.
      chars = np.asarray(batch['ref_atom_name_chars'])
      rmask = np.asarray(batch['ref_mask']).astype(bool)
      dec = lambda r: ''.join(chr(int(c) + 32) if 0 <= c < 64 else ''
                              for c in r).strip()
      rep = np.full((la + lb, 3), np.nan, np.float32)
      for t in range(la + lb):
        for a in range(chars.shape[1]):
          if rmask[t, a] and dec(chars[t, a]) == "C1'":
            rep[t] = x[t, a]
      A, B = rep[:la], rep[la:la + lb]
      ok = ~np.isnan(rep).any(-1)
      allref = np.concatenate([ca, cb])
      ua = kabsch(A, ca)
      print('%-14s %8.2f %9.2f %9.2f%s'
            % (m, rmsd(rep[ok], allref[ok]),
               rmsd_after(ua[0], ua[1], ua[2], B, cb),
               float(np.median(np.linalg.norm(np.diff(A, axis=0), axis=-1))),
               '' if not missing else '  (%d at INIT)' % len(missing)),
            flush=True)
    except Exception as e:                                  # noqa: BLE001
      print('%-14s FAILED  %s: %s' % (m, type(e).__name__, str(e)[:80]),
            flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
