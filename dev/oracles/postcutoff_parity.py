"""The question every other screen in this directory has to defer: do the ports
predict a structure NO model has seen?

rna_parity, ligand_parity, multimer_parity, dna_parity, ptm_parity and
covalent_parity are all parity screens -- they compare a port against its own
native implementation, on targets inside every model's training window (1EHZ is
from 1999). They establish that the port is faithful. They cannot establish that
the model is right, and this file does not confuse the two either: it just
measures what these ports do on structures that were not available to train on.

TARGETS, both DEPOSITED 2025-08 and RELEASED 2026-09-02, i.e. after every
checkpoint here was trained:

  9XZ9  QseC, quorum sensing and excision controller protein, 70 aa
  9WGR  E. coli SufE with residues 112-115 (TQHL) deleted, 134 aa

🔴 POST-CUTOFF IS NOT THE SAME AS NOVEL. The coordinates were unavailable, but
homologues may not have been -- 9WGR is explicitly a deletion mutant of SufE,
whose wild type has been in the PDB for years, so its fold is certainly known to
these models. Read that row as "can it place a small deletion", not as a
generalisation test. 9XZ9 is the more honest of the two and still says nothing
about whether a homologue was in training.

Single sequence, no MSA and no template, which is the hardest setting and the one
design actually uses -- and which is exactly why the 1STP control is in the table.
A post-cutoff protein failing from a single sequence proves nothing on its own;
it only means something if a PRE-cutoff protein of the same kind succeeds. pLDDT is reported alongside because a model that is wrong
should ideally know it -- the confidence heads all track error now (see
confidence_parity.py), so a high pLDDT on a bad fold is a different and worse
failure than a low one.

  MODELS=openfold3 RECYCLES=10 PYTHONPATH=/home/ubuntu/ColabDesign2 \\
      ~/venv/bin/python tools/oracles/postcutoff_parity.py

RESULTS (2026-09-03, single sequence, 10 recycles, seed 1). RMSD / mean CA-pLDDT:

  model            9XZ9 (post)   9WGR (post)   1STP (PRE, control)
  intellifold2      5.15/60.3     8.01/73.9      5.21/64.5
  rosettafold3     10.67/81.4    12.34/74.5      3.79/74.6
  boltz2           10.85/59.0    15.95/48.6      1.99/87.7
  opendde          15.25/86.4    13.54/68.6      5.42/69.7
  openfold3        15.62/56.9    13.01/43.7      8.51/43.2
  protenix2        18.39/90.3    12.94/77.4      6.47/61.6

WHAT THIS DOES AND DOES NOT SHOW.

The control decides how to read each row, and it reads differently per model.
openfold3 fails the PRE-cutoff control too (8.51 A, pLDDT 43), so its
post-cutoff numbers say nothing about the training boundary -- they say it does
not fold a natural protein from a single sequence. Same for intellifold2, whose
post-cutoff 9XZ9 (5.15) and pre-cutoff 1STP (5.21) are the same number.

boltz2 is the one row where the contrast is real: it SOLVES the pre-cutoff
control from a single sequence (1.99 A, confidently, pLDDT 87.7) and fails both
post-cutoff targets (10.85, 15.95). For that model the no-MSA setting cannot be
the explanation, and the training boundary is the visible difference.

🔴 STILL NOT A GENERALISATION VERDICT. n = 2 post-cutoff targets against 1
control, and 1STP is streptavidin -- one of the most over-represented structures
in the PDB, so "pre-cutoff" and "easy" are not separable here. 9WGR is a deletion
mutant of SufE, whose wild type these models have certainly seen. A real answer
needs MSAs for the post-cutoff targets (so the comparison is like-for-like with
how these models are actually used) and more than two of them.

CONFIDENCE BEHAVES WELL, MOSTLY, and that is worth as much as the RMSDs. openfold3
and boltz2 report 43-59 pLDDT exactly where they are wrong. But opendde (15.25 A
at pLDDT 86.4) and protenix2 (18.39 A at pLDDT 90.3) are CONFIDENTLY WRONG on
9XZ9 -- a worse failure than being unsure, and one that only appears off-
distribution: both track error well on 6MRR (confidence_parity.py, r 0.635 and
0.807). Anything that filters designs on pLDDT should know that.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/ColabDesign2')

ALL = ('openfold3', 'intellifold2', 'boltz2', 'opendde', 'protenix2',
       'rosettafold3')
# 1STP is the CONTROL and it is what makes the other two readable: a natural
# protein from 1992, deep inside every training set. If it also fails from a
# single sequence then these numbers are measuring MSA dependence, not memory.
TARGETS = ('9XZ9', '9WGR', '1STP')


def kabsch(P, Q):
  P = P - P.mean(0); Q = Q - Q.mean(0)
  V, _, Wt = np.linalg.svd(P.T @ Q)
  U = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ Wt))]) @ Wt
  return float(np.sqrt(((P @ U - Q) ** 2).sum(-1).mean()))


def target(code):
  import gemmi
  st = gemmi.read_structure(glob.glob('/home/ubuntu/**/%s.cif' % code,
                                      recursive=True)[0])
  st.setup_entities()
  st.remove_ligands_and_waters()
  st.remove_alternative_conformations()
  st.remove_hydrogens()
  ch = st[0]['A']
  seq = ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                for r in ch)
  ca = np.array([[r['CA'][0].pos.x, r['CA'][0].pos.y, r['CA'][0].pos.z]
                 for r in ch if r.find_atom('CA', '*')], np.float32)
  return seq, ca


def main():
  import jax
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  from colabdesign2.af3.runner import AF3Runner
  from tools.module_trace import models

  recycles = int(os.environ.get('RECYCLES', '10'))
  codes = (os.environ.get('TARGETS') or ','.join(TARGETS)).split(',')
  print('post-cutoff targets (released 2026-09-02), single sequence, %d recycles'
        % recycles)
  hdr = '%-14s' % 'model'
  for c in codes:
    hdr += ' %14s' % (c + ' RMSD/pLDDT')
  print(hdr)

  for m in (os.environ.get('MODELS').split(',') if os.environ.get('MODELS')
            else list(ALL)):
    row = '%-14s' % m
    for code in codes:
      try:
        seq, ref = target(code)
        L = len(seq)
        kw = models.featurise_kwargs(m)
        if 'struct_num_tokens' in kw:
          kw['struct_num_tokens'] = int(os.environ.get('STRUCT_TOKENS', '512'))
        b = f3.featurise_spec(parse_contigs('%d' % L).resolve(),
                              sequences={0: seq}, msa_crop_size=1, **kw)
        cfg, params, _f, _x = models.build(m, b, num_recycles=recycles,
                                           diffusion_steps=200, num_msa=1)
        r = AF3Runner(cfg=cfg, model_params=params, diffusion='forward',
                      num_msa=1)
        out = r.predict(b, key=jax.random.PRNGKey(1))
        x = np.asarray(out['diffusion_samples']['atom_positions'])
        pl = np.asarray(out['predicted_lddt'])
        while x.ndim > 3:
          x = x[0]
        if m == 'opendde':
          from colabdesign2.af3 import structural_features as sf
          g = np.asarray(b['structbook/residue_atom_gather'])
          x = np.asarray(sf.structural_to_residue_positions(x, g))
          d = np.asarray(b['struct/ref_mask']).shape[1]
          while pl.ndim > 1:
            pl = pl[0]
          pl = np.asarray(sf._gather_struct_feature(pl.reshape(-1, d), g))
        while pl.ndim > 2:
          pl = pl[0]
        n = min(L, len(ref))
        ca = pl[:n, 1] if pl.ndim == 2 else pl[:n]
        row += ' %7.2f/%5.1f' % (kabsch(x[:n, 1], ref[:n]), float(ca.mean()))
      except Exception as e:                                # noqa: BLE001
        row += ' %13s' % ('FAIL:' + type(e).__name__[:8])
    print(row, flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
