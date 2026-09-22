"""Is there a systematic PRE- vs POST-cutoff gap? Matched sets, one protocol.

postcutoff_parity.py measured two post-cutoff targets against one control and
found the reading depends on the model: openfold3 fails a PRE-cutoff protein too
(so its post-cutoff numbers measure MSA dependence, not memory), while boltz2
SOLVES the pre-cutoff control from a single sequence and fails both post-cutoff
targets. With n = 2 vs 1 that is an observation, not a result. This runs matched
SETS so the comparison can carry weight.

Both sets are single protein entity, 60-140 residues, resolution < 2.0 A, and NO
non-polymer entities, drawn from the RCSB search API and de-duplicated by
sequence -- the raw listings contain series from one study (10XD/10XF/10XG,
6EXT/6EXU/6EXX) and counting one protein three times would weight it three times.

  post: released after 2026-01-01, i.e. after every checkpoint here was trained
  pre:  released before 2019-01-01, i.e. deep inside every training set

Single sequence, no MSA, no template -- the setting design actually uses, and the
setting in which the pre-set tells you what "as good as it gets" looks like for
this model. The comparison is only meaningful because both sets go through the
identical protocol; a post-cutoff failure means something only against a pre-
cutoff success on a protein of the same kind and size.

  MODELS=boltz2 PYTHONPATH=/home/ubuntu/ColabDesign2 \\
      ~/venv/bin/python tools/oracles/cutoff_study.py

STOPPED PART-WAY, and what it found first is worth more than the numbers.

boltz2, single sequence, 10 recycles. The PRE set completed:

  6IGW  1.12/88.7   6IPY  2.63/87.3   6E4N  2.75/77.8   6CB6  9.91/52.0
  6EXU 12.02/61.8   6N2L 12.17/51.7   6BBK 12.96/43.1   6EXT 13.76/66.2
  -> 3 of 8 under 3 A, and pLDDT separates them cleanly (77-89 vs 43-66).

That alone kills the reading postcutoff_parity.py invited. boltz2 was the one
model there that looked like it had a pre/post gap, because it solved the single
PRE control (1STP) at 1.99 A and failed both post targets. With eight pre-cutoff
proteins it solves three. 1STP was an easy outlier -- streptavidin, one of the
most over-represented structures in the PDB -- not evidence of memorisation.

The first three POST targets then came back BETTER than the pre set:
  9TLN 1.03/96.1   9TLR 1.27/97.0   9W1D 2.11/93.5

🔴 WHY, AND WHY THIS DESIGN CANNOT ANSWER THE QUESTION. Checking the titles:
9TLN and 9TLR are DE NOVO DESIGNED coiled-coil hairpins, 9VGJ is an "artificially
designed" all-beta protein, 9W1D is a nanobody. The pre set is entirely natural
proteins. Matching on release date, length, resolution and entity count does NOT
match on the thing that decides difficulty here -- designed proteins are exactly
what single-sequence models are best at (6MRR, designed, folds at 1.5 A).

And a natural-only post set cannot be assembled from recent releases by filters
either. The pool is dominated by (a) de novo designed proteins, (b) RE-depositions
of famous proteins -- five of the most recent hits are lysozyme adducts -- and
(c) fragment-screening group depositions, ~40 entries of one 117-aa protein.
All three break either the "unseen" premise or the matching.

So a real generalisation study needs curation that a date+size filter cannot do:
natural proteins only, one entry per protein, and ideally a homology check
against the training window rather than a release-date proxy. Left here as a
recorded dead end, not a result.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/ColabDesign2')
MAN = os.path.expanduser('~/pdb_cutoff/manifest.json')


def kabsch(P, Q):
  P = P - P.mean(0); Q = Q - Q.mean(0)
  V, _, Wt = np.linalg.svd(P.T @ Q)
  U = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ Wt))]) @ Wt
  return float(np.sqrt(((P @ U - Q) ** 2).sum(-1).mean()))


def target(code, chain):
  import gemmi
  st = gemmi.read_structure(os.path.expanduser('~/pdb_cutoff/%s.cif' % code))
  st.setup_entities(); st.remove_ligands_and_waters()
  st.remove_alternative_conformations(); st.remove_hydrogens()
  ch = st[0][chain]
  rs = [r for r in ch if r.find_atom('CA', '*')
        and gemmi.find_tabulated_residue(r.name)]
  seq = ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                for r in rs)
  ca = np.array([[r['CA'][0].pos.x, r['CA'][0].pos.y, r['CA'][0].pos.z]
                 for r in rs], np.float32)
  return seq, ca


def main():
  import jax
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  from colabdesign2.af3.runner import AF3Runner
  from tools.module_trace import models

  man = json.load(open(MAN))
  recycles = int(os.environ.get('RECYCLES', '10'))
  for m in (os.environ.get('MODELS', 'boltz2').split(',')):
    print('=== %s, single sequence, %d recycles' % (m, recycles), flush=True)
    got = {}
    for tag in ('pre', 'post'):
      vals = []
      for code, chain, _n in man[tag]:
        try:
          seq, ref = target(code, chain)
          L = len(seq)
          kw = models.featurise_kwargs(m)
          if 'struct_num_tokens' in kw:
            kw['struct_num_tokens'] = 512
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
          rms = kabsch(x[:n, 1], ref[:n])
          conf = float((pl[:n, 1] if pl.ndim == 2 else pl[:n]).mean())
          vals.append((code, L, rms, conf))
          print('  %-4s %-6s %3d aa  RMSD %6.2f  pLDDT %5.1f'
                % (tag, code, L, rms, conf), flush=True)
        except Exception as e:                              # noqa: BLE001
          print('  %-4s %-6s FAILED %s' % (tag, code, type(e).__name__),
                flush=True)
      got[tag] = vals
    for tag in ('pre', 'post'):
      v = [x[2] for x in got[tag]]
      c = [x[3] for x in got[tag]]
      if v:
        print('%-5s n=%d  median RMSD %6.2f  under 3 A: %d/%d  median pLDDT %.1f'
              % (tag, len(v), float(np.median(v)),
                 sum(1 for y in v if y < 3.0), len(v), float(np.median(c))),
              flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
