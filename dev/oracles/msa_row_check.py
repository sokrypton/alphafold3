"""Does the featuriser's MSA say what the a3m said? A pure-numpy gate.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:. \
    ~/venv/bin/python dev/oracles/msa_row_check.py esmfold2_exp

Every MSA gate in this file's neighbourhood compares ACTIVATIONS, and all of
them inject the vendor's own embedded rows so that both sides start from an
identical tensor -- which is what makes them clean module gates and what makes
them blind to the rows themselves. This one compares the featurised batch
against the alignment it came from: row content, row ORDER, the query's
position, and the deletion counts.

What it found on 1STP (2144 a3m rows + the query):

  * every row we build is a real a3m row -- 2145 of 2145 present -- so the
    letter mapping, the gap class and the lowercase-insertion stripping are all
    right,
  * but only 21 rows sit where the a3m put them: the pipeline REORDERS. An a3m
    is ordered by similarity to the query, and a truncation to num_msa then
    keeps a different 1024 sequences than the vendor's would,
  * and the query is row 0 in the BATCH -- it is the model's own subsampling
    that can move or drop it (see `featurization.subsample_msa_keep_query` and
    model_config.MSA_KEEP_QUERY_ROW).

Non-polymer tokens are reported separately rather than compared: AF3 puts a GAP
on a ligand token in every row, and whether that is the vendor's convention is a
per-model question this gate deliberately does not answer.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_AA3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']


def encode(row, one, unk, gap):
  """One a3m row -> (class ids over the QUERY columns, deletion counts).

  Lowercase letters are INSERTIONS relative to the query: they occupy no query
  column and instead count into the deletion features of the column that
  follows. Keeping them would make every row longer than the query and the
  comparison would silently fall through a length filter.
  """
  ids, dele, pend = [], [], 0
  for ch in row:
    if ch.islower():
      pend += 1
      continue
    ids.append(gap if ch == '-' else one.get(ch.upper(), unk))
    dele.append(pend)
    pend = 0
  return ids, dele


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--json', default=os.path.expanduser('~/1stp_msa.json'))
  args = ap.parse_args(argv)
  sys.argv = sys.argv[:1]

  import fold_check
  from alphafold3.common import folding_input
  from alphafold3.constants import residue_names as R
  from alphafold3.model import feat_batch

  raw = open(args.json).read()
  d = json.loads(raw)
  prot = [c['protein'] for c in d['sequences'] if 'protein' in c][0]
  seq, a3m = prot['sequence'], prot.get('unpairedMsa', '')
  a3m_rows = [l.strip() for l in a3m.splitlines()
              if l.strip() and not l.startswith('>')]
  order = R.POLYMER_TYPES_ORDER_WITH_ALL_UNKS_AND_GAP
  one = {a: order[t] for a, t in zip('ARNDCQEGHILKMFPSTWYV', _AA3)}
  unk, gap = order[R.UNK], order['-']
  exp = [encode(r, one, unk, gap) for r in a3m_rows]
  exp = [e for e in exp if len(e[0]) == len(seq)]
  print('%s: %d a3m rows (%d of query length %d)'
        % (args.model, len(a3m_rows), len(exp), len(seq)))

  fi = folding_input.Input.from_json(raw)
  batch, _, _ = fold_check._fold_setup(args.model, seq, None,
                                       chains=list(fi.chains))
  fb = feat_batch.Batch.from_data_dict(batch)
  rows = np.asarray(fb.msa.rows)
  mask = np.asarray(fb.msa.mask)
  dm = np.asarray(fb.msa.deletion_matrix)
  asym = np.asarray(fb.token_features.asym_id)
  real = int((mask.sum(1) > 0).sum())
  cols = np.nonzero(asym == asym[0])[0]
  ours, oursd = rows[:real][:, cols], dm[:real][:, cols]
  print('  batch: %d real rows of %d, %d tokens (%d in the first chain)'
        % (real, mask.shape[0], mask.shape[1], len(cols)))
  lig = np.nonzero(asym != asym[0])[0]
  if len(lig):
    print('  non-polymer tokens: %d, msa classes %s (AF3 gap is %d)'
          % (len(lig), np.unique(rows[:real][:, lig]), gap))

  print('  query is row 0: %s' % (list(ours[0]) == exp[0][0]))
  n = min(real, len(exp))
  in_order = sum(list(ours[i]) == exp[i][0] for i in range(n))
  print('  rows in the a3m ORDER: %d of %d' % (in_order, n))
  have = collections.Counter(tuple(e[0]) for e in exp)
  missing = sum(1 for r in ours if tuple(r) not in have)
  print('  rows that are NOT an a3m row: %d of %d' % (missing, real))
  print('  deletion counts, total nonzero: ours %d, a3m %d'
        % (int((oursd > 0).sum()),
           int(sum(np.count_nonzero(e[1]) for e in exp))))
  return 1 if missing else 0


if __name__ == '__main__':
  sys.exit(main())
