"""Keep the LAST row per (gate, model) in a run's summary.tsv.

  python3 dev/oracles/dedupe_summary.py dev/oracles/parity_runs/<run>

Why this exists rather than a hand-written filter: re-running one level with
FORCE=1 into an existing LOGDIR appends a second row for every cell it redoes,
which is correct on disk (the log IS re-classified) and wrong in a count. The
hand-written version of this was `grep -v "^L1.trunk_ref\tesmfold2"`, which is
a PREFIX match -- it deleted four rows where one was meant, including the
esmfold2_fast and esmfold2_lm600m rows, and the count that followed was quietly
wrong. Splitting on the tab and keying the pair cannot do that.

The last row wins because a re-run is a correction: L0.audit read FAIL for all
14 models on 2026-09-09 from a syntax-level regression in dev/audit_coverage.py
(the ESMFold2 purge left four dangling one-element tuples), and the re-run after
the fix is the answer.
"""
import collections
import os
import sys


def main(argv):
  if len(argv) != 2:
    raise SystemExit(__doc__)
  path = argv[1]
  if os.path.isdir(path):
    path = os.path.join(path, 'summary.tsv')
  rows = open(path).read().splitlines()
  header = rows[0] if rows and rows[0].startswith('gate\t') else None
  body = rows[1:] if header else rows
  last, order = {}, []
  for r in body:
    parts = r.split('\t')
    if len(parts) < 3:
      continue
    key = (parts[0], parts[1])
    if key not in last:
      order.append(key)
    last[key] = r
  dropped = len(body) - len(last)
  with open(path, 'w') as f:
    if header:
      f.write(header + '\n')
    for key in order:
      f.write(last[key] + '\n')
  print('%s: %d rows -> %d cells (%d superseded row(s) dropped)'
        % (path, len(body), len(last), dropped))
  return 0


if __name__ == '__main__':
  raise SystemExit(main(sys.argv))
