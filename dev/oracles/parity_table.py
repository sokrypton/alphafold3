"""Turn a run_all_parity.sh summary.tsv into the tables PARITY.md carries.

The driver's summary is (gate, model, status, headline) with the LAST verdict per
cell winning. This extracts the numbers so a table in PARITY.md is transcribed
rather than retyped -- retyping is how 86 became 87 in a commit message.

  python dev/oracles/parity_table.py dev/oracles/parity_runs/<date>/summary.tsv --level L6
"""
import argparse
import collections
import re
import sys


def cells(path):
  """-> {(gate, model): (status, headline)}, last verdict per cell."""
  out = {}
  with open(path) as fh:
    next(fh, None)
    for line in fh:
      parts = line.rstrip('\n').split('\t')
      if len(parts) < 3:
        continue
      gate, model, status = parts[0], parts[1], parts[2]
      out[(gate, model)] = (status, parts[3] if len(parts) > 3 else '')
  return out


_NUM = {
    # what to pull out of each gate's headline, in order of preference
    'best': re.compile(r'best ([0-9.]+)'),
    'mean': re.compile(r'mean ([0-9.]+)'),
    'ligand': re.compile(r'in-frame RMSD ([0-9.]+)'),
    'corr': re.compile(r'corr ([0-9.]+)'),
}


def number(headline):
  for key in ('ligand', 'best', 'corr'):
    m = _NUM[key].search(headline)
    if m:
      return key, m.group(1)
  return None, ''


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('summary')
  ap.add_argument('--level', default=None, help='prefix filter, e.g. L6 or L6.rna')
  ap.add_argument('--status', action='store_true', help='print status not numbers')
  a = ap.parse_args(argv)

  c = cells(a.summary)
  gates = sorted({g for g, _ in c if not a.level or g.startswith(a.level)})
  models = sorted({m for g, m in c if not a.level or g.startswith(a.level)})
  print('| model | ' + ' | '.join(g.split('.', 1)[-1] for g in gates) + ' |')
  print('|---|' + '---|' * len(gates))
  for m in models:
    row = []
    for g in gates:
      st, head = c.get((g, m), ('', ''))
      if a.status or st != 'OK':
        row.append(st.lower() if st else '—')
      else:
        _, v = number(head)
        row.append(v or 'ok')
    print('| `%s` | %s |' % (m, ' | '.join(row)))
  tally = collections.Counter(st for (g, _), (st, _) in c.items()
                              if not a.level or g.startswith(a.level))
  print()
  print('  ' + '  '.join('%s %d' % kv for kv in sorted(tally.items())))
  return 0


if __name__ == '__main__':
  sys.exit(main())
