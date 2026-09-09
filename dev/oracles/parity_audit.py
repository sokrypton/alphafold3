"""Are the OK cells actually at PARITY? The driver never checks a NUMBER.

  python3 dev/oracles/parity_audit.py dev/oracles/parity_runs/2026-09-08

`run_all_parity.sh`'s `classify()` asks whether a line matching the gate's
pattern EXISTS. It never reads the value. So a cell whose comparison says

    p_atom_pair corr 0.967820  max|d| 170.24429  rms(native) 17.659

is reported OK, and a summary of "234 OK" is a count of gates that RAN, not of
gates that AGREE. That is the same blindness as counting SKIPs without asking
whether the module exists -- one level up.

This reads every `corr` line out of the logs and grades it:

  PARITY   corr >= 0.99999 and max|d|/rms <= 1e-3
  CLOSE    corr >= 0.9999  and max|d|/rms <= 1e-2
  LOOSE    corr >= 0.999
  BAD      anything else

The thresholds are deliberately stricter than "0.999 looks fine". This file's
own history is the argument: chai1's dropped template bias sat at corr 0.999648
and was 35.6% of the module's output, and the ESMFold2 OPM bias was a
per-channel constant at corr 0.999836. `max|d|/rms` is what catches those --
[[correlation-hides-bias]].
"""
import os
import re
import sys
import collections

_LINE = re.compile(
    r'^\s{2,}(?P<name>[A-Za-z_][\w()\'/ .-]*?)\s+corr\s+(?P<corr>[-\d.]+)'
    r'.*?max\|d\|\s+(?P<maxd>[\d.eE+-]+)'
    r'(?:.*?rms\(native\)\s+(?P<rms>[\d.eE+-]+))?')


def grade(corr, ratio):
  if ratio is None:
    return 'PARITY' if corr >= 0.99999 else (
        'CLOSE' if corr >= 0.9999 else 'LOOSE' if corr >= 0.999 else 'BAD')
  if corr >= 0.99999 and ratio <= 1e-3:
    return 'PARITY'
  if corr >= 0.9999 and ratio <= 1e-2:
    return 'CLOSE'
  if corr >= 0.999 and ratio <= 1e-1:
    return 'LOOSE'
  return 'BAD'


def audit(logdir):
  rows = []
  for f in sorted(os.listdir(logdir)):
    if not f.endswith('.log'):
      continue
    stem = f[:-4]
    # <gate>.<model>: the gate never contains a dot after the level prefix, so
    # split on the FIRST dot past the level.
    parts = stem.split('.')
    gate = '.'.join(parts[:2]) if len(parts) > 2 else parts[0]
    model = '.'.join(parts[2:]) if len(parts) > 2 else (
        parts[1] if len(parts) > 1 else '?')
    for line in open(os.path.join(logdir, f), errors='ignore'):
      m = _LINE.match(line.rstrip('\n'))
      if not m:
        continue
      corr = float(m.group('corr'))
      maxd = float(m.group('maxd'))
      rms = m.group('rms')
      ratio = (maxd / float(rms)) if rms and float(rms) else None
      rows.append((gate, model, m.group('name').strip(), corr, ratio,
                   grade(corr, ratio)))
  return rows


def main(argv):
  logdir = argv[1] if len(argv) > 1 else 'dev/oracles/parity_runs/2026-09-08'
  rows = audit(logdir)
  t = collections.Counter(r[5] for r in rows)
  print('%s: %d comparisons in %d logs' % (logdir, len(rows),
                                           len({(r[0], r[1]) for r in rows})))
  print('  ' + '   '.join('%s=%d' % (k, t[k])
                          for k in ('PARITY', 'CLOSE', 'LOOSE', 'BAD')
                          if t[k]))
  bad = [r for r in rows if r[5] in ('LOOSE', 'BAD')]
  if bad:
    print('\nNOT at parity, worst first (these all count as OK today):')
    for gate, model, name, corr, ratio, g in sorted(
        bad, key=lambda r: -(r[4] or 0))[:40]:
      print('  %-5s %-20s %-26s %-16s corr %.6f  max|d|/rms %s'
            % (g, gate, model, name, corr,
               '%.2e' % ratio if ratio is not None else 'n/a'))
  return 1 if t['BAD'] else 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
