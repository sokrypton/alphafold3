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
  FLOOR    below the cell's OWN resolution -- see below

A cell can also have no resolution left. `trunk_parity --blocks 48` runs a
48-block stack on synthetic input, and three models read BAD or LOOSE there
while a 1e-6 relative perturbation of the gate's INPUT moves its own output
FURTHER than our port does. Grading those cells on the number is grading noise.

So a gate may emit `<name>_floor` rows next to its `<name>` rows -- the same
comparison against a perturbed replicate of itself -- and any row at or below
its own floor is graded FLOOR rather than BAD. That is not a pass: it says the
cell cannot answer, and the answer has to come from a shallower or better
conditioned one (`L1.trunk1`, at one block, for this gate).

Earning the exemption requires the gate to MEASURE the floor. A cell with no
floor row is graded on its number as before, so this cannot quietly excuse
anything.

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

# `<name> corr <value>`, and then whatever relative measure the gate happens to
# print. THREE formats are in use and this file used to read only one of them:
#
#   trunk_parity / msa_parity   max|d| X ... rms(native) Y   -> ratio = X/Y
#   prot_parity                 relerr Z                     -> ratio = Z
#   the esmfold2 reference gate corr only, no magnitude       -> ratio = None
#
# and a name may contain '->' (`msa -> pair`). Requiring `max|d|` and excluding
# '>' from the name silently dropped 36 comparisons -- every L1b cell plus
# L1.trunk_ref, which is esmfold2's ONLY trunk gate. A parser that skips a line
# it does not recognise is the same failure as a driver that does not run a gate
# ([[harness-rot]]), so `--strict` now reports unparsed `corr` lines instead of
# ignoring them.
# The name is OPTIONAL: the esmfold2 reference gate prints a bare
# `corr ... relerr ...` under a heading line, and four such rows went ungraded
# because the name was required. Nameless rows are labelled with the heading
# that precedes them.
_LINE = re.compile(
    r'^\s{2,}(?:(?P<name>[A-Za-z_][\w()\'/ .>-]*?)\s+)?corr\s+(?P<corr>[-\d.]+)'
    r'(?P<rest>.*)$')
_HEAD = re.compile(r'^\S.*?(?P<h>[\w ]+?)\s*(?:\(|:|$)')
_MAXD = re.compile(r'max\|d\|\s+([\d.eE+-]+)')
_RMS = re.compile(r'rms\(native\)\s+([\d.eE+-]+)')
_RELERR = re.compile(r'relerr\s+([\d.eE+-]+)')


def _ratio(rest):
  """The row's relative error, whichever way its gate spells it."""
  md, rms = _MAXD.search(rest), _RMS.search(rest)
  if md and rms and float(rms.group(1)):
    return float(md.group(1)) / float(rms.group(1))
  re_ = _RELERR.search(rest)
  if re_:
    return float(re_.group(1))
  return None


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


def audit(logdir, unparsed=None):
  rows = []
  floors = collections.defaultdict(dict)
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
    heading = ''
    for line in open(os.path.join(logdir, f), errors='ignore'):
      if line[:1] not in (' ', '\t', '\n') and 'corr' not in line:
        h = _HEAD.match(line.rstrip('\n'))
        heading = (h.group('h').strip().lower().replace(' ', '_')
                   if h else '') or heading
      m = _LINE.match(line.rstrip('\n'))
      if not m:
        if unparsed is not None and 'corr' in line:
          unparsed.append((gate, model, line.strip()))
        continue
      corr = float(m.group('corr'))
      ratio = _ratio(m.group('rest'))
      name = (m.group('name') or heading or '?').strip()
      if name.endswith('_floor'):
        floors[(gate, model)][name[:-len('_floor')]] = ratio
        continue
      rows.append([gate, model, name, corr, ratio, grade(corr, ratio)])

  # Second pass: a row at or below its own measured floor is unresolvable.
  for r in rows:
    f = floors.get((r[0], r[1]), {}).get(r[2])
    if f is not None and r[4] is not None and r[4] <= f and r[5] != 'PARITY':
      r[5] = 'FLOOR'
  return [tuple(r) for r in rows]


def main(argv):
  logdir = argv[1] if len(argv) > 1 else 'dev/oracles/parity_runs/2026-09-08'
  unparsed = []
  rows = audit(logdir, unparsed)
  t = collections.Counter(r[5] for r in rows)
  print('%s: %d comparisons in %d logs' % (logdir, len(rows),
                                           len({(r[0], r[1]) for r in rows})))
  print('  ' + '   '.join('%s=%d' % (k, t[k])
                          for k in ('PARITY', 'CLOSE', 'FLOOR', 'LOOSE',
                                    'BAD')
                          if t[k]))
  fl = [r for r in rows if r[5] == 'FLOOR']
  if fl:
    print('\nBELOW THE CELL\'S OWN RESOLUTION -- the gate cannot answer here:')
    for gate, model, name, corr, ratio, _ in sorted(fl, key=lambda r: -(r[4] or 0)):
      print('  %-5s %-20s %-26s %-16s corr %.6f  max|d|/rms %.2e'
            % ('FLOOR', gate, model, name, corr, ratio))
  bad = [r for r in rows if r[5] in ('LOOSE', 'BAD')]
  if bad:
    print('\nNOT at parity, worst first (these all count as OK today):')
    for gate, model, name, corr, ratio, g in sorted(
        bad, key=lambda r: -(r[4] or 0))[:40]:
      print('  %-5s %-20s %-26s %-16s corr %.6f  max|d|/rms %s'
            % (g, gate, model, name, corr,
               '%.2e' % ratio if ratio is not None else 'n/a'))
  if unparsed:
    print('\n%d `corr` line(s) THIS FILE COULD NOT READ -- a comparison it '
          'cannot parse is a comparison nobody grades:' % len(unparsed))
    for gate, model, line in unparsed[:10]:
      print('  %-20s %-26s %s' % (gate, model, line[:70]))
  return 1 if (t['BAD'] or unparsed) else 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
