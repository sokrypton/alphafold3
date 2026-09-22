"""Diff two traces module by module.

Verdict per module key:
  MATCH     identical fingerprint (bit-for-bit equal statistics AND projection)
  CLOSE     within tol (relative, scaled by the reference's absmax)
  DIFFER    beyond tol -- with the first-divergence marker, which is what you
            actually want: the earliest module that differs is the bug site;
            everything after it is downstream contamination.
  ONLY_A / ONLY_B   present in one trace and not the other (a graph change)
"""
import json
import os

import numpy as np


def load(path):
  path = os.path.expanduser(path)
  if os.path.isdir(path):
    path = os.path.join(path, 'manifest.json')
  with open(path) as fh:
    return json.load(fh)


def _delta_entry(a, b):
  """divergence between two CALL SITES (each a list of per-call fingerprints)."""
  if a['calls'] != b['calls']:
    return float('inf'), 'call count %d vs %d' % (a['calls'], b['calls'])
  worst, why = 0.0, ''
  for i, (fa, fb) in enumerate(zip(a['fp'], b['fp'])):   # capped at max_calls_per_site
    d, w = _delta(fa, fb)
    if d > worst:
      worst, why = d, (w if a['calls'] == 1 else 'call %d: %s' % (i, w))
  return worst, why


def _delta(a, b):
  """relative divergence between two fingerprints, scaled by the reference."""
  if a.get('shape') != b.get('shape'):
    return float('inf'), 'shape %s vs %s' % (a.get('shape'), b.get('shape'))
  scale = max(abs(a.get('absmax', 0.0)), 1e-12)
  worst, why = 0.0, ''
  for stat in ('mean', 'std', 'min', 'max', 'absmax'):
    if stat in a and stat in b:
      d = abs(a[stat] - b[stat]) / scale
      if d > worst:
        worst, why = d, stat
  pa, pb = a.get('proj'), b.get('proj')
  if pa and pb:
    n = max(np.abs(np.asarray(pa)).max(), 1e-12)
    d = float(np.abs(np.asarray(pa) - np.asarray(pb)).max() / n)
    if d > worst:
      worst, why = d, 'projection'
  return worst, why


def compare(a_path, b_path, tol=1e-6, section='modules', limit=40):
  A, B = load(a_path), load(b_path)
  ka, kb = A[section], B[section]
  keys = list(ka)                                  # A's order == execution order
  keys += [k for k in kb if k not in ka]
  rows, first_diff = [], None
  for k in keys:
    if k not in kb:
      rows.append((k, 'ONLY_A', 0.0, '')); continue
    if k not in ka:
      rows.append((k, 'ONLY_B', 0.0, '')); continue
    d, why = _delta_entry(ka[k], kb[k])
    if d == 0.0:
      v = 'MATCH'
    elif d <= tol:
      v = 'CLOSE'
    else:
      v = 'DIFFER'
      if first_diff is None:
        first_diff = k
    rows.append((k, v, d, why))
  summary = {v: sum(1 for r in rows if r[1] == v)
             for v in ('MATCH', 'CLOSE', 'DIFFER', 'ONLY_A', 'ONLY_B')}
  return dict(a=A['label'], b=B['label'], model=A['model'], case=A['case'],
              summary=summary, first_divergence=first_diff,
              rows=rows, metrics_a=A.get('metrics'), metrics_b=B.get('metrics'),
              limit=limit)


def report(res):
  out = ['%s / %s   %s vs %s' % (res['model'], res['case'], res['a'], res['b']),
         '  ' + '  '.join('%s=%d' % (k, v) for k, v in res['summary'].items())]
  if res['first_divergence']:
    out.append('  FIRST DIVERGENCE: ' + res['first_divergence'])
  bad = [r for r in res['rows'] if r[1] in ('DIFFER', 'ONLY_A', 'ONLY_B')]
  for k, v, d, why in bad[:res['limit']]:
    out.append('    %-9s %-11s %s' % (v, ('%.3g' % d) if d else '', k)
               + (('  [%s]' % why) if why else ''))
  if len(bad) > res['limit']:
    out.append('    ... and %d more' % (len(bad) - res['limit']))
  out.append('  metrics A: %s' % json.dumps(res['metrics_a']))
  out.append('  metrics B: %s' % json.dumps(res['metrics_b']))
  return '\n'.join(out)
