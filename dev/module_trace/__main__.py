"""CLI: python -m tools.module_trace {list,run,compare,show}"""
import argparse
import json
import os
import sys

DEFAULT_ROOT = '~/module_traces'


def main(argv=None):
  ap = argparse.ArgumentParser(prog='module_trace')
  sub = ap.add_subparsers(dest='cmd', required=True)

  sub.add_parser('list', help='cases + models and whether their inputs exist')

  r = sub.add_parser('run', help='trace one (model, case)')
  r.add_argument('--model', required=True)
  r.add_argument('--case', default='6mrr_single_seq')
  r.add_argument('--label', default=None, help='trace name (default: timestamp)')
  r.add_argument('--out', default=DEFAULT_ROOT)
  r.add_argument('--num-recycles', type=int, default=0)
  r.add_argument('--diffusion-steps', type=int, default=200)
  r.add_argument('--seed', type=int, default=1)
  r.add_argument('--max-calls-per-site', type=int, default=8,
                 help='fingerprint at most N calls per module (all are counted)')
  r.add_argument('--save-arrays', action='store_true',
                 help='also write modules.npz (arrays <= 8 MB) for debugging')

  c = sub.add_parser('compare', help='diff two traces')
  c.add_argument('a')
  c.add_argument('b')
  c.add_argument('--tol', type=float, default=1e-6)
  c.add_argument('--section', default='modules', choices=('modules', 'outputs'))
  c.add_argument('--limit', type=int, default=40)

  s = sub.add_parser('show', help='summarise one trace')
  s.add_argument('path')

  t = sub.add_parser('table', help='one row per stored trace')
  t.add_argument('--out', default=DEFAULT_ROOT)

  a = ap.parse_args(argv)
  # tokamax/absl parse sys.argv lazily on first use and abort on flags they do
  # not own, so hide ours once they are parsed.
  sys.argv = sys.argv[:1]
  from . import cases, models

  if a.cmd == 'list':
    print('CASES')
    for n, c in cases.CASES.items():
      ok, why = cases.available(n)
      print('  %-30s %-7s %-26s %s' % (n, c['status'], ','.join(c['uses']) or '-',
                                       'ok' if ok else why))
    print('MODELS')
    for m in models.CHECKPOINTS:
      ok, why = models.available(m)
      print('  %-30s %s' % (m, 'ok' if ok else why))
    return 0

  if a.cmd == 'run':
    from . import trace
    d, man = trace.run(a.model, a.case, a.out, label=a.label,
                       num_recycles=a.num_recycles,
                       diffusion_steps=a.diffusion_steps, seed=a.seed,
                       save_arrays=a.save_arrays,
                       max_calls_per_site=a.max_calls_per_site)
    print('wrote', d)
    print('  %d call sites, %d module calls, %d outputs, %.0fs'
          % (man['n_call_sites'], man['n_module_calls'],
             len(man['outputs']), man['wall_seconds']))
    if man['channels_unmet']:
      print('  WARNING declared-but-empty input channels:', man['channels_unmet'])
    if man['n_params_left_at_init']:
      print('  %d params left at init (unported):' % man['n_params_left_at_init'],
            ', '.join(man['params_left_at_init'][:5]), '...')
    print('  metrics', json.dumps(man['metrics']))
    return 0

  if a.cmd == 'compare':
    from . import compare
    print(compare.report(compare.compare(a.a, a.b, tol=a.tol,
                                         section=a.section, limit=a.limit)))
    return 0

  if a.cmd == 'table':
    from . import compare
    root = os.path.expanduser(a.out)
    rows = []
    for model in sorted(os.listdir(root)):
      for case in sorted(os.listdir(os.path.join(root, model))):
        for label in sorted(os.listdir(os.path.join(root, model, case))):
          d = os.path.join(root, model, case, label)
          try:
            m = compare.load(d)
          except Exception:
            continue
          met = m.get('metrics', {})
          rows.append((model, case, label, m['n_call_sites'], m['n_module_calls'],
                       m.get('n_params_left_at_init', 0),
                       met.get('ca_rmsd'), met.get('ca_ca_mean')))
    print('%-14s %-18s %-10s %6s %10s %7s %8s %8s'
          % ('model', 'case', 'label', 'sites', 'calls', 'atinit', 'rmsd', 'ca-ca'))
    for r in rows:
      print('%-14s %-18s %-10s %6d %10d %7d %8s %8s'
            % (r[0], r[1], r[2], r[3], r[4], r[5],
               ('%.2f' % r[6]) if r[6] is not None else '-',
               ('%.2f' % r[7]) if r[7] is not None else '-'))
    return 0

  if a.cmd == 'show':
    from . import compare
    m = compare.load(a.path)
    print('%s / %s / %s  (%s, %.0fs)' % (m['model'], m['case'], m['label'],
                                         m['created'], m['wall_seconds']))
    print('  %d call sites, %d module calls, %d outputs'
          % (m['n_call_sites'], m['n_module_calls'], len(m['outputs'])))
    print('  channels live:', {k: v for k, v in m['channels_live'].items() if v is not None})
    print('  metrics:', json.dumps(m['metrics']))
    return 0


if __name__ == '__main__':
  sys.exit(main())
