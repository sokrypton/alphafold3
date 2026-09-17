"""Ours vs native, one panel per model, from the sweep plus the 2026-09-03 natives.

The native files do NOT all measure the same thing, and mixing them silently
would be the whole error. Only forward-only (model-only) numbers are plotted:

* native_verified_64.tsv is already forward-only, featurisation subtracted --
  it is the canonical 64-token column.
* native_boltz2.tsv agrees with it at 64 (7.594 vs 7.590), so that curve is
  forward-only throughout and is plotted whole.
* native_rf3.tsv steady INCLUDES featurisation; only native_rf3_split.tsv
  reports steady-minus-featurise, so just its three lengths are plotted.
* opendde and protenix2 have two values each from a torch/cuequivariance A/B.
  The torch ones are used, being the lower and the ones in the verified table.
* chai-1 buckets to 256, so its 64/128/192 all cost the same ~38 s. Plotted as
  measured, with the bucket noted -- it is a property of the model, not noise.
"""
import collections, statistics, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OURS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else 'ours_vs_native.png'

# Natives are parsed from run_native_sweep.sh's TSV when one is given, else
# they fall back to the 2026-09-03 single points below.
#
# WHICH FIELD, per harness -- getting this wrong is the whole error:
#   rosettafold3  `steady-minus-featurise`, because its steady INCLUDES
#                 featurisation and ours does not.
#   others        `steady`, already model-only and warm.
# FAILED / OOM / NO_TIMINGS rows are dropped, not plotted as zero.
FALLBACK = {
    'boltz2':       {64: 7.594, 128: 8.669, 192: 11.639, 256: 17.151, 384: 38.686},
    'rosettafold3': {64: 11.725, 192: 18.382, 384: 56.190},
    'openfold3':    {64: 12.880},
    'protenix2':    {64: 10.398},
    'opendde':      {64: 8.425},
    'chai1':        {64: 37.967, 128: 38.177, 192: 38.470, 256: 28.394},
}
NOTE = {'chai1': 'buckets to 256', 'rosettafold3': 'featurisation subtracted'}


def parse_native(path):
    import re
    out = collections.defaultdict(dict)
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        if len(f) < 3 or any(k in f[2] for k in ('FAILED', 'OOM', 'NO_TIMINGS')):
            continue
        m, L, txt = f[0], int(f[1]), f[2]
        key = 'steady-minus-featurise' if 'steady-minus-featurise' in txt else 'steady'
        hit = re.search(key + r'\s+([0-9.]+)', txt)
        if hit:
            out[m][L] = float(hit.group(1))
    return out


NATIVE = dict(FALLBACK)
if len(sys.argv) > 3:
    parsed = parse_native(sys.argv[3])
    for m, d in parsed.items():
        NATIVE[m] = d                      # a fresh same-session curve wins outright
    print('natives from', sys.argv[3], '->',
          {m: len(d) for m, d in parsed.items()})

ours = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(OURS):
    f = line.rstrip('\n').split('\t')
    if len(f) < 6 or f[5] in ('OOM', 'FAILED'):
        continue
    try:
        ours[f[0]][int(f[1])].append(float(f[5]))
    except ValueError:
        pass

models = [m for m in NATIVE if m in ours and NATIVE[m]]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
for ax, m in zip(axes.flat, models):
    L = sorted(ours[m])
    y = [statistics.mean(ours[m][l]) for l in L]
    ax.plot(L, y, 'o-', color='#1f77b4', label='ours (jax)', zorder=3)
    nl = sorted(NATIVE[m])
    ax.plot(nl, [NATIVE[m][l] for l in nl], 's--', color='#d62728', label='native')
    for l in nl:                       # speedup where both exist
        if l in ours[m]:
            r = NATIVE[m][l] / statistics.mean(ours[m][l])
            ax.annotate(f'{r:.1f}x', (l, NATIVE[m][l]), textcoords='offset points',
                        xytext=(4, 6), fontsize=8, color='#d62728')
    ax.set_title(m + (f'  ({NOTE[m]})' if m in NOTE else ''), fontsize=10)
    ax.set_xticks(L); ax.set_xticklabels(L)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel('tokens')
for ax in axes[:, 0]:
    ax.set_ylabel('steady-state seconds')
fig.suptitle('Steady-state runtime, ours vs native (A10, forward-only, 3 recycles)',
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print('wrote', OUT)
for m in models:
    both = [(l, NATIVE[m][l] / statistics.mean(ours[m][l]))
            for l in sorted(NATIVE[m]) if l in ours[m]]
    print(f'{m:14} ' + '  '.join(f'{l}:{r:.2f}x' for l, r in both))
