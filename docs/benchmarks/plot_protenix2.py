"""protenix2 alone: ours vs native across length, on one axis."""
import re, os, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = sys.argv[1]
HERE = '/home/ubuntu/alphafold3/docs/benchmarks'

ours = {}
for l in open(f'{S}/sweep_a10.tsv'):
    f = l.split('\t')
    if len(f) >= 6 and f[0] == 'protenix2' and f[1].isdigit():
        ours[int(f[1])] = float(f[5])
a100 = {}
for l in open(f'{S}/sweep_a100.tsv'):
    f = l.split('\t')
    if len(f) >= 6 and f[0] == 'protenix2' and f[1].isdigit():
        a100[int(f[1])] = float(f[5])
nat = {64: 10.398}                     # verified on a quiet machine, >=6 calls
for l in open(f'{S}/native_sweep.tsv'):
    f = l.rstrip('\n').split('\t')
    if f[0] != 'protenix2':
        continue
    m = re.search(r'steady\s+([\d.]+)', f[2]) if len(f) > 2 else None
    if m:
        nat[int(f[1])] = float(m.group(1))

ox = sorted(ours); nx = sorted(nat); ax_ = sorted(a100)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

ax1.plot(ox, [ours[L] for L in ox], 'o-', ms=6, label='ours (jax), A10')
ax1.plot(nx, [nat[L] for L in nx], '^--', ms=7, color='crimson',
         label='native protenix-v2 (torch), A10')
ax1.plot(ax_, [a100[L] for L in ax_], 's:', ms=5, color='tab:orange',
         label='ours (jax), A100')
ax1.set_xlabel('tokens'); ax1.set_ylabel('s / prediction')
ax1.set_title('protenix2: runtime vs length'); ax1.grid(alpha=.3); ax1.legend(fontsize=8)

shared = [L for L in ox if L in nat]
ax2.plot(shared, [nat[L] / ours[L] for L in shared], 'o-', color='crimson', ms=6)
ax2.axhline(1.0, color='k', lw=.9, ls=':')
ax2.set_xlabel('tokens'); ax2.set_ylabel('native / ours   (>1 = we are faster)')
ax2.set_title('advantage decays to parity by ~384 tokens'); ax2.grid(alpha=.3)
for L in shared:
    ax2.annotate(f'{nat[L]/ours[L]:.2f}x', (L, nat[L]/ours[L]),
                 textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
fig.suptitle('protenix2 has pair_channel=256 (double every other port), so its O(L^2) work '
             'is 2x -- it becomes arithmetic-bound early\n'
             'and our fixed per-op advantage amortises away', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(f'{HERE}/protenix2_vs_native.png', dpi=130)
print('wrote protenix2_vs_native.png')
print(f'{"L":>5s} {"ours":>7s} {"native":>8s} {"ratio":>7s}')
for L in shared:
    print(f'{L:5d} {ours[L]:7.2f} {nat[L]:8.2f} {nat[L]/ours[L]:6.2f}x')
