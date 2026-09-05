"""Peak device memory per (model, length), and why it is not the runtime story.

    python docs/benchmarks/plot_memory.py <dir holding mem_a10.tsv, sweep_a10.tsv>

Reads the PEAK column that bench_sweep.py records from
`memory_stats()['peak_bytes_in_use']` -- the allocator's high-water mark, not
nvidia-smi, which with JAX pre-allocating most of the card reports a constant
that says nothing about the model.

Two plots:
  memory_all_A10.png    peak GiB vs tokens, with the card's usable ceiling and
                        an X where a model OOMs
  memory_vs_runtime.png the same cells as (seconds, GiB) -- the point being that
                        the two do not rank the same way, so a runtime plot
                        cannot tell you which model will fit
"""
import os, sys, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = sys.argv[1] if len(sys.argv) > 1 else '.'
HERE = os.path.dirname(os.path.abspath(__file__))
CARD_GIB = 22.0   # usable on a 23 GB A10 once the context and params are resident


def load(path, col):
  """-> ({model: [(L, value)]}, {model: first OOM length})."""
  vals, oom = collections.defaultdict(list), {}
  if not os.path.exists(path):
    return vals, oom
  for line in open(path):
    f = line.rstrip('\n').split('\t')
    if len(f) < 3 or not f[1].isdigit():
      continue
    m, L = f[0], int(f[1])
    if f[2] == 'OOM':
      oom[m] = min(oom.get(m, 10 ** 9), L)
    elif len(f) > col and f[col]:
      try:
        vals[m].append((L, float(f[col])))
      except ValueError:
        pass
  for m in vals:
    vals[m].sort()
  return vals, oom


mem, mem_oom = load(f'{S}/mem_a10.tsv', 7)      # column 8: peak GiB
sec, sec_oom = load(f'{S}/sweep_a10.tsv', 5)    # column 6: steady seconds
if not mem:
  raise SystemExit('no mem_a10.tsv rows with a peak column; re-run the memory sweep')

order = sorted(mem, key=lambda m: -max(v for _, v in mem[m]))
cmap = plt.get_cmap('tab10')
colour = {m: cmap(i % 10) for i, m in enumerate(sorted(mem))}
# openfold3, openbind and rosettafold3 sit exactly on top of each other -- all
# three are pair_channel=128 with the same block counts, so their footprints
# agree to ~0.02 GiB. That is a result, not a plotting bug, but three curves
# under one line reads as one curve, so vary the dash.
STYLE = {'openfold3': '-', 'openbind0': '--', 'rosettafold3': ':'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
for m in order:
  xs = [p[0] for p in mem[m]]
  ys = [p[1] for p in mem[m]]
  for ax in (ax1, ax2):
    ax.plot(xs, ys, 'o' + STYLE.get(m, '-'), ms=4, color=colour[m], label=m)
  if m in mem_oom:                    # mark where it stopped fitting
    ax1.plot([mem_oom[m]], [CARD_GIB], 'x', ms=11, mew=2.5, color=colour[m])
for ax, t, sc in ((ax1, 'peak memory vs length (A10, 23 GB)', 'linear'),
                  (ax2, 'same, log-log', 'log')):
  ax.axhline(CARD_GIB, color='k', lw=.9, ls=':')
  ax.set_xscale(sc); ax.set_yscale(sc)
  ax.set_xlabel('tokens'); ax.set_ylabel('peak GiB'); ax.set_title(t)
  ax.grid(alpha=.3)
ax1.annotate('card limit (X = OOM)', (0.02, 0.86), xycoords='axes fraction',
             fontsize=8, color='dimgray')
ax1.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(f'{HERE}/memory_all_A10.png', dpi=130)
print('wrote memory_all_A10.png')

# Runtime does not predict footprint. Same cells, plotted against each other.
fig, ax = plt.subplots(figsize=(6.6, 5.2))
for m in sorted(mem):
  d = dict(sec.get(m, []))
  pts = [(d[L], g) for L, g in mem[m] if L in d]
  if not pts:
    continue
  ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o' + STYLE.get(m, '-'),
          ms=4, color=colour[m], label=m)
ax.axhline(CARD_GIB, color='k', lw=.9, ls=':')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('s / prediction'); ax.set_ylabel('peak GiB')
ax.set_title('footprint vs runtime — the orderings differ')
ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(f'{HERE}/memory_vs_runtime.png', dpi=130)
print('wrote memory_vs_runtime.png')

print('\npeak GiB by length')
Ls = sorted({L for m in mem for L, _ in mem[m]})
print(f'{"model":14s}' + ''.join(f'{L:>8d}' for L in Ls))
for m in order:
  d = dict(mem[m])
  row = ''.join(f'{d[L]:>8.2f}' if L in d else
                f'{"OOM":>8s}' if m in mem_oom and L >= mem_oom[m] else f'{"-":>8s}'
                for L in Ls)
  print(f'{m:14s}{row}')
