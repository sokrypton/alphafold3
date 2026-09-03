"""Runtime plots for docs/benchmarks/, from the sweep TSVs.

  python docs/benchmarks/make_plots.py <scratch_dir>

Reads sweep_a10.tsv / sweep_a100.tsv (ours, one row per model x length) and
nb_final.tsv (native boltz-2, one row per length) and writes the PNGs next to
this script, i.e. INTO THE REPO -- a plot left in a session scratch directory is
gone with the session.
"""
import sys, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = sys.argv[1] if len(sys.argv) > 1 else '.'
HERE = os.path.dirname(os.path.abspath(__file__))

def load_ours(path):
  out = {}
  if not os.path.exists(path):
    return out
  for line in open(path):
    f = line.rstrip('\n').split('\t')
    if len(f) < 6 or f[-1] == 'FAIL' or f[0].startswith('SWEEP'):
      continue
    out.setdefault(f[0], []).append((int(f[1]), float(f[5])))
  for m in out:
    out[m].sort()
  return out

def load_native(path):
  pts = []
  if not os.path.exists(path):
    return pts
  for line in open(path):
    line = line.strip()
    if not line or line.startswith('NBF'):
      continue
    f = line.split('\t')
    try:
      L = int(f[0])
    except ValueError:
      continue
    m = re.search(r'steady\s+([\d.]+)s', line)
    if m:
      pts.append((L, float(m.group(1))))
    elif len(f) == 2:
      try:
        pts.append((L, float(f[1])))
      except ValueError:
        pass
  return sorted(pts)

a10 = load_ours(f'{S}/sweep_a10.tsv')
a100 = load_ours(f'{S}/sweep_a100.tsv')
nat = load_native(f'{S}/nb_final.tsv')

# ---------- ours vs native boltz-2 ---------------------------------------
if nat and 'boltz2' in a10:
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
  ox = [p[0] for p in a10['boltz2']]; oy = [p[1] for p in a10['boltz2']]
  nx = [p[0] for p in nat]; ny = [p[1] for p in nat]
  ax1.plot(ox, oy, 'o-', label='ours (JAX)', ms=5)
  ax1.plot(nx, ny, 's--', label='native boltz-2 (torch)', ms=5)
  ax1.set_xlabel('tokens'); ax1.set_ylabel('s / prediction')
  ax1.set_title('boltz-2: ours vs native, A10'); ax1.grid(alpha=.3); ax1.legend()
  shared = [(L, dict(a10['boltz2'])[L], t) for L, t in nat if L in dict(a10['boltz2'])]
  if shared:
    ax2.plot([s[0] for s in shared], [s[2] / s[1] for s in shared], 'o-', color='crimson')
    ax2.axhline(1.0, color='k', lw=.8, ls=':')
    ax2.set_xlabel('tokens'); ax2.set_ylabel('native / ours  (>1 = we are faster)')
    ax2.set_title('speedup vs native, by size'); ax2.grid(alpha=.3)
  fig.tight_layout()
  fig.savefig(f'{HERE}/boltz2_vs_native.png', dpi=130)
  print('wrote boltz2_vs_native.png')
  for L, o, n in shared:
    print(f'  L={L:4d}  ours {o:6.2f}s  native {n:6.2f}s  {n / o:.2f}x')

# ---------- ours vs native rosettafold3 ----------------------------------
def load_rf3(path):
  """-> {L: (total, featurise, fwd_plus_io)} from the split harness."""
  out = {}
  if not os.path.exists(path):
    return out
  import re as _re
  for line in open(path):
    f = line.split('\t')
    try:
      L = int(f[0])
    except (ValueError, IndexError):
      continue
    tot = _re.search(r'steady\s+([\d.]+)s', line)
    ft = _re.search(r'featurise\s+([\d.]+)s', line)
    fw = _re.search(r'steady-minus-featurise\s+([\d.]+)s', line)
    if tot:
      out[L] = (float(tot.group(1)),
                float(ft.group(1)) if ft else None,
                float(fw.group(1)) if fw else None)
  return out

rf3_tot = load_rf3(f'{S}/nr_final.tsv')
rf3_tot.update({k: v for k, v in load_rf3(f'{S}/nr_split.tsv').items()})
if rf3_tot and 'rosettafold3' in a10:
  ours = dict(a10['rosettafold3'])
  Ls = sorted(set(rf3_tot) & set(ours))
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
  ax1.plot(Ls, [ours[L] for L in Ls], 'o-', label='ours (forward)', ms=5)
  ax1.plot(Ls, [rf3_tot[L][0] for L in Ls], 's--',
           label="native run() (incl. featurise + IO)", ms=5)
  split = [L for L in Ls if rf3_tot[L][2] is not None]
  if split:
    ax1.plot(split, [rf3_tot[L][2] for L in split], '^:', color='crimson',
             label='native minus featurisation', ms=6)
  ax1.set_xlabel('tokens'); ax1.set_ylabel('s / prediction'); ax1.grid(alpha=.3)
  ax1.set_title('rosettafold3: ours vs native, A10'); ax1.legend(fontsize=8)
  if split:
    ax2.plot(split, [rf3_tot[L][2] / ours[L] for L in split], 'o-', color='crimson',
             label='vs native forward')
  ax2.plot(Ls, [rf3_tot[L][0] / ours[L] for L in Ls], 's--', color='grey',
           label='vs native run() (upper bound)')
  ax2.axhline(1.0, color='k', lw=.8, ls=':')
  ax2.set_xlabel('tokens'); ax2.set_ylabel('native / ours'); ax2.grid(alpha=.3)
  ax2.set_title('speedup by size'); ax2.legend(fontsize=8)
  fig.tight_layout(); fig.savefig(f'{HERE}/rf3_vs_native.png', dpi=130)
  print('wrote rf3_vs_native.png')
  for L in Ls:
    t, ft, fw = rf3_tot[L]
    extra = f'  fwd {fw:6.2f}s  ratio_fwd {fw / ours[L]:.2f}x' if fw else ''
    print(f'  L={L:4d}  ours {ours[L]:6.2f}s  native {t:6.2f}s'
          f'  ratio_total {t / ours[L]:.2f}x{extra}')

# Native data per port. A CURVE where a length sweep was run; otherwise the one
# verified point, drawn as a lone marker so a single measurement is never
# mistaken for a trend. chai-1's point is at its 256 BUCKET boundary (it pads
# 64/128/192 all up to 256, so those lengths are not comparable).
def load_sweep(path):
  """native_sweep.tsv -> {model: [(L, steady), ...]} plus the OOM ceiling.

  Rows that never reached the forward ("NO_TIMINGS") are OOM: inference.py
  catches its own out-of-memory and LOGS it rather than raising, so the harness
  completes with nothing timed. Recorded as the ceiling, not dropped -- where an
  implementation stops on a 23 GB card is a result.
  """
  import re as _re
  curves, ceiling = {}, {}
  if not os.path.exists(path):
    return curves, ceiling
  for line in open(path):
    f = line.rstrip('\n').split('\t')
    if len(f) < 3:
      continue
    m, L, rest = f[0], int(f[1]), f[2]
    st = _re.search(r'steady\s+([\d.]+)', rest)
    if st:
      curves.setdefault(m, []).append((L, float(st.group(1))))
    elif 'NO_TIMINGS' in rest or rest == 'OOM':
      ceiling[m] = min(ceiling.get(m, 10 ** 9), L)
  for m in curves:
    curves[m].sort()
  return curves, ceiling

sweep, oom = load_sweep(f'{S}/native_sweep.tsv')
# verified single points at 64 (quiet machine, >=6 calls) prepended to the sweep
BASE_64 = {'boltz2': 7.590, 'rosettafold3': 11.867, 'opendde': 8.425,
           'protenix2': 10.398, 'openfold3': 12.880}
NATIVE_CURVES = {}
for m, pts in sweep.items():
  base = [(64, BASE_64[m])] if m in BASE_64 else []
  NATIVE_CURVES[m] = base + pts
if 'chai1' in NATIVE_CURVES:                 # chai-1's 256 point came earlier
  NATIVE_CURVES['chai1'] = sorted(set([(256, 28.394)] + NATIVE_CURVES['chai1']))
NATIVE_POINTS = {}

# ---------- per-model panels ---------------------------------------------
models = sorted(set(a10) | set(a100))
if models:
  cols = 4; rows = (len(models) + cols - 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
  natd = dict(nat)
  for i, m in enumerate(models):
    ax = axes[i // cols][i % cols]
    for data, lbl, st in ((a10, 'ours, A10', 'o-'), (a100, 'ours, A100*', 's--')):
      if data.get(m):
        ax.plot([p[0] for p in data[m]], [p[1] for p in data[m]], st, label=lbl, ms=4)
    # native for EVERY port, not just boltz2: a full curve where one was
    # measured, otherwise the verified single point. Without this the figure
    # silently implied native data existed only for boltz2.
    ncurve = NATIVE_CURVES.get(m)
    if ncurve:
      ax.plot([q[0] for q in ncurve], [q[1] for q in ncurve], '^:',
              color='crimson', label='native (torch)', ms=5)
    elif m in NATIVE_POINTS:
      nx, ny = NATIVE_POINTS[m]
      ax.plot([nx], [ny], '^', color='crimson', ms=9,
              label=f'native @{nx} (single point)')
    if m in oom:                              # mark where native runs out of memory
      ax.axvline(oom[m], color='crimson', ls=':', lw=1, alpha=.7)
      ax.text(oom[m], ax.get_ylim()[1] * 0.55, f' native OOM\n @{oom[m]}',
              color='crimson', fontsize=7, ha='left', va='top')
    ax.set_title(m, fontsize=11); ax.set_xlabel('tokens'); ax.set_ylabel('s / pred')
    ax.grid(alpha=.3); ax.legend(fontsize=7)
  for j in range(len(models), rows * cols):
    axes[j // cols][j % cols].axis('off')
  fig.suptitle('Steady-state runtime vs length -- 3 recycles, 200 diffusion steps, '
               '1 sample, num_msa=1024\n'
               '*A100 also runs tokamax Triton kernels the A10 cannot, so it is '
               'not a pure hardware comparison', fontsize=11)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  fig.savefig(f'{HERE}/runtime_per_model.png', dpi=130)
  print('wrote runtime_per_model.png')

# ---------- all models, one axis -----------------------------------------
for data, tag in ((a10, 'A10'), (a100, 'A100')):
  if not data:
    continue
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
  for m in sorted(data):
    if not data[m]:
      continue
    x = [p[0] for p in data[m]]; y = [p[1] for p in data[m]]
    ax1.plot(x, y, 'o-', label=m, ms=4); ax2.loglog(x, y, 'o-', label=m, ms=4)
  ax1.set_xlabel('tokens'); ax1.set_ylabel('s / prediction'); ax1.grid(alpha=.3)
  ax1.set_title(f'{tag}: runtime vs length')
  ax2.set_xlabel('tokens'); ax2.set_ylabel('s / prediction')
  ax2.grid(alpha=.3, which='both'); ax2.legend(fontsize=7, ncol=2)
  ax2.set_title(f'{tag}: log-log (slope = scaling exponent)')
  fig.tight_layout(); fig.savefig(f'{HERE}/runtime_all_{tag}.png', dpi=130)
  print(f'wrote runtime_all_{tag}.png')

# ---------- ours vs native, one bar per port (64 tokens) ------------------
# Populated from the native harnesses; None = not measured yet. chai-1 is quoted
# at its 256 BUCKET boundary because 64 tokens pads to 256 there and the
# comparison would be meaningless (see NATIVE_SETUP.md).
# All re-verified on a QUIET machine (>=6 calls, tail spread ~1%). The first
# readings for protenix2 (26.74) and opendde (16.22) were contaminated by host
# load and inflated ~2x -- always in the direction that flattered us.
NATIVE_64 = {
    'boltz2': 7.59, 'rosettafold3': 11.87, 'opendde': 8.43, 'protenix2': 10.40,
    'openfold3': 12.88,
}
CHAI_AT_256 = (13.96, 28.39)      # (ours, native model-only) at the 256 bucket

if a10 and NATIVE_64:
  ours64 = {m: dict(v).get(64) for m, v in a10.items()}
  ports = [p for p in sorted(NATIVE_64) if ours64.get(p)]
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
  x = np.arange(len(ports))
  ax1.bar(x - 0.2, [ours64[p] for p in ports], 0.4, label='ours (jax)')
  ax1.bar(x + 0.2, [NATIVE_64[p] for p in ports], 0.4, label='native (torch)')
  ax1.set_xticks(x); ax1.set_xticklabels(ports, rotation=20, ha='right')
  ax1.set_ylabel('s / prediction'); ax1.grid(alpha=.3, axis='y')
  ax1.set_title('64 tokens, A10: forward vs forward'); ax1.legend()

  rat = [NATIVE_64[p] / ours64[p] for p in ports]
  labels = list(ports) + ['chai1\n(256 bucket)']
  rat_all = rat + [CHAI_AT_256[1] / CHAI_AT_256[0]]
  ax2.bar(np.arange(len(labels)), rat_all,
          color=['tab:blue'] * len(ports) + ['tab:grey'])
  ax2.axhline(1.0, color='k', lw=.8, ls=':')
  ax2.set_xticks(np.arange(len(labels)))
  ax2.set_xticklabels(labels, rotation=20, ha='right')
  ax2.set_ylabel('native / ours'); ax2.grid(alpha=.3, axis='y')
  ax2.set_title('speedup over the reference implementation')
  for i, v in enumerate(rat_all):
    ax2.text(i, v + 0.1, f'{v:.1f}x', ha='center', fontsize=9)
  fig.tight_layout(); fig.savefig(f'{HERE}/ours_vs_native_64.png', dpi=130)
  print('wrote ours_vs_native_64.png')
  for p, r in zip(ports, rat):
    print(f'  {p:14s} ours {ours64[p]:6.2f}  native {NATIVE_64[p]:6.2f}  {r:5.2f}x')
