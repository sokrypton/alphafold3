"""Read the sweep TSVs and print the native-vs-jax table.

  python tools/benchmarks/bench_table.py ours.tsv [native.tsv]

Reads the shape bench_sweep.py emits (model, length, num_msa, tokens, compile,
steady, all) and the free-text lines run_native_sweep.sh collects. OOM and
FAILED rows are carried through as themselves: an OOM is WHERE AN
IMPLEMENTATION STOPS on this card, which is a result, and a plot that renders it
the same as a crash hides one of them.
"""
import re, sys, collections

def read_ours(path):
    out = {}
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        if len(f) >= 6 and f[0] != 'model':
            try: out[(f[0], int(f[1]), int(f[2]))] = float(f[5])
            except ValueError: pass
        elif len(f) == 4 and f[3] in ('OOM', 'FAILED'):
            out[(f[0], int(f[1]), int(f[2]))] = f[3]
    return out

def read_native(path):
    """`<length>\tNATIVE-X first Ns | steady Ns | all [...]` and similar."""
    out = {}
    for line in open(path):
        m = re.match(r'^(\d+)\s+NATIVE-(\S+).*?steady\s+([0-9.]+)', line)
        if m: out[(m.group(2).lower(), int(m.group(1)))] = float(m.group(3))
    return out

ours = read_ours(sys.argv[1])
native = read_native(sys.argv[2]) if len(sys.argv) > 2 else {}
models = sorted({k[0] for k in ours})
lengths = sorted({k[1] for k in ours})
msas = sorted({k[2] for k in ours})

for n in msas:
    print('\n=== num_msa %d   (steady-state seconds)' % n)
    print('%-17s %s' % ('model', ''.join('%9d' % L for L in lengths)))
    for m in models:
        cells = []
        for L in lengths:
            v = ours.get((m, L, n))
            cells.append('%9s' % ('-' if v is None else
                                  v if isinstance(v, str) else '%.2f' % v))
        print('%-17s %s' % (m, ''.join(cells)))

if native:
    print('\n=== native torch, steady-state seconds (length only)')
    for k in sorted(native): print('  %-14s %-6d %.2f' % (k[0], k[1], native[k]))
    print('\n=== speedup, ours vs native at num_msa %d' % max(msas))
    for m in models:
        for L in lengths:
            nv = native.get((m.replace('rosettafold3', 'rf3'), L))
            ov = ours.get((m, L, max(msas)))
            if nv and isinstance(ov, float):
                print('  %-14s %-6d %.2fx' % (m, L, nv / ov))
