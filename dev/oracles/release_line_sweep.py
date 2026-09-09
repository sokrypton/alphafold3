"""Do a vendor's release lines diverge by MODULE, and does our port know?

  python3 dev/oracles/release_line_sweep.py /home/ubuntu/protenix

Written after the bug that cost the most this session. ESMFold2 ships two
`MSAEncoder` classes, one per release line, with the block in OPPOSITE order --
and `esmfold2_msa_dump.py` imported the released one for all three variants, so
L1b compared us against a module that release does not run, read 1.000000, and
certified the bug. 1STP with a real MSA was 14.364 A; it is 0.477 A now.

Two passes, because ONE IS NOT ENOUGH -- I closed this after pass 1 and the
fourth divergence walked straight through:

  1. **duplicated class names.** A vendor that ships release lines as separate
     files (`x.py` / `xv2.py`, `modeling_*.py` / `modeling_*_experimental.py`)
     defines the same class twice. Intersect the names and diff the
     intersection's submodules.
  2. **constructor arguments of SHARED classes.** ESMFold2's
     `OuterProductMean` lives in the file BOTH lines import, so it is not
     duplicated at all -- the lines differ by the argument each block passes
     (`divide_outer_before_proj`, hardcoded True on one side, defaulted False on
     the other, worth a constant `output_b * (1 - 1/n)` on every output). Pass 1
     cannot see that.

Neither pass proves a port is right. What they give is a CLOSED LIST of places
it could be wrong by release line, which is what turns an open-ended worry into
a morning's work. Reported per class; a `DIFFERS` line is a question to answer
against the checkpoint, not a bug on its own -- boltz's `post_layer_norm`
differs and is `nn.Identity` in practice.
"""
import argparse
import ast
import collections
import os
import re
import sys


def class_defs(path):
  """{class name: [submodule attribute names]} for one file."""
  out, cur = {}, None
  for line in open(path, errors='ignore'):
    m = re.match(r'class (\w+)\(', line)
    if m:
      cur = m.group(1)
      out[cur] = []
    elif cur is not None:
      a = re.search(r'self\.(\w+)\s*=\s*(?:nn\.)?([A-Z_]\w*)', line)
      if a:
        out[cur].append(a.group(1))
  return out


def ctor_kwargs(path):
  """{constructed class: {(kwarg, source)}} for one file."""
  out = collections.defaultdict(set)
  try:
    tree = ast.parse(open(path, errors='ignore').read())
  except SyntaxError:
    return out
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      for kw in node.keywords:
        if kw.arg:
          out[node.func.id].add((kw.arg, ast.unparse(kw.value)))
  return out


# Argument names that carry a DIMENSION rather than a behaviour. A width
# mismatch fails loudly in load_state_dict or in our converter's shape check;
# a behaviour flag does not, which is the whole point of this screen.
_DIMS = ('dim', 'num', 'n_', '_n', 'depth', 'size', 'heads', 'channel', 'token',
         'atom', 'max', 'in_', 'out_', 'c_', 'd_', 'hidden', 'width', 'bins')


def _is_flag(name):
  return not any(t in name for t in _DIMS)


def find_pairs(root):
  """[(a, b)] files that look like two lines of the same module."""
  py = []
  for dirpath, _, files in os.walk(root):
    if any(p in dirpath for p in ('/.git', '__pycache__', '/test')):
      continue
    py += [os.path.join(dirpath, f) for f in files if f.endswith('.py')]
  pairs = []
  for f in py:
    base = os.path.basename(f)
    stem = base[:-3]
    for suffix in ('v2', '_v2', '2'):
      if stem.endswith(suffix):
        other = os.path.join(os.path.dirname(f), stem[:-len(suffix)] + '.py')
        if other in py:
          pairs.append((other, f))
        break
    for marker in ('_experimental', '_exp'):
      if marker in stem:
        other = os.path.join(os.path.dirname(f),
                             stem.replace(marker, '') + '.py')
        if other in py:
          pairs.append((other, f))
        break
  return sorted(set(pairs)), py


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('root', help='the vendor source tree')
  ap.add_argument('--shared', action='append', default=[],
                  help='a file whose classes BOTH lines import, for pass 2; '
                       'repeatable. Without it pass 2 uses every class the '
                       'tree defines outside the pair.')
  args = ap.parse_args(argv)
  root = os.path.expanduser(args.root)
  pairs, allpy = find_pairs(root)
  print('%s: %d python files, %d candidate release-line pairs'
        % (root, len(allpy), len(pairs)))
  if not pairs:
    print('  no x.py / xv2.py or _experimental pairs -- this vendor ships one '
          'line, so there is nothing for either pass to find.')
    return 0

  shared = {}
  for f in (args.shared or [f for f in allpy
                            if not any(f in p for p in pairs)]):
    shared.update({k: f for k in class_defs(os.path.expanduser(f))})

  for a, b in pairs:
    ca, cb = class_defs(a), class_defs(b)
    both = sorted(set(ca) & set(cb))
    print('\n=== %s  vs  %s' % (os.path.relpath(a, root),
                               os.path.relpath(b, root)))
    print('  PASS 1: %d duplicated class names' % len(both))
    for k in both:
      sa, sb = set(ca[k]), set(cb[k])
      if sa == sb:
        print('    %-30s identical submodules (%d)' % (k, len(sa)))
      else:
        print('    %-30s DIFFERS  first-only %s  second-only %s'
              % (k, sorted(sa - sb) or '-', sorted(sb - sa) or '-'))

    ka, kb = ctor_kwargs(a), ctor_kwargs(b)
    hits = []
    for cls in sorted(set(ka) & set(kb)):
      fa = {x for x in ka[cls] - kb[cls] if _is_flag(x[0])}
      fb = {x for x in kb[cls] - ka[cls] if _is_flag(x[0])}
      if fa or fb:
        # A class that pass 1 already listed has two implementations, so a
        # differing flag at the CALL site is less surprising; one defined in a
        # file both lines import has only ONE implementation, and a differing
        # flag is the only thing separating the two lines. That second kind is
        # what the ESMFold2 outer-product bug was, so say which is which.
        hits.append((cls, fa, fb, cls in both, shared.get(cls)))
    print('  PASS 2: %d classes constructed with different FLAGS' % len(hits))
    for cls, fa, fb, dup, where in hits:
      kind = ('also duplicated (see pass 1)' if dup
              else 'SHARED implementation' if where else 'defined elsewhere')
      print('    %-30s %s' % (cls, kind))
      print('      %-28s first-only %s  second-only %s'
            % ('', sorted(fa) or '-', sorted(fb) or '-'))

    # PASS 3: module-level FUNCTIONS defined in both files. boltz ships its
    # featuriser and its tokeniser as functions, not classes -- so passes 1 and
    # 2 report nothing for `featurizer.py` vs `featurizerv2.py`, and
    # featurisation is where a good share of this project's bugs have lived.
    fna, fnb = functions(a), functions(b)
    shared_fn = sorted(set(fna) & set(fnb))
    diff = [(k, fna[k], fnb[k]) for k in shared_fn if fna[k] != fnb[k]]
    print('  PASS 3: %d functions defined in both, %d with a different '
          'SIGNATURE' % (len(shared_fn), len(diff)))
    for k, sa, sb in diff:
      onlya, onlyb = sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))
      print('    %-30s first-only %s  second-only %s'
            % (k, onlya or '-', onlyb or '-'))
    if shared_fn and not diff:
      print('    (same signatures -- the BODIES can still differ, which no '
            'signature diff can see; those need reading)')
  return 0


def functions(path):
  """{module-level function name: [argument names]} for one file."""
  out = {}
  try:
    tree = ast.parse(open(path, errors='ignore').read())
  except SyntaxError:
    return out
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      a = node.args
      out[node.name] = [x.arg for x in
                        list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)]
  return out


if __name__ == '__main__':
  sys.exit(main())
