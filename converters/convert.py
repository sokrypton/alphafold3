"""Convert a published AF3-family checkpoint into AlphaFold3 haiku parameters.

    python -m converters.convert --model boltz2 --out ~/ported/boltz2

Standalone, offline tooling: it imports torch to read the checkpoint and
`alphafold3` only for the parameter blob format. Nothing in the served model
path imports this package. The product is one `<model>.bin.zst`, which is what
`run_alphafold.py --model <name> --model_dir <dir>` loads -- and what we publish
to Hugging Face so that a normal run never touches torch or a 2 GB checkpoint.

With no --checkpoint the original download is fetched into --out first (see
converters/sources.py for where each one comes from and why).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

from converters import CONVERTERS
from converters.sources import SOURCES


def _download(url: str, dst: str) -> None:
  import urllib.request

  print(f'downloading {url}\n         -> {dst}', flush=True)
  tmp = dst + '.part'

  def hook(blocks, bsize, total):
    if total and total > 0:
      pct = min(100.0, 100.0 * blocks * bsize / total)
      sys.stderr.write(f'\r  {pct:5.1f}%  ({blocks * bsize >> 20} / {total >> 20} MB)')
      sys.stderr.flush()

  urllib.request.urlretrieve(url, tmp, reporthook=hook)
  sys.stderr.write('\n')
  os.replace(tmp, dst)


def fetch_checkpoint(model: str, out_dir: str) -> str:
  """Return a local path to `model`'s original checkpoint, downloading if absent.

  For a model published as several archives the return value is the directory
  holding them -- that is what its converter expects.
  """
  src = SOURCES.get(model)
  if src is None:
    raise SystemExit(
        f'{model!r} has no registered download; pass --checkpoint explicitly.')
  os.makedirs(out_dir, exist_ok=True)
  if 'files' in src:
    for url, rel in src['files']:
      dst = os.path.join(out_dir, rel)
      os.makedirs(os.path.dirname(dst), exist_ok=True)
      if not os.path.isfile(dst):
        _download(url, dst)
    return out_dir
  dst = os.path.join(out_dir, src['file'])
  if not os.path.isfile(dst):
    _download(src['url'], dst)
  return dst


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
      description=__doc__.split('\n', 1)[0],
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('--model', required=True, choices=sorted(CONVERTERS),
                 help='which model family the checkpoint is')
  p.add_argument('--checkpoint', type=pathlib.Path, default=None,
                 help='the original checkpoint (or, for chai-1, the directory '
                      'of TorchScript archives). Downloaded into --out if omitted.')
  p.add_argument('--out', type=pathlib.Path, required=True,
                 help='directory to write <model>.bin.zst into')
  p.add_argument('--no_shapes', action='store_true',
                 help='skip the parameter-shape manifest (it needs jax)')
  args = p.parse_args(argv)

  out_dir = str(args.out.expanduser())
  os.makedirs(out_dir, exist_ok=True)
  ckpt = (str(args.checkpoint.expanduser()) if args.checkpoint
          else fetch_checkpoint(args.model, out_dir))

  t0 = time.time()
  print(f'converting {args.model}: {ckpt}', flush=True)
  CONVERTERS[args.model](ckpt, out_dir)
  blobs = sorted(pathlib.Path(out_dir).glob('*.bin.zst'))
  if not blobs:
    raise SystemExit(f'conversion produced no *.bin.zst in {out_dir}')
  for b in blobs:
    print(f'  wrote {b} ({b.stat().st_size >> 20} MB)')

  if not args.no_shapes:
    # The parameter-shape manifest, so loading never has to run jax.eval_shape --
    # and so a gap in the conversion is named rather than discovered.
    from converters import shapes

    print('deriving the parameter-shape manifest...', flush=True)
    # HIDE OUR ARGV FROM ABSL. Deriving the manifest traces the graph, which
    # reaches tokamax's lazily-read `cross_compile` flag, whose getter calls
    # `flags.FLAGS(sys.argv)` -- and absl then aborts on the first flag it does
    # not own: "Unknown command line flag 'model'". Nothing about conversion
    # needs argv past this point, so blank it for the trace.
    import sys as _sys

    _argv, _sys.argv = _sys.argv, _sys.argv[:1]
    from alphafold3.model import params as af3_params

    loaded = af3_params.get_model_haiku_params(model_dir=out_dir)
    try:
      path = shapes.write(args.model, out_dir, params=loaded,
                          checkpoint=ckpt)
    finally:
      _sys.argv = _argv
    import json

    with open(path) as fh:
      missing = json.load(fh)['missing']
    print(f'  wrote {path}'
          + (f' ({len(missing)} parameters not covered by the conversion)'
             if missing else ' (conversion covers every parameter)'))
  print(f'done in {time.time() - t0:.1f}s', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
