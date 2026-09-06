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

  print(f'done in {time.time() - t0:.1f}s', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
