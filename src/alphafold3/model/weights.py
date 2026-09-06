# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Fetching converted weights, so a first run needs no manual download.

A model's blob is converted offline (see `converters/`) and published; this
pulls that published blob, and its shape manifest, straight to disk. Plain HTTPS
rather than huggingface_hub: it is two files by exact name, and a run should not
need another dependency to get them.

AlphaFold 3's own parameters are deliberately not fetchable. DeepMind requires
you to request them, and it is not ours to redistribute -- point --model_dir at
your own copy.
"""

from __future__ import annotations

import os
import sys
import urllib.request

from alphafold3.model import model_config
from alphafold3.model import model_registry


_HF_URL = 'https://huggingface.co/{repo}/resolve/main/{file}'


def default_dir(model_name: str, precision: str = 'fp32') -> str:
  """Where a fetched model lands when no --model_dir says otherwise.

  Each precision gets its own directory. params.select_model_files picks the
  blob out of a directory by filename, and `x.bin.zst` beside `x.int8.bin.zst`
  is two candidates for one slot -- so keeping them apart is what makes
  --weights_precision mean anything. float32 keeps the original path, so an
  existing cache is still found.
  """
  root = os.environ.get('AF3_WEIGHTS_DIR') or os.path.join(
      os.path.expanduser('~'), '.cache', 'alphafold3', 'weights')
  suffix = '' if precision == 'fp32' else f'-{precision}'
  return os.path.join(root, model_name + suffix)


def _download(url: str, dst: str, log=print) -> None:
  log(f'downloading {url}\n         -> {dst}')
  tmp = dst + '.part'

  def hook(blocks, bsize, total):
    if total and total > 0:
      pct = min(100.0, 100.0 * blocks * bsize / total)
      sys.stderr.write(f'\r  {pct:5.1f}%  ({blocks * bsize >> 20} / '
                       f'{total >> 20} MB)')
      sys.stderr.flush()

  urllib.request.urlretrieve(url, tmp, reporthook=hook)
  sys.stderr.write('\n')
  os.replace(tmp, dst)


# Companion artifacts a model needs beside its blob. ESMFold2's LM shim turns
# ESM-C's hidden states into the pair representation its trunk reads; it is
# 3.7 MB, and without it the ESM-C path raises on the first fold. The tower
# ITSELF is fetched on demand by converters.esm_lm, because it is 5.5 GB and
# ESMFold2 also runs from an MSA.
_COMPANIONS = {m: ('%s.lm.npz' % m,) for m in model_config.ESMFOLD2_FAMILY}


def ensure_weights(model_name: str, model_dir=None, *, download=True,
                   precision='fp32', log=print) -> str:
  """Make `model_name`'s converted weights exist on disk; return their directory.

  Idempotent: a directory that already holds the wanted blob is left alone,
  which is every run after the first.

  precision picks which published form to fetch. float32 is the default and the
  only one that existed before; 'fp16' and 'int8' are smaller downloads of the
  same weights (converters/quantise.py), which the loader expands on read. The
  existence check is precision-aware on purpose: a directory holding the
  float32 blob must not satisfy a request for int8, or --weights_precision
  would silently do nothing.
  """
  spec = model_registry.get(model_name)
  model_dir = os.path.expanduser(
      str(model_dir or default_dir(spec.name, precision)))
  import glob

  wanted = spec.weights_file_for(precision)
  if os.path.exists(os.path.join(model_dir, wanted)):
    return model_dir
  if precision == 'fp32' and glob.glob(os.path.join(model_dir, '*.bin.zst')):
    # a hand-converted or hand-placed blob under any name
    return model_dir

  if spec.weights_repo is None:
    raise FileNotFoundError(
        f'no weights in {model_dir}, and {spec.name} is not published here. '
        + ('AlphaFold 3 parameters must be requested from Google DeepMind; '
           'point --model_dir at your own copy.' if spec.name == 'alphafold3'
           else 'Convert them yourself: python -m converters.convert '
                f'--model {spec.name} --out {model_dir}'))
  if not download:
    raise FileNotFoundError(
        f'no weights in {model_dir}; re-run with downloading enabled to fetch '
        f'{wanted} from {spec.weights_repo}')

  os.makedirs(model_dir, exist_ok=True)
  _download(_HF_URL.format(repo=spec.weights_repo, file=wanted),
            os.path.join(model_dir, wanted), log=log)
  # The shape manifest rides along; it is small, and without it a gap in the
  # conversion is only discovered as an opaque failure mid-forward.
  manifest = f'{spec.name}.shapes.json'
  try:
    _download(_HF_URL.format(repo=spec.weights_repo, file=manifest),
              os.path.join(model_dir, manifest), log=log)
  except Exception as err:  # pylint: disable=broad-except
    log(f'note: no shape manifest published for {spec.name} ({err})')
  for extra in _COMPANIONS.get(spec.name, ()):
    dst = os.path.join(model_dir, extra)
    if os.path.exists(dst):
      continue
    try:
      _download(_HF_URL.format(repo=spec.weights_repo, file=extra), dst, log=log)
    except Exception as err:  # pylint: disable=broad-except
      log(f'note: could not fetch {extra} for {spec.name} ({err})')
  return model_dir
