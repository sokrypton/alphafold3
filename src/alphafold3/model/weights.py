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

import glob
import os
import sys
import tarfile
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
# ITSELF is fetched on demand by alphafold3.model.esm, because it is 5.5 GB and
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
  if glob.glob(os.path.join(model_dir, '*.bin.zst')):
    # A hand-converted or hand-placed blob under any name. NOT gated on the
    # precision: a directory that already holds weights is weights you already
    # have, which is what --weights_precision says it ignores. Gating it on
    # fp32 meant that once int8 became the default, pointing --model_dir at a
    # directory you had converted yourself DOWNLOADED the int8 blob into it and
    # then failed, because the directory then held two.
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
  # `wanted` is the LOCAL basename; the repo groups blobs by family, so the URL
  # carries a folder the local directory does not.
  _download(_HF_URL.format(repo=spec.weights_repo,
                           file=spec.weights_path_for(precision)),
            os.path.join(model_dir, wanted), log=log)
  for extra in _COMPANIONS.get(spec.name, ()):
    dst = os.path.join(model_dir, extra)
    if os.path.exists(dst):
      continue
    try:
      _download(_HF_URL.format(repo=spec.weights_repo,
                               file=spec.companion_path(extra)), dst, log=log)
    except Exception as err:  # pylint: disable=broad-except
      log(f'note: could not fetch {extra} for {spec.name} ({err})')
  return model_dir


# AlphaFold 2's parameters are DeepMind's own release under CC BY 4.0 -- open
# for commercial and non-commercial use -- so they are FETCHED FROM SOURCE
# rather than republished here. One tar holds every model: the five monomer
# sets, their _ptm variants and the multimer ones.
AF2_PARAMS_URL = ('https://storage.googleapis.com/alphafold/'
                  'alphafold_params_2022-12-06.tar')


def _download_parallel(url: str, dst: str, log=print) -> None:
  """`_download`, but many connections at once where that is possible.

  AF2's parameter tar is 5.3 GB and `urlretrieve` fetches it on ONE connection,
  which is the bottleneck rather than the link -- it is minutes of a Colab
  session. ColabFold has used `aria2c -x 16` for this from the start, so prefer
  it when present and fall back to the single-stream path when it is not.

  Only AF2 needs this. Every converted af3-family blob is 130-350 MB, where the
  setup cost of a second process is a larger share than the saving.
  """
  import shutil
  import subprocess

  aria = shutil.which('aria2c')
  if aria:
    log(f'downloading {url}\n         -> {dst}  (aria2c, 16 connections)')
    try:
      subprocess.run(
          [aria, '-x', '16', '-s', '16', '-k', '1M', '--file-allocation=none',
           '--summary-interval=10', '--console-log-level=warn',
           '-d', os.path.dirname(dst) or '.', '-o', os.path.basename(dst), url],
          check=True)
      return
    except (subprocess.CalledProcessError, OSError) as err:
      # A partial file from a failed run would be resumed as if complete, so
      # clear it before falling back.
      log(f'aria2c failed ({err}); falling back to a single connection')
      if os.path.exists(dst):
        os.remove(dst)
  _download(url, dst, log=log)


def ensure_af2_params(model_dir: str, download: bool = True, log=print) -> str:
  """-> a directory holding `params_model_*.npz`, downloading them if needed.

  AF2 reads its parameters by filename, so this only has to guarantee the files
  exist; nothing is converted. `model_dir` may already be a user's own params
  directory, in which case it is left alone.
  """
  model_dir = os.path.expanduser(model_dir)
  if glob.glob(os.path.join(model_dir, 'params_model_*.npz')):
    return model_dir
  # AF2's own layout puts them in a `params/` subdirectory; accept either.
  nested = os.path.join(model_dir, 'params')
  if glob.glob(os.path.join(nested, 'params_model_*.npz')):
    return nested
  if not download:
    raise FileNotFoundError(
        f'no params_model_*.npz in {model_dir}; re-run with downloading '
        f'enabled to fetch them from {AF2_PARAMS_URL}')
  os.makedirs(model_dir, exist_ok=True)
  tar_path = os.path.join(model_dir, 'alphafold_params.tar')
  _download_parallel(AF2_PARAMS_URL, tar_path, log=log)
  log(f'extracting {tar_path}')
  with tarfile.open(tar_path) as tar:
    # filter='data' refuses absolute paths and traversal; it is the default from
    # Python 3.14 and is spelled out here so the behaviour does not depend on it
    tar.extractall(model_dir, filter='data')
  os.remove(tar_path)
  if not glob.glob(os.path.join(model_dir, 'params_model_*.npz')):
    raise FileNotFoundError(
        f'{AF2_PARAMS_URL} did not yield params_model_*.npz in {model_dir}')
  return model_dir
