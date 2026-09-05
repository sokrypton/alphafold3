"""Upload the converted models to the Hugging Face repo a run fetches from.

`alphafold3.model.weights` pulls `<model>.bin.zst` and `<model>.shapes.json` by
exact filename over plain HTTPS, so the repo is flat and this just puts the
files there under those names.

    python -m converters.publish --dir ~/ported --repo you/af3-ported-weights

Needs `huggingface_hub` and a token (`huggingface-cli login`, or HF_TOKEN).
AlphaFold 3's own parameters are refused: DeepMind requires you to request them
and they are not ours to redistribute.
"""

from __future__ import annotations

import argparse
import os
import pathlib

def _registry():
  """The model registry, or None where alphafold3 is not importable.

  Uploading needs huggingface_hub, which does not have to live in the same
  environment as the model code -- so this falls back to reading the directory
  rather than making the two dependencies meet.
  """
  try:
    from alphafold3.model import model_registry
  except ImportError:
    return None
  return model_registry


_COMPANIONS = {m: ('%s.lm.npz' % m,) for m in (
    'esmfold2', 'esmfold2_fast', 'esmfold2_exp', 'esmfold2_exp_fast',
    'esmfold2_exp_cutoff2025', 'esmfold2_exp_fast_cutoff2025')}


def files_for(model: str, root: pathlib.Path, precisions=('fp32',)):
  """-> [(local path, name in the repo)] for one model, or [] if not converted.

  `precisions` names the storage forms to publish. float32 keeps the original
  filename, so republishing never moves the bytes a previously handed-out link
  resolves to; fp16 and int8 are ADDITIONAL files beside it
  (`<stem>.int8.bin.zst`), which is why they are opt-in per call rather than a
  replacement. A form that has not been built locally is reported and skipped,
  not silently dropped.
  """
  registry = _registry()
  directory = root / model
  # An artifact with no ModelSpec is still publishable: ESM-C is the separate
  # graph ESMFold2 conditions on, not an AF3 model, and the code below already
  # has a spec-less path for exactly this.
  try:
    spec = registry.get(model) if registry else None
  except ValueError:
    spec = None
  weights_file = spec.weights_file if spec else f'{model}.bin.zst'
  wanted = []
  for precision in precisions:
    if precision == 'fp32':
      wanted.append(weights_file)
    elif spec is not None:
      wanted.append(spec.weights_file_for(precision))
    else:
      wanted.append(f"{weights_file.split('.', 1)[0]}.{precision}.bin.zst")
  wanted.append(f'{model}.shapes.json')
  # Companion artifacts a model cannot be run without. ESMFold2's LM shim turns
  # ESM-C's 81 hidden states into the pair representation its trunk reads; it is
  # 3.7 MB, it is converted from the same checkpoint as the blob, and without it
  # the esmc weights are unusable -- which would leave the published model
  # single-sequence-or-MSA only. Unlike chai's std_conformers.npz (below) there
  # is no licence question: it is ours from a checkpoint we already redistribute.
  wanted.extend(_COMPANIONS.get(model, ()))
  # chai-1's std_conformers.npz is deliberately NOT published. It is derived from
  # chai's own RDKit cache rather than from anything we converted, and chai's
  # weights licence is not established, so redistributing it is a claim we have
  # not earned. Without it chai falls back to the CCD ideal conformers every
  # other model uses -- measured at 1.776 A vs 1.698 A on 6MRR, i.e. 0.08 A, and
  # ligands are unaffected either way because chai center_random_augments those.
  # featurise_chai1(ref_conformers=...) still takes them if a user has the file.
  found = []
  for name in wanted:
    path = directory / name
    if path.exists():
      found.append((path, name))
    else:
      print(f'  missing {path}')
  return found


def _converter_is_current(manifest_path):
  """-> True / False / None (no provenance recorded) for a shape manifest."""
  import json

  try:
    from converters import shapes
  except ImportError:
    return None
  with open(manifest_path) as fh:
    return shapes.converter_is_current(json.load(fh))


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
  p.add_argument('--dir', required=True, type=pathlib.Path,
                 help='root holding <model>/<model>.bin.zst')
  p.add_argument('--repo', default=None,
                 help='Hugging Face repo id (default: the one the runtime '
                      'fetches from)')
  p.add_argument('--model', action='append', default=[],
                 help='only this model, repeatable')
  p.add_argument('--precision', action='append', default=[],
                 choices=['fp32', 'fp16', 'int8'],
                 help='storage form to publish, repeatable (default fp32). '
                      'fp16 and int8 go up BESIDE the float32 blob under '
                      '<stem>.<precision>.bin.zst, so existing links keep '
                      'resolving to the same bytes.')
  p.add_argument('--dry_run', action='store_true')
  p.add_argument('--allow_stale', action='store_true',
                 help='publish even where the shape manifest says the blob was '
                      'built by a different version of the converter')
  args = p.parse_args(argv)

  registry = _registry()
  repo = args.repo or (registry.MODEL_SPECS['openfold3'].weights_repo
                       if registry else None)
  if repo is None:
    raise SystemExit('pass --repo (alphafold3 is not importable here, so the '
                     'default cannot be read from the registry)')
  root = args.dir.expanduser()
  known = (sorted(registry.MODEL_SPECS) if registry
           else sorted(d.name for d in root.iterdir() if d.is_dir()))
  models = args.model or [m for m in known if m != 'alphafold3']
  if 'alphafold3' in models:
    raise SystemExit('AlphaFold 3 parameters are not ours to redistribute.')

  precisions = tuple(args.precision) or ('fp32',)
  uploads = []
  stale = []
  for model in models:
    print(f'{model}:')
    for path, name in files_for(model, root, precisions):
      print(f'  {path} -> {repo}/{name} ({path.stat().st_size >> 20} MB)')
      uploads.append((path, name))
      if name.endswith('.shapes.json'):
        current = _converter_is_current(path)
        if current is False:
          stale.append(model)
          print(f'  !! converted by a DIFFERENT version of converters/{model}.py')
        elif current is None:
          print('  ?? no converter provenance recorded; reconvert to get it')
  if stale and not args.allow_stale:
    # Publishing a blob built by an older converter is how the OpenDDE weights
    # went out carrying a residue-alphabet bug that had already been fixed --
    # gaps embedded as RNA adenine, every RNA base read one slot high, and
    # nothing about the file saying so.
    raise SystemExit(
        f'refusing to publish {", ".join(sorted(set(stale)))}: built by a '
        'different version of the converter than the one in this tree. '
        'Reconvert, or pass --allow_stale if you are certain.')
  if args.dry_run:
    return 0
  if not uploads:
    raise SystemExit(f'nothing to upload under {root}')

  from huggingface_hub import HfApi

  api = HfApi(token=os.environ.get('HF_TOKEN'))
  api.create_repo(repo, repo_type='model', exist_ok=True)
  for path, name in uploads:
    print(f'uploading {name}...', flush=True)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                    repo_id=repo, repo_type='model')
  print(f'done: https://huggingface.co/{repo}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
