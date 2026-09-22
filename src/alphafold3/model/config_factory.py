# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Building a `Model.Config`, with the overrides a caller actually sets.

THIS LIVED IN `run_alphafold.py`, WHICH IS A SCRIPT. Design code -- anything
that wants a configured model in-process rather than a CLI run -- cannot import
a script without dragging in absl flags and a `main()`, and `staged.py` says so
in as many words. So the factory lives in the library and the script imports it
back, which also means there is one copy of the two ordering traps below rather
than one per caller.
"""

from __future__ import annotations

import os

import jax

from alphafold3.model import model
from alphafold3.model import model_registry


def bfloat16_default() -> str:
  """'all' where bf16 matmuls are native, 'intermediate' otherwise.

  The sampler is the part that cares. A TPU has no compute_capability to read
  and bfloat16 is its native matmul type, so the except branch's answer would
  leave it in f32 on the one accelerator built around bf16.
  AF3_SAMPLER_BF16=1 / =0 forces 'all' / 'intermediate'.
  """
  env = os.environ.get('AF3_SAMPLER_BF16')
  if env is not None:
    return 'all' if env not in ('', '0', 'false', 'False') else 'intermediate'
  try:
    device = jax.devices()[0]
    if device.platform == 'tpu':
      return 'all'
    cc = str(getattr(device, 'compute_capability', '') or '')
    major, _, minor = cc.partition('.')
    return 'all' if (int(major), int(minor or 0)) >= (8, 0) else 'intermediate'
  except Exception:  # pylint: disable=broad-except
    return 'intermediate'


def make_model_config(
    *,
    flash_attention_implementation: str = 'triton',
    glu_kernel: str = 'auto',
    num_diffusion_samples: int = 5,
    num_sampling_steps: int | None = None,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
    model_name: str = 'alphafold3',
    num_msa: int | None = None,
) -> model.Model.Config:
  """Returns a model config with some defaults overridden.

  `model_name` selects the family: it lands in global_config.model, which every
  ported-family forward branch keys on, and brings that family's config shapes
  and sampler constants with it (model_registry.ModelSpec.configure).
  """
  config = model.Model.Config()
  config.global_config.flash_attention_implementation = (
      flash_attention_implementation
  )
  config.global_config.glu_kernel = glu_kernel
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  config.return_distogram = return_distogram
  config.global_config.bfloat16 = bfloat16_default()
  model_registry.get(model_name).configure(config)
  # AFTER configure(), NOT BEFORE. The model spec sets its own sampler
  # constants -- ESMFold2's 15 steps, AF3's 200 -- so an assignment made before
  # it is silently overwritten, and a sweep at 5/15/60 steps then returns three
  # IDENTICAL structures, which is how this was caught.
  if num_sampling_steps is not None:
    config.heads.diffusion.eval.steps = num_sampling_steps
  # HOW MANY MSA ROWS THE TRUNK SEES. Featurisation always hands over a fixed
  # 16384-row buffer (pipeline.msa_crop_size) and the trunk subsamples to this,
  # keeping the query at row 0. Lowering it is the one MSA knob a user can turn.
  #
  # Measured, and worth saying before anyone reaches for it as a speed dial:
  # sweeping 1 / 256 / 1024 moved steady-state runtime by 0.4% at 512 tokens.
  # It changes MEMORY and it changes the compiled executable -- a different
  # value is a different shape, so it misses a compile cache built at another.
  #
  # ANY value is allowed, not just the notebook's ladder. Two need a guard:
  # below 1 there is no query row, and above the featurisation buffer jax does
  # NOT raise -- a gather CLAMPS out-of-range indices, so `--num_msa=20000`
  # against a 16384-row buffer would hand the trunk 3616 duplicates of the
  # last padded row and fold on quietly (checked: a[arange(3,8)] on a 5-row
  # array repeats row 4 four times).
  if num_msa is not None:
    num_msa = int(num_msa)
    from alphafold3.model.pipeline import pipeline as _model_pipeline  # pylint: disable=g-import-not-at-top

    buffer = _model_pipeline.WholePdbPipeline.Config().msa_crop_size
    if num_msa < 1:
      raise ValueError(
          f'--num_msa must be at least 1 -- row 0 is the query -- got {num_msa}.'
      )
    if num_msa > buffer:
      print(
          f'--num_msa={num_msa} is above the {buffer}-row featurisation '
          f'buffer, so {buffer} is what the trunk can actually read; using '
          'that. Rows past the buffer would be copies of its last padded row.'
      )
      num_msa = buffer
    config.evoformer.num_msa = num_msa

  return config
