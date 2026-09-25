"""A structural-token model's frames are not in the output layout."""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
import live_frames as LF  # noqa: E402

N_RES, N_STRUCT, MAX_ATOMS = 4, 9, 24


def _batch(gather=None):
  """The two things _to_residue_layout reads, and nothing else."""
  names = np.full((N_RES, MAX_ATOMS), '', dtype=object)
  names[:, 0] = 'CA'
  layout = types.SimpleNamespace(atom_name=names)
  b = types.SimpleNamespace(
      convert_model_output=types.SimpleNamespace(token_atoms_layout=layout))
  if gather is not None:
    b._structbook_gather = gather
  return b


def test_a_structural_token_frame_is_mapped_to_the_residue_layout():
  """opendde diffuses in `struct/` and drew 128 rows against a 59-row mask.

  The symptom was the live view dying with
  `Incompatible shapes: mask_shape=(59, 24, 1), value_shape=(128, 24, 3)` and
  the fold then finishing normally with no frames -- a structure at the end and
  nothing during.
  """
  # one structural token per residue here, plus padding rows the gather ignores
  gather = np.full((N_RES, MAX_ATOMS), -1, dtype=np.int32)
  for r in range(N_RES):
    gather[r, 0] = r * MAX_ATOMS          # each residue's first atom
  frame = np.arange(N_STRUCT * MAX_ATOMS * 3, dtype=np.float32).reshape(
      N_STRUCT, MAX_ATOMS, 3)
  out = LF._to_residue_layout(frame, _batch(gather))
  assert out.shape[0] == N_RES, out.shape


def test_a_model_without_structural_tokens_is_untouched():
  frame = np.zeros((N_RES, MAX_ATOMS, 3), np.float32)
  out = LF._to_residue_layout(frame, _batch(None))
  assert out is frame


def test_a_frame_already_in_the_residue_layout_is_untouched():
  """The gather being present does not mean this frame needs it."""
  gather = np.zeros((N_RES, MAX_ATOMS), dtype=np.int32)
  frame = np.zeros((N_RES, MAX_ATOMS, 3), np.float32)
  out = LF._to_residue_layout(frame, _batch(gather))
  assert out.shape == frame.shape
  assert np.array_equal(out, frame)
