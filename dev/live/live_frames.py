"""Turn live fold frames into things a notebook can draw.

Pairs with `ModelRunner.live_model()`, which calls back after every trunk pass
and every denoise step. Recycles carry a contact map (the distogram head
already computes `contact_probs`); diffusion steps carry coordinates, which
become one mmCIF model per frame.

Nothing here is on the fold path -- it is display only.
"""

from alphafold3.model import feat_batch, model


def as_batch(batch_dict):
  """The featurised dict -> the Batch the structure helpers want.

  Pass the dict as featurisation produced it, BEFORE
  `utils.remove_invalidly_typed_feats`. That strips the string arrays (chain_id,
  res_name) which the output layout is built from, and a Batch made from the
  cleaned dict has `convert_model_output` set to None -- surfacing later as
  `AttributeError: 'NoneType' object has no attribute 'chain_id'` from inside
  the coordinate conversion.
  """
  return feat_batch.Batch.from_data_dict(batch_dict)


def frame_cif(positions, batch, name='frame'):
  """One diffusion frame -> an mmCIF string.

  `positions` is (num_tokens, max_atoms_per_token, 3) for ONE sample, i.e.
  on_frame's data indexed by sample. Reuses the library's own conversion, so an
  animation frame has the same atom naming and ordering as the final answer
  and a viewer cannot show something the fold did not produce.
  """
  struc = model.predicted_structure_from_coords(positions, batch)
  return struc.to_mmcif()


def contact_map(contact_probs, num_tokens=None):
  """Recycle frame -> a square array in [0, 1], trimmed to the real tokens."""
  import numpy as np
  cm = np.asarray(contact_probs)
  if num_tokens:
    cm = cm[:num_tokens, :num_tokens]
  return cm


def recycle_distance(prev, cur):
  """How much a pass moved the contact map -- localfold's recycleDistance.

  A single number per pass is what tells you the trunk has converged, which the
  animation should show rather than leaving the viewer to eyeball it.
  """
  import numpy as np
  if prev is None:
    return float('nan')
  return float(np.abs(np.asarray(cur) - np.asarray(prev)).mean())
