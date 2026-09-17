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


# py2Dmol's `add(coords, ...)` wants ONE coordinate per position, not the dense
# (token, atom_slot) grid the model works in. The representative atom differs by
# polymer: CA for protein, C1' for a nucleotide, and for a ligand there is no
# canonical one, so the first atom present is used. Read from the layout's own
# atom names rather than assuming a slot -- CA happens to be slot 1 for every
# standard residue, which is exactly the kind of coincidence that breaks on the
# first ligand.
_REP_ATOMS = ("CA", "C1'")


def rep_atom_index(batch):
  """(token -> slot, valid) for the atom that represents each token."""
  import numpy as np
  layout = batch.convert_model_output.token_atoms_layout
  names = np.asarray(layout.atom_name)
  present = names != ''
  idx = np.zeros(names.shape[0], dtype=int)
  for t in range(names.shape[0]):
    hit = [j for j in range(names.shape[1]) if names[t, j] in _REP_ATOMS]
    if hit:
      idx[t] = hit[0]
    else:                       # ligand or an unexpected residue: first atom
      first = np.flatnonzero(present[t])
      idx[t] = first[0] if first.size else 0
  return idx, present.any(-1)


def frame_positions(positions, batch):
  """One diffusion frame -> (coords Nx3, chains, residue_numbers), masked."""
  import numpy as np
  idx, valid = rep_atom_index(batch)
  layout = batch.convert_model_output.token_atoms_layout
  xyz = np.asarray(positions)[np.arange(len(idx)), idx]
  chains = np.asarray(layout.chain_id)[:, 0]
  resids = np.asarray(layout.res_id)[:, 0]
  # PLAIN python types. py2Dmol serialises these to JSON, and a numpy int64
  # raises "Object of type int64 is not JSON serializable" -- which surfaces as
  # every frame being silently dropped by the viewer while the fold itself runs
  # perfectly (0 of 8 frames arrived, measured on Colab).
  return (np.asarray(xyz[valid], dtype=np.float32),
          [str(c) for c in chains[valid]],
          [int(r) for r in resids[valid]])


# ---------------------------------------------------------------- steering
#
# `live_model()`'s run(..., on_step=f) calls f(step, positions, sigma) after
# every denoise step and denoises whatever it returns. The examples below are
# deliberately simple: the point is that a restraint is a few lines of numpy
# against the coordinates, with no change to the graph and no recompile.
#
# Guidance is scaled by SIGMA. The sampler is still at hundreds of angstroms of
# noise early on, where a 1 A nudge is nothing, and near zero at the end, where
# the same nudge is a distortion. Scaling by sigma/sigma_max applies the push
# while the structure is still forming and lets it go as it sets.


def pull_together(token_a, token_b, target=8.0, strength=0.5, batch=None):
  """Nudge two tokens' representative atoms towards `target` angstroms apart.

  A distance restraint, which is the simplest thing anyone actually wants from
  steering. Returns a callback for `on_step`.
  """
  import numpy as np
  idx, _ = rep_atom_index(batch)
  ia, ib = int(idx[token_a]), int(idx[token_b])

  def steer(step, positions, sigma):
    pos = np.array(positions)                 # (samples, tokens, atoms, 3)
    a = pos[:, token_a, ia]
    b = pos[:, token_b, ib]
    d = np.linalg.norm(b - a, axis=-1, keepdims=True)
    unit = (b - a) / np.maximum(d, 1e-6)
    # how far each end has to move, halved because both move
    shift = ((d - target) * 0.5 * strength
             * min(1.0, sigma / 16.0))[..., None, :]
    # WHOLE TOKENS, not the representative atom alone. Moving one atom of one
    # residue is undone by the next denoise step -- measured: pulling two CAs
    # from 39.2 A towards 6 A at full strength ended at 38.7 A, i.e. nothing.
    # Translating every atom of the residue is a restraint the score has to
    # argue with rather than simply repair.
    pos[:, token_a] = pos[:, token_a] + unit[:, None, :] * shift[:, 0]
    pos[:, token_b] = pos[:, token_b] - unit[:, None, :] * shift[:, 0]
    return pos

  return steer


def symmetrise(n_copies, batch=None):
  """Average the coordinates of `n_copies` equal-length chains.

  C_n symmetry the crude way -- average in the frame of the first copy. Enough
  to show that a whole-structure operation is the same kind of callback.
  """
  import numpy as np

  def steer(step, positions, sigma):
    pos = np.array(positions)
    n_tok = pos.shape[1]
    per = n_tok // n_copies
    if per * n_copies != n_tok:
      return None                             # not equal chains; leave it alone
    block = pos[:, :per * n_copies].reshape(pos.shape[0], n_copies, per,
                                            *pos.shape[2:])
    mean = block.mean(axis=1, keepdims=True)
    w = min(1.0, sigma / 16.0)
    block = block * (1 - w) + mean * w
    pos[:, :per * n_copies] = block.reshape(pos.shape[0], per * n_copies,
                                            *pos.shape[2:])
    return pos

  return steer
