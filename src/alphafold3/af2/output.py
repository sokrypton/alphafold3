'''AlphaFold 2 outputs -> the shared AlphaFold 3 output path.

AF2 predicts coordinates in `atom37`: a fixed 37-slot-per-residue layout keyed by
`residue_constants.atom_types`. Everything downstream in this package -- the
mmCIF writer, the b-factor column, the missing-atom warning, the confidence
files -- is driven instead by the batch's `token_atoms_layout`, which names its
atoms per token.

So the whole of the integration is one gather: for each (token, slot) in the
model-output layout, find that atom name's slot in atom37 and take it. After
that AF2 hands its coordinates to `model.predicted_structure_from_coords` and
gets the same Structure, written by the same code, as any of the AF3-family
models. There is deliberately no second mmCIF path here to drift out of step.
'''

from __future__ import annotations

import numpy as np

from alphafold3.af2.common import residue_constants as rc


def atom37_to_token_atoms(atom37, batch, per_residue=None):
  '''-> (coords, values) in the batch's token_atoms_layout

  Args:
    atom37: (num_res, 37, 3) AF2 coordinates.
    batch: the featurised batch, for `convert_model_output.token_atoms_layout`.
    per_residue: optional (num_res,) per-residue quantity -- AF2's pLDDT -- to
      broadcast onto the atoms of each token, for the b-factor column.

  Returns:
    (coords, values): (num_tokens, max_atoms, 3) and (num_tokens, max_atoms),
    with padding slots and any atom AF2 does not predict left at zero.
  '''
  layout = batch.convert_model_output.token_atoms_layout
  names = np.asarray(layout.atom_name, dtype=object)
  num_tokens, max_atoms = names.shape
  full_tokens = num_tokens

  # AF2 is protein-only and one token is one residue, so the token axis and the
  # residue axis are the same axis. Anything that breaks that -- an atomised
  # modified residue, a ligand -- would silently shift every coordinate by
  # however many extra tokens preceded it, so check rather than assume.
  #
  # The one legitimate mismatch is bucket padding: AF2 is handed the real tokens
  # only (features.num_real_tokens), so the layout can be WIDER than what it
  # predicted, and the padding tokens stay at zero exactly as they would for a
  # model that folded them under a mask.
  num_res = atom37.shape[0]
  if num_res > num_tokens:
    raise ValueError(
        f'AlphaFold 2 predicted {num_res} residues but the featurised input has '
        f'only {num_tokens} tokens')
  if num_res < num_tokens:
    real = np.asarray(batch.token_features.mask).astype(bool)
    if real[:num_res].sum() != num_res or real[num_res:].any():
      raise ValueError(
          f'the featurised input has {num_tokens} tokens and AlphaFold 2 '
          f'predicted {num_res} residues, but the extra tokens are not padding. '
          'AF2 has one token per residue, so this input tokenises to something '
          'it cannot express (an atomised modified residue, or a non-protein '
          'chain).')
    names = names[:num_res]
  num_tokens = names.shape[0]

  # Name -> atom37 slot, once per distinct name rather than per element.
  slot = np.full(names.shape, -1, dtype=np.int32)
  for name, j in rc.atom_order.items():
    slot[names == name] = j
  present = slot >= 0

  token_idx = np.broadcast_to(
      np.arange(num_tokens)[:, None], (num_tokens, max_atoms))

  coords = np.zeros((num_tokens, max_atoms, 3), dtype=np.float32)
  coords[present] = np.asarray(atom37)[token_idx[present], slot[present]]

  values = np.zeros((num_tokens, max_atoms), dtype=np.float32)
  if per_residue is not None:
    values[present] = np.asarray(per_residue)[token_idx[present]]

  if num_tokens != full_tokens:
    pad = ((0, full_tokens - num_tokens), (0, 0))
    coords = np.pad(coords, pad + ((0, 0),))
    values = np.pad(values, pad)
  return coords, values


def predicted_structure(af2_outputs, batch):
  '''AF2's raw output dict + the featurised batch -> a Structure.

  The b-factor column carries AF2's pLDDT, which is what AlphaFold itself writes
  there. AF2 predicts it per residue; the AF3 writer wants it per atom, so it is
  broadcast over each token's atoms.
  '''
  from alphafold3.model import model as af3_model

  atom37 = np.asarray(af2_outputs['structure_module']['final_atom_positions'])
  plddt = af2_outputs.get('plddt')
  if plddt is None and 'predicted_lddt' in af2_outputs:
    logits = np.asarray(af2_outputs['predicted_lddt']['logits'])
    # AF2's pLDDT head is 50 bins over [0, 100]; the score is the bin-centre
    # expectation, not an argmax. Reading the logits as if they were a score is
    # a trap this repo has hit before.
    bins = (np.arange(logits.shape[-1]) + 0.5) * (100.0 / logits.shape[-1])
    e = np.exp(logits - logits.max(-1, keepdims=True))
    plddt = ((e / e.sum(-1, keepdims=True)) * bins).sum(-1)

  coords, b_factors = atom37_to_token_atoms(atom37, batch, per_residue=plddt)
  return af3_model.predicted_structure_from_coords(
      coords[None], batch, predicted_lddt=b_factors[None])
