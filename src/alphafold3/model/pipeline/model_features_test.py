# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tests for the per-model featurisation conventions.

Every convention here is SILENT when it is wrong: the batch stays the right
shape, the model runs, and the fold is merely worse. So each is pinned against
what the source model actually does rather than against a previous run.
"""

import numpy as np

from absl.testing import absltest
from alphafold3.model import model_registry
from alphafold3.model.pipeline import model_features


def _atom_window_batch(num_subsets=6, q_size=32, k_size=128, n_real=None):
  """A minimal batch carrying only the atom-attention gathers."""
  n_padded = num_subsets * q_size
  n_real = n_padded if n_real is None else n_real
  flat = np.arange(n_padded)
  mask = flat < n_real
  return {
      'token_atoms_to_queries:gather_idxs': flat.reshape(num_subsets, q_size),
      'token_atoms_to_queries:gather_mask': mask.reshape(num_subsets, q_size),
      'queries_to_keys:gather_idxs': np.zeros((num_subsets, k_size), np.int32),
      'queries_to_keys:gather_mask': np.zeros((num_subsets, k_size), bool),
      'tokens_to_queries:gather_idxs': (flat // 4).reshape(num_subsets, q_size),
      'tokens_to_queries:gather_mask': mask.reshape(num_subsets, q_size),
      'tokens_to_keys:gather_idxs': np.zeros((num_subsets, k_size), np.int32),
      'tokens_to_keys:gather_mask': np.zeros((num_subsets, k_size), bool),
  }


class KeyWindowTest(absltest.TestCase):

  def test_padding_masks_the_edge_blocks_instead_of_sliding_them(self):
    """opendde/protenix pad the key window; AF3 slides it back in bounds.

    Both give each block of 32 queries a 128-key window running from
    block_start - 48 to block_start + 79. AF3 shifts a window that runs off
    either end bodily back inside the atom count, so every block keeps 128 valid
    keys. opendde zero-pads instead and masks those slots, so its edge blocks
    genuinely see fewer neighbours -- and the ones they do see sit at DIFFERENT
    key slots, so the per-slot pair bias is misaligned too.
    """
    batch = model_features._padded_key_window(_atom_window_batch())

    keep = np.asarray(batch['queries_to_keys:gather_mask'])
    # The first block's leading 48 keys are before atom 0, the last block's
    # trailing 48 are past the end. Interior blocks are untouched.
    self.assertEqual(keep[0, :48].sum(), 0)
    self.assertTrue(keep[0, 48:].all())
    self.assertEqual(keep[-1, -48:].sum(), 0)
    for block in range(2, 4):
      self.assertTrue(keep[block].all(), f'interior block {block} lost keys')

  def test_padding_never_points_a_key_outside_the_atoms(self):
    batch = model_features._padded_key_window(_atom_window_batch())
    idxs = np.asarray(batch['queries_to_keys:gather_idxs'])
    self.assertTrue((idxs >= 0).all())
    self.assertTrue((idxs < 6 * 32).all())

  def test_padding_masks_atoms_that_are_only_padding(self):
    # A window slot may be in bounds and still be padding: the batch is padded
    # up to whole blocks, and those atoms are not real.
    batch = model_features._padded_key_window(_atom_window_batch(n_real=100))
    idxs = np.asarray(batch['queries_to_keys:gather_idxs'])
    keep = np.asarray(batch['queries_to_keys:gather_mask'])
    self.assertFalse(keep[idxs >= 100].any())

  def test_the_circular_window_wraps_rather_than_clipping(self):
    """chai-1 takes the window modulo the padded atom count.

    So the first block's leading 48 keys land at the END of the atom list, not
    at atom 0 -- which is where they land if you clip, and where AF3's slide
    would put real neighbours.
    """
    batch = model_features._circular_key_window(_atom_window_batch())
    idxs = np.asarray(batch['queries_to_keys:gather_idxs'])
    n_padded = 6 * 32
    np.testing.assert_array_equal(idxs[0, :48],
                                  np.arange(n_padded - 48, n_padded))
    self.assertTrue((idxs >= 0).all() and (idxs < n_padded).all())

  def test_the_two_conventions_disagree_only_at_the_edges(self):
    padded = model_features._padded_key_window(_atom_window_batch())
    circular = model_features._circular_key_window(_atom_window_batch())
    p = np.asarray(padded['queries_to_keys:gather_idxs'])
    c = np.asarray(circular['queries_to_keys:gather_idxs'])
    # The window reaches 48 atoms back and 79 forward, so with 32 queries per
    # block the first TWO and last TWO blocks run off the end; everything
    # between them is in bounds and the conventions cannot differ there.
    np.testing.assert_array_equal(p[2:-2], c[2:-2])
    self.assertFalse(np.array_equal(p[0], c[0]))
    self.assertFalse(np.array_equal(p[1], c[1]))


class ApplyTest(absltest.TestCase):

  def test_a_model_with_no_conventions_is_untouched(self):
    for name in ('alphafold3', 'openfold3', 'intellifold2'):
      with self.subTest(model=name):
        before = _atom_window_batch()
        after = model_features.apply(_atom_window_batch(),
                                     model_registry.get(name))
        for key, value in before.items():
          np.testing.assert_array_equal(np.asarray(after[key]), value)

  def test_opendde_without_a_refeaturise_callable_is_an_error(self):
    # Rather than silently skipping the structural-token batch its diffusion
    # runs on, which would fold the residue tokens and look almost right.
    with self.assertRaises(ValueError):
      model_features.apply(_atom_window_batch(), model_registry.get('opendde'))


class _Chain:

  def __init__(self, chain_id, ptms=()):
    self.id = chain_id
    self.ptms = list(ptms)


class _FoldInput:

  def __init__(self, chains):
    self.chains = chains


def _token_batch(asym_id, residue_index, aatype=None):
  asym_id = np.asarray(asym_id)
  return {
      'asym_id': asym_id,
      'residue_index': np.asarray(residue_index),
      'aatype': np.asarray(aatype if aatype is not None
                           else np.zeros_like(asym_id)),
  }


class CyclicPeriodTest(absltest.TestCase):
  """Cyclic is not a per-model convention: it wraps the SHARED encoding."""

  def test_every_chain_wraps_at_its_own_residue_count(self):
    batch = _token_batch([1, 1, 1, 2, 2, 0], [1, 2, 3, 1, 2, 0])
    out = model_features.cyclic_period(batch)['cyclic_period']
    np.testing.assert_array_equal(out, [3, 3, 3, 2, 2, 0])

  def test_the_period_is_residues_not_tokens(self):
    """A ligand contributes many tokens at ONE residue index.

    Wrapping on the token count would misplace every polymer offset in the same
    chain, and nothing would error.
    """
    batch = _token_batch([1, 1, 1, 1], [1, 2, 3, 3])
    out = model_features.cyclic_period(batch)['cyclic_period']
    np.testing.assert_array_equal(out, [3, 3, 3, 3])

  def test_naming_chains_leaves_the_others_linear(self):
    batch = _token_batch([1, 1, 2, 2], [1, 2, 1, 2])
    fold_input = _FoldInput([_Chain('A'), _Chain('B')])
    out = model_features.cyclic_period(batch, ['B'],
                                       fold_input=fold_input)['cyclic_period']
    np.testing.assert_array_equal(out, [0, 0, 2, 2])

  def test_an_unknown_chain_id_is_an_error(self):
    batch = _token_batch([1, 1], [1, 2])
    with self.assertRaises(ValueError):
      model_features.cyclic_period(batch, ['Z'],
                                   fold_input=_FoldInput([_Chain('A')]))


class ModifiedResidueTest(absltest.TestCase):
  """Boltz-2 reads a modified residue as one token, UNK, and a flag -- all three.

  Measured on phospho-ubiquitin: the restype alone leaves the phosphate 6.42 A
  out, the flag alone 6.53 A, the two together 3.10 A.
  """

  def test_the_flag_and_the_restype_land_on_the_modified_residue(self):
    batch = _token_batch([1, 1, 1, 2], [1, 2, 3, 1], aatype=[5, 6, 7, 8])
    fold_input = _FoldInput([_Chain('A', [('PTR', 2)]), _Chain('B')])
    out = model_features._mark_modified_residues(batch, fold_input,
                                                 unknown_restype=True)
    np.testing.assert_array_equal(out['is_modified'], [0, 1, 0, 0])
    self.assertEqual(int(out['aatype'][1]), 20)   # UNK, not the parent restype
    np.testing.assert_array_equal(out['aatype'][[0, 2, 3]], [5, 7, 8])

  def test_every_token_of_an_atomised_residue_is_marked(self):
    """Under AF3's convention one residue is several tokens sharing one index.

    So the tokens have to be found by (chain, residue), not by position.
    """
    batch = _token_batch([1, 1, 1, 1], [1, 2, 2, 3], aatype=[5, 6, 6, 7])
    out = model_features._mark_modified_residues(
        batch, _FoldInput([_Chain('A', [('PTR', 2)])]))
    np.testing.assert_array_equal(out['is_modified'], [0, 1, 1, 0])

  def test_the_restype_change_is_opt_in(self):
    # Every family except boltz2 keeps AF3's convention, the PARENT restype.
    batch = _token_batch([1, 1], [1, 2], aatype=[5, 6])
    out = model_features._mark_modified_residues(
        batch, _FoldInput([_Chain('A', [('PTR', 2)])]))
    np.testing.assert_array_equal(out['aatype'], [5, 6])
    np.testing.assert_array_equal(out['is_modified'], [0, 1])


if __name__ == '__main__':
  absltest.main()
