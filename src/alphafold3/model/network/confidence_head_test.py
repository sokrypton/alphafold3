# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tests for the confidence head's shared pieces."""

import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from alphafold3.model.network import confidence_head


class MaskedGlobalNormTest(absltest.TestCase):
  """The invariant: what bucket an input is padded into cannot change the answer.

  RoseTTAFold3's confidence head normalises each trunk input over the whole
  tensor, so this is the one place in the head where padding can reach a real
  token's value. It did: 76 residues in the 128-token bucket produced a PAE of
  ~28 A everywhere and pTM 0.04, against 0.89 for the same fold unpadded.
  """

  def _padded(self, real, total, channels=8, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((total, channels), np.float32)
    x[:real] = rng.normal(3.0, 2.0, (real, channels))
    mask = np.arange(total) < real
    return jnp.asarray(x), jnp.asarray(mask)

  def test_padding_does_not_change_the_real_tokens(self):
    x_small, mask_small = self._padded(real=12, total=12)
    x_big, mask_big = self._padded(real=12, total=64)
    np.testing.assert_allclose(
        np.asarray(confidence_head.masked_global_norm(x_small, mask_small)),
        np.asarray(confidence_head.masked_global_norm(x_big, mask_big))[:12],
        rtol=1e-5, atol=1e-5)

  def test_unmasked_statistics_would_have_differed(self):
    """The bug this replaced, stated as a fact rather than a memory."""
    x_small, _ = self._padded(real=12, total=12)
    x_big, _ = self._padded(real=12, total=64)
    naive = lambda x: (x - x.mean()) / jnp.sqrt(x.var() + 1e-5)
    self.assertFalse(np.allclose(np.asarray(naive(x_small)),
                                 np.asarray(naive(x_big))[:12],
                                 rtol=1e-3, atol=1e-3))

  def test_the_masked_region_is_normalised(self):
    x, mask = self._padded(real=20, total=48)
    out = np.asarray(confidence_head.masked_global_norm(x, mask))[:20]
    self.assertAlmostEqual(float(out.mean()), 0.0, places=4)
    self.assertAlmostEqual(float(out.std()), 1.0, places=3)

  def test_a_pair_mask_broadcasts_over_two_token_axes(self):
    rng = np.random.default_rng(1)
    x = np.zeros((16, 16, 4), np.float32)
    x[:6, :6] = rng.normal(size=(6, 6, 4))
    mask1d = np.arange(16) < 6
    mask = jnp.asarray(mask1d[:, None] & mask1d[None, :])
    out = np.asarray(confidence_head.masked_global_norm(jnp.asarray(x), mask))
    self.assertAlmostEqual(float(out[:6, :6].mean()), 0.0, places=4)
    self.assertAlmostEqual(float(out[:6, :6].std()), 1.0, places=3)


if __name__ == '__main__':
  absltest.main()
