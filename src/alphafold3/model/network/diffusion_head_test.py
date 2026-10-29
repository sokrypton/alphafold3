"""The sampler's guidance hook (diffusion_head.DENOISED_HOOK / make_denoising_body), and its Blitz options."""

import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from alphafold3.model.network import diffusion_head

_SHAPE = (4, 3, 3)   # tokens, atoms per token, xyz


def _run(hook, chai=False, steps=3):
  """Scan the step body over a short schedule with a denoiser that predicts the origin."""
  config = diffusion_head.SampleConfig(steps=steps, gamma_0=0.0)
  denoise = lambda x, t, **kw: jnp.zeros_like(x)
  body = diffusion_head.make_denoising_body(
      denoise, jnp.ones(_SHAPE[:2], bool), config, None, chai, hook=hook)
  levels = jnp.asarray([10.0, 5.0, 2.0, 1.0][:steps + 1])
  xs = (levels[1:], levels[:-1])
  if hook is not None:
    xs = (*xs, jnp.arange(steps))
  carry = (jax.random.PRNGKey(0), jnp.ones(_SHAPE))
  _, (_, denoised) = jax.lax.scan(body, carry, xs)
  return np.asarray(denoised)


class DenoisedHookTest(absltest.TestCase):

  def test_hook_replaces_every_denoised_prediction(self):
    hook = lambda x, sigma, step: x + 1.0 + step
    denoised = _run(hook)
    for step in range(3):
      np.testing.assert_allclose(denoised[step], 1.0 + step)

  def test_hook_sees_noise_level_prev(self):
    hook = lambda x, sigma, step: jnp.full_like(x, sigma)
    denoised = _run(hook)
    np.testing.assert_allclose(denoised[:, 0, 0, 0], [10.0, 5.0, 2.0])

  def test_no_hook_is_the_plain_denoiser(self):
    denoised = _run(None)
    np.testing.assert_allclose(denoised, 0.0)

  def test_chai_second_order_step_is_hooked_too(self):
    calls = []

    def hook(x, sigma, step):
      calls.append(step)
      return x
    _run(hook, chai=True)
    # traced once by scan: the first denoising call and the second-order one
    self.assertLen(calls, 2)


_SHIFT = jnp.asarray([1.0, -2.0, 0.5])


def _one_step(chai=False, **options):
  """One step from sigma 10 to 5 with D(x) = x + d: the move it makes off the noisy state."""
  config = diffusion_head.SampleConfig(steps=1, churn=False, **options)
  denoise = lambda x, t, **kw: x + _SHIFT
  body = diffusion_head.make_denoising_body(
      denoise, jnp.ones(_SHAPE[:2], bool), config, None, chai)
  carry = (jax.random.PRNGKey(0), jnp.ones(_SHAPE))
  (_, out), (_, denoised) = body(carry, (jnp.asarray(5.0), jnp.asarray(10.0)))
  noisy = np.asarray(denoised) - np.asarray(_SHIFT)
  move = (np.asarray(out) - noisy).reshape(-1, 3)
  np.testing.assert_allclose(move, np.broadcast_to(move[0], move.shape), atol=1e-5)   # every atom alike
  return move[0]


class BlitzOptionsTest(absltest.TestCase):
  """localfold opus-55-opt's sampler: the exponential update and the geometric schedule."""

  def test_euler_moves_s_times_one_minus_r(self):
    np.testing.assert_allclose(_one_step(step_scale=1.5), 1.5 * 0.5 * _SHIFT, rtol=1e-5)

  def test_exponential_moves_one_minus_r_to_the_s(self):
    move = _one_step(update='exponential', step_scale=1.5)
    np.testing.assert_allclose(move, (1 - 0.5 ** 1.5) * _SHIFT, rtol=1e-5)

  def test_chai_one_call_steps(self):
    np.testing.assert_allclose(_one_step(chai=True, calls_per_step=1, step_scale=2.0),
                               2.0 * 0.5 * _SHIFT, rtol=1e-4)
    np.testing.assert_allclose(_one_step(chai=True, update='exponential', step_scale=2.0),
                               (1 - 0.5 ** 2) * _SHIFT, rtol=1e-4)

  def test_chai_exponential_heun_step(self):
    # x1 = x + (1 - r) d, so D2 = x + (2 - r) d, and the mean prediction is x + (1.5 - r / 2) d
    move = _one_step(chai=True, update='exponential', step_scale=2.0, calls_per_step=2)
    np.testing.assert_allclose(move, (1 - 0.5 ** 2) * (1.5 - 0.25) * _SHIFT, rtol=1e-4)

  def test_two_calls_are_chai_only(self):
    with self.assertRaisesRegex(ValueError, 'chai-1 takes 1 or 2'):
      _one_step(calls_per_step=2)
    with self.assertRaisesRegex(ValueError, 'euler or exponential'):
      _one_step(update='heun')

  def test_geometric_schedule_is_even_in_log_sigma_at_the_same_ends(self):
    karras = np.asarray(diffusion_head.noise_schedule_for(diffusion_head.SampleConfig(steps=25), False))
    geometric = np.asarray(diffusion_head.noise_schedule_for(
        diffusion_head.SampleConfig(steps=25, schedule='geometric'), False))
    self.assertLen(geometric, 26)
    np.testing.assert_allclose(geometric[[0, -1]], karras[[0, -1]], rtol=1e-5)
    np.testing.assert_allclose(np.diff(np.log(geometric)), np.log(0.0004 / 160) / 25, rtol=1e-4)
    with self.assertRaisesRegex(ValueError, 'karras or geometric'):
      diffusion_head.noise_schedule_for(diffusion_head.SampleConfig(steps=2, schedule='cosine'), False)


if __name__ == '__main__':
  absltest.main()
