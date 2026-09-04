"""ESMFold2 (biohub/ESMFold2) -> AF3 haiku parameter conversion.

ESMFold2 is an ESM-C-conditioned all-atom diffusion predictor.  Two things make
it unlike every other model in this package:

  * **The trunk carries no single track.**  It is a stack of pair-only blocks
    (tri_mul_out / tri_mul_in / transition); there is no triangle attention and
    no pairformer single update.  `structure_head` is handed `s_trunk=None` --
    the only single representation anywhere is `s_inputs` (451-d).

  * **Recycling is a linear recurrence ("parcae"), not an addition.**  Instead
    of AF3's `z = z_init + z_recycle(norm(z_prev))`, ESMFold2 runs a discretised
    diagonal SSM over the loop axis:

        a = exp(-softplus(log_delta) * exp(log_a))          # (c,)   decay
        b = softplus(log_delta)[:, None] * b_cont           # (c, c) input map
        z = a * z + linear(parcae_input_norm(z_inject), b)  # per loop
        z = folding_trunk(z)

    `a` and `b` are input-independent, so they fold to plain arrays at
    conversion time -- a weight transform, not a forward branch (see the
    branch-vs-weight-fold rule in the playbook).  The initial state is *random*
    (truncated normal, std=sqrt(2/(5c))), so a parity harness must inject it.

Weight-layout divergences handled by the shared dialect:
  * tri-mul packs a/b projection *and* both gates into ONE `proj_bundle`
    Linear(c, 4h), split as [signal(2h) | gate_logits(2h)] then each chunked
    into (left, right)  ->  `tm_fused='bundle'`.
  * transitions pack SwiGLU as one `ffn.w12` Linear(c, 2h), split(h) then
    silu(x1)*x2  ->  `tr_mode='fused'`.
"""

import numpy as np

from . import common
from .common import Dialect, _arr, t, ln, stack_blocks, nest, populate


DIALECT_ESMFOLD2 = Dialect(
    # transitions: LN + fused-SwiGLU
    tr_ln='norm',
    tr_mode='fused',
    tr_lin='ffn.w12',
    tr_out='ffn.w3',
    # triangle multiplication: one bundled projection
    tm_fused='bundle',
    tm_ln_in='_engine.norm_start',
    tm_ln_out='_engine.norm_mix',
    tm_g='_engine.proj_gate',
    tm_z='_engine.proj_emit',
    tm_ab_p='_engine.proj_bundle',
)


# ---------------------------------------------------------------------------
# dims read off the checkpoint (never hardcoded -- the D36 lesson)
# ---------------------------------------------------------------------------

def derive_dims(sd):
  """Read every block count and width off the state dict."""
  def n_blocks(prefix):
    idx = set()
    for k in sd:
      if k.startswith(prefix + '.blocks.'):
        idx.add(int(k[len(prefix) + 8:].split('.')[0]))
    return len(idx)

  c_z = sd['parcae_log_a'].shape[0]
  s_inputs = sd['z_init_1.weight'].shape[1]
  d = dict(
      c_z=c_z,
      s_inputs=s_inputs,
      c_single=sd['confidence_head.s_input_to_s.weight'].shape[0],
      n_trunk=n_blocks('folding_trunk'),
      n_lm_encoder=n_blocks('lm_encoder'),
      n_coda=n_blocks('parcae_coda'),
      n_msa=n_blocks('msa_encoder'),
      n_conf=n_blocks('confidence_head.folding_trunk'),
      num_bins=sd['distogram_head.weight'].shape[0],
      lm_layers=sd['language_model.base_z_combine'].shape[0],
      lm_width=sd['language_model.base_z_linear.1.weight'].shape[1],
      relpos_in=sd['rel_pos.embed.weight'].shape[1],
      c_msa=sd['msa_encoder.embed.weight'].shape[0],
      msa_in=sd['msa_encoder.embed.weight'].shape[1],
      c_atom=sd['inputs_embedder.atom_attention_encoder.atom_linear.weight'].shape[0],
      atom_in=sd['inputs_embedder.atom_attention_encoder.atom_linear.weight'].shape[1],
      c_diff=sd['structure_head.diffusion_module.s_to_token.weight'].shape[0],
  )
  # tri-mul latent width: proj_bundle is (4*h, c)
  d['tri_hidden'] = sd['folding_trunk.blocks.0.tri_mul_out._engine.proj_bundle.weight'].shape[0] // 4
  d['tri_transition_hidden'] = sd['folding_trunk.blocks.0.pair_transition.ffn.w12.weight'].shape[0] // 2
  return d


# ---------------------------------------------------------------------------
# the pair-only stacks (folding_trunk / lm_encoder / parcae_coda / confidence)
# ---------------------------------------------------------------------------

def pair_only_block(sd, prefix):
  """tri_mul_out + tri_mul_in + pair_transition.  No attention, no single."""
  d = DIALECT_ESMFOLD2
  out = {}
  out.update(nest('triangle_multiplication_outgoing',
                  common.triangle_mul(sd, prefix + '.tri_mul_out', d, outgoing=True)))
  out.update(nest('triangle_multiplication_incoming',
                  common.triangle_mul(sd, prefix + '.tri_mul_in', d, outgoing=False)))
  out.update(nest('pair_transition',
                  common.transition(sd, prefix + '.pair_transition', d)))
  return out


def pair_only_stack(sd, prefix, n):
  return stack_blocks(lambda i: pair_only_block(sd, '%s.blocks.%d' % (prefix, i)), n)


# ---------------------------------------------------------------------------
# parcae: fold the static SSM dynamics into plain arrays
# ---------------------------------------------------------------------------

def parcae_dynamics(sd):
  """Return (a, b_T) with a=(c,) decay and b_T=(c,c) haiku-oriented input map.

  Mirrors ESMFold2Model._discretized_dynamics exactly:
      delta = softplus(log_delta);  a = exp(-delta * exp(log_a))
      b     = delta[:, None] * b_cont
  and the forward applies `F.linear(x, b)` = x @ b.T, so haiku wants b.T.
  """
  log_delta = _arr(sd['parcae_log_delta']).astype(np.float64)
  log_a = _arr(sd['parcae_log_a']).astype(np.float64)
  b_cont = _arr(sd['parcae_b_cont']).astype(np.float64)
  delta = np.logaddexp(0.0, log_delta)              # softplus, stable
  a = np.exp(-delta * np.exp(log_a))
  b = delta[:, None] * b_cont
  return a.astype(np.float32), b.T.astype(np.float32).copy()
