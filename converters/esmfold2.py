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


# ---------------------------------------------------------------------------
# the LM shim: 81 ESM-C layers -> a pair representation
# ---------------------------------------------------------------------------

def language_model_shim(sd, prefix='language_model'):
  """base_z_linear (LN + Linear) -> softmax-weighted layer mix -> SingleToPair.

  The layer mix `softmax(base_z_combine)` is a constant, so it folds into a
  plain (n_layers,) array.  Note it peaks on the LAST ESM-C layers (79/80/78
  hold 58% of the mass), so the tower cannot be truncated -- but only ~31 of
  81 hidden states carry 99%, which bounds what has to be materialised.
  """
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  w = _arr(g('base_z_combine')).astype(np.float64)
  mix = np.exp(w - w.max()); mix /= mix.sum()
  return {
      'combine': mix.astype(np.float32),
      'lm_norm/scale': _arr(g('base_z_linear.0.weight')),
      'lm_norm/offset': _arr(g('base_z_linear.0.bias')),
      'lm_projection/weights': t(g('base_z_linear.1.weight')),
      # SingleToPair: downproject, then output_mlp = Linear -> GELU -> Linear
      'downproject/weights': t(g('base_z_mlp.0.downproject.weight')),
      'downproject/bias': _arr(g('base_z_mlp.0.downproject.bias')),
      'pair_mlp_1/weights': t(g('base_z_mlp.0.output_mlp.0.weight')),
      'pair_mlp_1/bias': _arr(g('base_z_mlp.0.output_mlp.0.bias')),
      'pair_mlp_2/weights': t(g('base_z_mlp.0.output_mlp.2.weight')),
      'pair_mlp_2/bias': _arr(g('base_z_mlp.0.output_mlp.2.bias')),
      'pair_norm/scale': _arr(g('base_z_mlp.1.weight')),
      'pair_norm/offset': _arr(g('base_z_mlp.1.bias')),
  }


# ---------------------------------------------------------------------------
# the MSA encoder  (token-major [B, L, M, c], unlike AF3's [B, M, L, c])
# ---------------------------------------------------------------------------

def outer_product_mean(sd, prefix):
  """W is one Linear(d_msa, 2*d_hidden) chunked into (a, b).

  NOTE the divide order: ESMFold2's default is `Wout(outer) / n_valid`, so the
  output BIAS is scaled by 1/n_valid too.  (`divide_outer_before_proj=True`
  would put the divide inside; some checkpoints were trained that way.)
  """
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  w = _arr(g('W.weight'))
  h = w.shape[0] // 2
  return {'layer_norm/scale': _arr(g('norm.weight')),
          'layer_norm/offset': _arr(g('norm.bias')),
          'left_projection/weights': t(w[:h]),
          'right_projection/weights': t(w[h:]),
          'output/weights': t(g('Wout.weight')),
          'output/bias': _arr(g('Wout.bias'))}


def msa_pair_weighted_averaging(sd, prefix):
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  return {'msa_norm/scale': _arr(g('norm_single.weight')),
          'msa_norm/offset': _arr(g('norm_single.bias')),
          'pair_norm/scale': _arr(g('compute_bias.0.weight')),
          'pair_norm/offset': _arr(g('compute_bias.0.bias')),
          'bias/weights': t(g('compute_bias.1.weight')),
          'value/weights': t(g('Wv.weight')),
          'gate/weights': t(g('Wgate.weight')),
          'output/weights': t(g('Wout.weight'))}


def msa_block(sd, prefix, is_final):
  """OPM into pair, then (unless final) the MSA update, then the pair block.

  The last block drops the MSA update entirely -- its pair output is all that
  is consumed, so msa_pair_weighted_averaging / msa_transition are absent from
  the checkpoint (the same 'dead final block' shape intellifold2 has).
  """
  d = DIALECT_ESMFOLD2
  out = nest('outer_product_mean', outer_product_mean(sd, prefix + '.outer_product_mean'))
  if not is_final:
    out.update(nest('msa_pair_weighted_averaging',
                    msa_pair_weighted_averaging(sd, prefix + '.msa_pair_weighted_averaging')))
    out.update(nest('msa_transition', common.transition(sd, prefix + '.msa_transition', d)))
  out.update(pair_only_block(sd, prefix))
  return out


def msa_encoder(sd, dims, prefix='msa_encoder'):
  n = dims['n_msa']
  out = {'embed/weights': t(sd['%s.embed.weight' % prefix]),
         'project_inputs/weights': t(sd['%s.project_inputs.weight' % prefix])}
  # the final block has a different param set, so stack the common part only
  out.update(nest('blocks', stack_blocks(
      lambda i: msa_block(sd, '%s.blocks.%d' % (prefix, i), is_final=False),
      n - 1)))
  out.update(nest('final_block', msa_block(sd, '%s.blocks.%d' % (prefix, n - 1), is_final=True)))
  return out


# ---------------------------------------------------------------------------
# the whole trunk
# ---------------------------------------------------------------------------

def map_trunk(sd, dims=None):
  """Every parameter from features to the final pair representation."""
  dims = dims or derive_dims(sd)
  a_vec, b_T = parcae_dynamics(sd)
  p = {
      'z_init_1/weights': t(sd['z_init_1.weight']),
      'z_init_2/weights': t(sd['z_init_2.weight']),
      'rel_pos/weights': t(sd['rel_pos.embed.weight']),
      'token_bonds/weights': t(sd['token_bonds.weight']),
      'parcae_a': a_vec,
      'parcae_b/weights': b_T,
      'parcae_input_norm/scale': _arr(sd['parcae_input_norm.weight']),
      'parcae_input_norm/offset': _arr(sd['parcae_input_norm.bias']),
      'parcae_readout/weights': t(sd['parcae_readout.weight']),
      'distogram/weights': t(sd['distogram_head.weight']),
      'distogram/bias': _arr(sd['distogram_head.bias']),
  }
  p.update(nest('language_model', language_model_shim(sd)))
  p.update(nest('lm_encoder', pair_only_stack(sd, 'lm_encoder', dims['n_lm_encoder'])))
  p.update(nest('folding_trunk', pair_only_stack(sd, 'folding_trunk', dims['n_trunk'])))
  p.update(nest('parcae_coda', pair_only_stack(sd, 'parcae_coda', dims['n_coda'])))
  p.update(nest('msa_encoder', msa_encoder(sd, dims)))
  return p
