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
      n_diff_atom=len({k[len('structure_head.diffusion_module.atom_encoder.atom_transformer.blocks.'):].split('.')[0]
                       for k in sd if k.startswith('structure_head.diffusion_module.atom_encoder.atom_transformer.blocks.')}),
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


# ---------------------------------------------------------------------------
# the SWA / 3D-RoPE atom transformer  (the one primitive with no AF3 analogue)
# ---------------------------------------------------------------------------
#
# ESMFold2 does NOT use AF3's 32-query/128-key windowed atom attention with a
# pair bias.  Instead it runs a plain sliding-window self-attention over atoms
# (half_window = swa_window_size // 2) whose positional signal is a 3D rotary
# embedding built from the reference conformer:
#
#   spatial: ref_pos (3 axes) x n_spatial_rope_pairs_per_axis, base freq 20
#   uid:     ref_space_uid    x n_uid_rope_pairs,              base freq 10000
#   -> 3*2 + 10 = 16 = head_dim/2, which fills the rotary half exactly.
#
# The window is over the RANK among *valid* atoms, not raw index, and the
# diagonal is always allowed.  Blocks are adaLN-Zero with NON-affine RMSNorm,
# and q/k get an extra affine-free RMSNorm before the rotation.

def swa_atom_block(sd, prefix):
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  return {
      # Sequential(SiLU, Linear(d, 6d)) -> shift_a, scale_a, gate_a, shift_f, scale_f, gate_f
      'adaln/weights': t(g('adaln_modulation.1.weight')),
      'qkv/weights': t(g('attn.Wqkv.weight')),
      'attn_gate/weights': t(g('attn.gate_proj.weight')),
      'attn_out/weights': t(g('attn.out_proj.weight')),
      'ffn_up/weights': t(g('ffn.w_up.weight')),
      'ffn_down/weights': t(g('ffn.w_down.weight')),
  }


def atom_encoder(sd, prefix, n_blocks, structure_prediction=False):
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  out = {
      'atom_linear/weights': t(g('atom_linear.weight')),
      'atom_norm/scale': _arr(g('atom_norm.weight')),
      'atom_norm/offset': _arr(g('atom_norm.bias')),
      'atom_to_token/weights': t(g('atom_to_token_linear.weight')),
  }
  if structure_prediction:
    out['coords_linear/weights'] = t(g('coords_linear.weight'))
  out.update(nest('blocks', stack_blocks(
      lambda i: swa_atom_block(sd, '%s.atom_transformer.blocks.%d' % (prefix, i)),
      n_blocks)))
  return out


def rope_inv_freq(n_pairs, base):
  """1 / base**(arange(n_pairs)/n_pairs) -- the ESMFold2 spacing."""
  return (1.0 / (base ** (np.arange(n_pairs, dtype=np.float32) / n_pairs))).astype(np.float32)


# ---------------------------------------------------------------------------
# the diffusion module
# ---------------------------------------------------------------------------
#
# Conditioning (note there is NO s_trunk anywhere -- the head is handed None):
#   z = z_proj(z_input_norm([z_trunk | rel_pos]))  then 2 unconditioned transitions
#   s = s_proj(s_input_norm(s_inputs))             then + noise, then 2 transitions
#   t_noise = 0.25 * log(t / sigma_data)           (== AF3's log(sigma/sigma_data)/4)
#   n = noise_proj(noise_norm(fourier(t_noise))),  fourier = cos(2*pi*(t*w + b))
#
# The z/s transitions use SEPARATE a_proj/b_proj SwiGLU halves (unlike the trunk's
# fused ffn.w12), so they take the 'block' dialect rather than 'fused'.

DIALECT_ESMFOLD2_DIFF = Dialect(
    tr_ln='norm',
    tr_mode='block',
    tr_a='a_proj',
    tr_b='b_proj',
    tr_out='out_proj',
)


def adaptive_layer_norm(sd, prefix):
  """sigmoid(s_gate(LN(s; scale=s_scale, no offset))) * LN(a) + s_shift(LN(s)).

  Note s_norm has a learned SCALE but NO offset, and s_gate carries a bias
  (initialised to -2, i.e. the adaLN-Zero gate starts near-closed).
  """
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  return {'s_norm/scale': _arr(g('s_scale')),
          'gate/weights': t(g('s_gate.weight')),
          'gate/bias': _arr(g('s_gate.bias')),
          'shift/weights': t(g('s_shift.weight'))}


def diffusion_attn_block(sd, prefix):
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  out = nest('adaln', adaptive_layer_norm(sd, prefix + '.adaln'))
  out.update({
      'q/weights': t(g('q_proj.weight')),
      'q/bias': _arr(g('q_proj.bias')),
      'kv/weights': t(g('kv_proj.weight')),        # fused [k | v]
      'g/weights': t(g('g_proj.weight')),
      'out/weights': t(g('out_proj.weight')),
      'out_gate/weights': t(g('out_gate.weight')),
      'out_gate/bias': _arr(g('out_gate.bias')),
      'pair_norm/scale': _arr(g('pair_norm.weight')),
      'pair_norm/offset': _arr(g('pair_norm.bias')),
      'pair_bias/weights': t(g('pair_bias_proj.weight')),
  })
  return out


def diffusion_transition_block(sd, prefix):
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  out = nest('adaln', adaptive_layer_norm(sd, prefix + '.adaln'))
  out.update({
      'swish/weights': t(g('lin_swish.weight')),   # fused [a | b], silu(a)*b
      'out/weights': t(g('lin_out.weight')),
      'out_gate/weights': t(g('output_gate.weight')),
      'out_gate/bias': _arr(g('output_gate.bias')),
  })
  return out


def diffusion_conditioning(sd, prefix):
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  d = DIALECT_ESMFOLD2_DIFF
  out = {
      'z_input_norm/scale': _arr(g('z_input_norm.weight')),
      'z_input_norm/offset': _arr(g('z_input_norm.bias')),
      'z_projection/weights': t(g('z_proj.weight')),
      's_input_norm/scale': _arr(g('s_input_norm.weight')),
      's_input_norm/offset': _arr(g('s_input_norm.bias')),
      's_projection/weights': t(g('s_proj.weight')),
      'fourier_w': _arr(g('fourier.w')),
      'fourier_b': _arr(g('fourier.b')),
      'noise_norm/scale': _arr(g('noise_norm.weight')),
      'noise_norm/offset': _arr(g('noise_norm.bias')),
      'noise_projection/weights': t(g('noise_proj.weight')),
  }
  n_z = len({k.split('.')[len(prefix.split('.')) + 1] for k in sd
             if k.startswith(prefix + '.z_transitions.')})
  n_s = len({k.split('.')[len(prefix.split('.')) + 1] for k in sd
             if k.startswith(prefix + '.s_transitions.')})
  out.update(nest('z_transitions', stack_blocks(
      lambda i: common.transition(sd, '%s.z_transitions.%d' % (prefix, i), d), n_z)))
  out.update(nest('s_transitions', stack_blocks(
      lambda i: common.transition(sd, '%s.s_transitions.%d' % (prefix, i), d), n_s)))
  return out


def map_diffusion(sd, dims=None):
  dims = dims or derive_dims(sd)
  P = 'structure_head.diffusion_module'
  n_tok = len({k[len(P + '.token_transformer.attn_blocks.'):].split('.')[0]
               for k in sd if k.startswith(P + '.token_transformer.attn_blocks.')})
  out = nest('conditioning', diffusion_conditioning(sd, P + '.conditioning'))
  out.update({
      's_step_norm/scale': _arr(sd[P + '.s_step_norm.weight']),
      's_step_norm/offset': _arr(sd[P + '.s_step_norm.bias']),
      's_to_token/weights': t(sd[P + '.s_to_token.weight']),
      'token_norm/scale': _arr(sd[P + '.token_norm.weight']),
      'token_norm/offset': _arr(sd[P + '.token_norm.bias']),
  })
  out.update(nest('token_attn', stack_blocks(
      lambda i: diffusion_attn_block(sd, '%s.token_transformer.attn_blocks.%d' % (P, i)), n_tok)))
  out.update(nest('token_transition', stack_blocks(
      lambda i: diffusion_transition_block(sd, '%s.token_transformer.transition_blocks.%d' % (P, i)), n_tok)))
  out.update(nest('atom_encoder', atom_encoder(sd, P + '.atom_encoder',
                                               dims['n_diff_atom'], structure_prediction=True)))
  out.update(nest('atom_decoder', {
      'token_to_atom/weights': t(sd[P + '.atom_decoder.token_to_atom_linear.weight']),
      'norm/scale': _arr(sd[P + '.atom_decoder.norm.weight']),
      'norm/offset': _arr(sd[P + '.atom_decoder.norm.bias']),
      'output/weights': t(sd[P + '.atom_decoder.output_linear.weight']),
      **nest('blocks', stack_blocks(
          lambda i: swa_atom_block(sd, '%s.atom_decoder.atom_transformer.blocks.%d' % (P, i)),
          dims['n_diff_atom'])),
  }))
  return out


# ---------------------------------------------------------------------------
# the confidence head
# ---------------------------------------------------------------------------
#
# Structure: build a pair rep from z + rel_pos + token_bonds + three s_inputs
# projections (row, column and an outer PRODUCT), add a distogram-bin embedding
# of the PREDICTED representative-atom distances, run 4 pair-only blocks, and
# read pLDDT / resolved off a row-attention pooling of the pair, pae / pde off
# the pair directly.
#
# Two things a name-level check would miss:
#   * the 4-block trunk is wrapped in ONE MORE residual: `pair.add_(trunk(pair))`.
#     The blocks already carry their own residuals, so this is a stack-level skip
#     on top of them.
#   * THREE parameters are DEAD in the released checkpoint -- s_inputs_to_single,
#     s_input_to_s and s_norm are constructed in __init__ and never read by
#     forward.  They are converted anyway (harmless, and it keeps coverage at
#     100%), but nothing consumes them.

CONFIDENCE_DEAD_PARAMS = ('s_inputs_to_single', 's_input_to_s', 's_norm')


def confidence_head(sd, dims, prefix='confidence_head'):
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  out = {
      'boundaries': _arr(g('boundaries')),
      'dist_bin_embed/weights': _arr(g('dist_bin_pairwise_embed.weight')),
      's_inputs_norm/scale': _arr(g('s_inputs_norm.weight')),
      's_inputs_norm/offset': _arr(g('s_inputs_norm.bias')),
      'z_norm/scale': _arr(g('z_norm.weight')),
      'z_norm/offset': _arr(g('z_norm.bias')),
      's_to_z/weights': t(g('s_to_z.weight')),
      's_to_z_transpose/weights': t(g('s_to_z_transpose.weight')),
      's_to_z_prod_in1/weights': t(g('s_to_z_prod_in1.weight')),
      's_to_z_prod_in2/weights': t(g('s_to_z_prod_in2.weight')),
      's_to_z_prod_out/weights': t(g('s_to_z_prod_out.weight')),
      'row_pool_attn/weights': t(g('row_attention_pooling.attn_proj.weight')),
      'row_pool_out/weights': t(g('row_attention_pooling.out_proj.weight')),
      'plddt_norm/scale': _arr(g('plddt_ln.weight')),
      'plddt_norm/offset': _arr(g('plddt_ln.bias')),
      'plddt_weight': _arr(g('plddt_weight')),          # (max_atoms_per_token, c_s, bins)
      'resolved_norm/scale': _arr(g('resolved_ln.weight')),
      'resolved_norm/offset': _arr(g('resolved_ln.bias')),
      'resolved_weight': _arr(g('resolved_weight')),
      'pae_norm/scale': _arr(g('pae_ln.weight')),
      'pae_norm/offset': _arr(g('pae_ln.bias')),
      'pae/weights': t(g('pae_head.weight')),
      'pde_norm/scale': _arr(g('pde_ln.weight')),
      'pde_norm/offset': _arr(g('pde_ln.bias')),
      'pde/weights': t(g('pde_head.weight')),
      # dead in the released checkpoint -- carried for coverage only
      'unused_s_inputs_to_single/weights': t(g('s_inputs_to_single.weight')),
      'unused_s_input_to_s/weights': t(g('s_input_to_s.weight')),
      'unused_s_norm/scale': _arr(g('s_norm.weight')),
      'unused_s_norm/offset': _arr(g('s_norm.bias')),
  }
  out.update(nest('folding_trunk',
                  pair_only_stack(sd, prefix + '.folding_trunk', dims['n_conf'])))
  return out


def map_esmfold2_to_af3(sd, **overrides):
  """Everything converted so far: trunk + inputs embedder + diffusion module.

  Everything except the ESM-C tower, which is a separate artifact.
  """
  dims = derive_dims(sd)
  dims.update(overrides)
  p = map_trunk(sd, dims)
  p.update(nest('inputs_embedder',
                atom_encoder(sd, 'inputs_embedder.atom_attention_encoder',
                             dims.get('n_input_atom', 3))))
  p.update(nest('diffusion', map_diffusion(sd, dims)))
  p.update(nest('confidence', confidence_head(sd, dims)))
  return p


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def load_esmfold2_checkpoint(checkpoint):
  """Accept a .safetensors file, a HF snapshot dir, or an .npz of the state dict."""
  import glob
  import os
  path = os.path.expanduser(str(checkpoint))
  if os.path.isdir(path):
    hits = sorted(glob.glob(os.path.join(path, '*.safetensors')))
    if not hits:
      raise FileNotFoundError('no .safetensors under %s' % path)
    return _read_safetensors(hits)          # ESM-C ships 6 shards, ESMFold2 one
  if path.endswith('.npz'):
    return {k: v for k, v in np.load(path).items()}
  return _read_safetensors([path])


_ST_DTYPES = {'F64': '<f8', 'F32': '<f4', 'F16': '<f2', 'I64': '<i8', 'I32': '<i4',
              'I16': '<i2', 'I8': 'i1', 'U8': 'u1', 'BOOL': '?'}


def _read_safetensors(paths):
  """Minimal pure-numpy safetensors reader.

  The `safetensors` package is not in every venv here and this repo's GPU venv
  must not grow dependencies, so read the container directly: 8-byte
  little-endian header length, then a JSON header mapping name ->
  {dtype, shape, data_offsets}, then the raw buffers.  BF16 is returned as
  float32 (numpy has no bfloat16) by widening the 16-bit pattern.
  """
  import json
  out = {}
  for path in paths:
    with open(path, 'rb') as fh:
      n = int.from_bytes(fh.read(8), 'little')
      header = json.loads(fh.read(n))
      base = 8 + n
      for key, meta in header.items():
        if key == '__metadata__' or key.endswith('_extra_state'):
          continue
        start, end = meta['data_offsets']
        fh.seek(base + start)
        raw = fh.read(end - start)
        if meta['dtype'] == 'BF16':
          u16 = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
          arr = u16.view(np.float32) if u16.dtype == np.uint32 else u16.astype(np.float32)
          arr = u16.astype(np.uint32).view(np.float32)
        else:
          arr = np.frombuffer(raw, dtype=_ST_DTYPES[meta['dtype']])
        out[key] = arr.reshape(meta['shape'])
  return out


def convert_esmfold2_weights(checkpoint, output_dir):
  """Convert biohub/ESMFold2 to a loadable AF3-haiku blob dir.

  This covers the 234.8M FOLDING model only.  ESMFold2 additionally requires
  ESM-C 6B (6.35B params, 25.4 GB fp32) whose hidden states feed the
  LanguageModelShim; that tower is a separate artifact and is not written here.
  """
  from pathlib import Path
  sd = load_esmfold2_checkpoint(checkpoint)
  # the GRAPH map, not map_esmfold2_to_af3: the latter keys on the reference
  # implementation's own scopes and is the oracle's input, not the model's.
  flat = map_esmfold2_to_af3_graph(sd)
  params = {}
  for key, arr in flat.items():
    scope, name = key.rsplit('/', 1)
    params.setdefault(scope, {})[name] = arr
  return Path(common.write_params_blob(output_dir, 'esmfold2.bin.zst',
                                       params, add_meta=True))


# ---------------------------------------------------------------------------
# AF3-scope mapping: reuse the shared graph, put the work here
# ---------------------------------------------------------------------------
#
# ESMFold2's pair-only block IS an AF3 PairFormerIteration(with_single=False)
# whose two pair attentions are ZERO. Proven, not assumed: filling 20 of the 36
# parameters from the converter and zeroing the other 16 reproduces our
# ESMFold2 pair block at corr 1.00000000 / relerr 4.0e-07. A residual block
# with zero weights contributes exactly zero, the same identity that already
# lets protenix2's dead 4th MSA block ride the shared graph.
#
# So the 48-block trunk, the 4-block lm_encoder, the 2-block coda, the MSA
# encoder's pair stacks and the confidence trunk -- ~95 M of 234.8 M parameters
# -- need no graph code at all, only zero-fill.

_AF3_PAIR_ATTENTION_LEAVES = (
    'pair_attention1/act_norm/offset', 'pair_attention1/act_norm/scale',
    'pair_attention1/gating_query/weights', 'pair_attention1/k_projection/weights',
    'pair_attention1/output_projection/weights', 'pair_attention1/pair_bias_projection/weights',
    'pair_attention1/q_projection/weights', 'pair_attention1/v_projection/weights',
    'pair_attention2/act_norm/offset', 'pair_attention2/act_norm/scale',
    'pair_attention2/gating_query/weights', 'pair_attention2/k_projection/weights',
    'pair_attention2/output_projection/weights', 'pair_attention2/pair_bias_projection/weights',
    'pair_attention2/q_projection/weights', 'pair_attention2/v_projection/weights',
)


def zero_pair_attention(block, c_z, num_head, qkv_dim=None):
  """The 16 leaves AF3's pair block has and ESMFold2's does not, as zeros.

  Shapes follow AF3's GridSelfAttention: q/k/v are (c_z, num_head, qkv_dim),
  the output projection comes back the other way, and the pair bias is
  (c_z, num_head).
  """
  qkv_dim = qkv_dim or c_z // num_head
  # AF3's GridSelfAttention is not uniform in layout: q and k are HEAD-major
  # (num_head, qkv_dim, c_z) while v is CHANNEL-major (c_z, num_head, qkv_dim),
  # and the output projection is a plain (c_z, c_z). qkv_dim is c_z // num_head.
  shapes = {
      'act_norm/offset': (c_z,), 'act_norm/scale': (c_z,),
      'q_projection/weights': (num_head, qkv_dim, c_z),
      'k_projection/weights': (num_head, qkv_dim, c_z),
      'v_projection/weights': (c_z, num_head, qkv_dim),
      'gating_query/weights': (c_z, c_z),
      'output_projection/weights': (c_z, c_z),
      'pair_bias_projection/weights': (c_z, num_head),
  }
  out = dict(block)
  for which in ('pair_attention1', 'pair_attention2'):
    for leaf, shape in shapes.items():
      out['%s/%s' % (which, leaf)] = np.zeros(shape, np.float32)
  return out


AF3_TRUNK = 'diffuser/evoformer'


def zero_single_track(block, c_z, c_s, num_head=16, qkv_dim=None):
  """The single-track leaves an AF3 pairformer has and a pair-only model lacks."""
  qkv_dim = qkv_dim or c_s // num_head
  out = dict(block)
  z = lambda *shape: np.zeros(shape, np.float32)
  out.update({
      'single_attention_layer_norm/scale': z(c_s),
      'single_attention_layer_norm/offset': z(c_s),
      'single_attention_q_projection/weights': z(c_s, num_head, qkv_dim),
      'single_attention_q_projection/bias': z(num_head, qkv_dim),
      'single_attention_k_projection/weights': z(c_s, num_head, qkv_dim),
      'single_attention_v_projection/weights': z(c_s, num_head, qkv_dim),
      'single_attention_gating_query/weights': z(c_s, c_s),
      'single_attention_transition2/weights': z(c_s, c_s),
      'single_pair_logits_norm/scale': z(c_z),
      'single_pair_logits_norm/offset': z(c_z),
      'single_pair_logits_projection/weights': z(c_z, num_head),
      'single_transition/input_layer_norm/scale': z(c_s),
      'single_transition/input_layer_norm/offset': z(c_s),
      'single_transition/transition1/weights': z(c_s, 8 * c_s),
      'single_transition/transition2/weights': z(4 * c_s, c_s),
  })
  return out


def af3_pair_stack(sd, prefix, n_blocks, c_z, num_head, scope):
  """One ESMFold2 pair-only stack as an AF3 pairformer stack, attention zeroed."""
  stacked = stack_blocks(
      lambda i: zero_pair_attention(
          pair_only_block(sd, '%s.blocks.%d' % (prefix, i)), c_z, num_head),
      n_blocks)
  return nest(scope, stacked)


def map_esmfold2_af3(sd, dims=None, num_head=4):
  """ESMFold2 -> the SHARED AF3 graph.

  The pair stacks need no graph code: an AF3 PairFormerIteration(with_single=
  False) whose two pair attentions are zero IS ESMFold2's pair-only block
  (corr 1.00000000, relerr 4.5e-07). The recycle step is the one trunk
  divergence and rides model_config.SSM_RECYCLE:

    prev_embedding            <- parcae_b      (b, folded from log_delta @ b_cont)
    prev_embedding_layer_norm <- parcae_input_norm
    recycle_decay             <- parcae a      (exp(-softplus(log_delta) * exp(log_a)))
  """
  dims = dims or derive_dims(sd)
  c_z = dims['c_z']
  a_vec, b_T = parcae_dynamics(sd)
  p = {
      'left_single/weights': t(sd['z_init_1.weight']),
      'right_single/weights': t(sd['z_init_2.weight']),
      'prev_embedding/weights': b_T,
      'prev_embedding_layer_norm/scale': _arr(sd['parcae_input_norm.weight']),
      'prev_embedding_layer_norm/offset': _arr(sd['parcae_input_norm.bias']),
      'recycle_decay': a_vec,
  }
  p.update(af3_pair_stack(sd, 'folding_trunk', dims['n_trunk'], c_z, num_head,
                          '__layer_stack_no_per_layer_1/trunk_pairformer'))
  return {AF3_TRUNK + '/' + k if '/' in k else AF3_TRUNK + '/' + k: v
          for k, v in p.items()}


def af3_msa_block(sd, prefix, c_z, num_head, is_final, c_m, msa_heads=8,
                  c_hidden=32, msa_head_width=None):
  """One ESMFold2 MSA block as an AF3 EvoformerIteration.

  The leaf names line up without moving anything:
      norm_single      -> msa_attention1/act_norm
      compute_bias.0/1 -> msa_attention1/pair_norm, pair_logits
      Wv / Wgate / Wout-> v_projection, gating_query, output_projection
      OPM norm/W/Wout  -> layer_norm_input, left+right_projection, output_w/b
  Two zero-fills: the pair attentions (ESMFold2 has none), and on the LAST block
  the whole MSA update -- protenix's MSABlock builds msa_stack only
  `if not is_last_block`, and ESMFold2's final block is the same shape. Zero is
  exactly equivalent to skipping for a residual.
  """
  # AF3 states the MSA value width as c_m // num_head rather than carrying it in
  # the config, and v_projection is (c_m, num_head, value_dim).
  msa_head_width = msa_head_width or c_m // msa_heads
  d = DIALECT_ESMFOLD2
  out = {}
  g = lambda leaf: sd['%s.outer_product_mean.%s' % (prefix, leaf)]
  w = _arr(g('W.weight'))
  h = w.shape[0] // 2
  out['outer_product_mean/layer_norm_input/scale'] = _arr(g('norm.weight'))
  out['outer_product_mean/layer_norm_input/offset'] = _arr(g('norm.bias'))
  out['outer_product_mean/left_projection/weights'] = t(w[:h])
  out['outer_product_mean/right_projection/weights'] = t(w[h:])
  # AF3 keeps the OPM output as (c_hidden, c_hidden, c_z) rather than a flat
  # (c_hidden^2, c_z) linear -- the opm_out_direct convention.
  out['outer_product_mean/output_w'] = _arr(g('Wout.weight')).T.reshape(
      c_hidden, c_hidden, c_z)
  out['outer_product_mean/output_b'] = _arr(g('Wout.bias'))

  if is_final:
    dh = msa_head_width
    out['msa_attention1/act_norm/scale'] = np.zeros((c_m,), np.float32)
    out['msa_attention1/act_norm/offset'] = np.zeros((c_m,), np.float32)
    out['msa_attention1/pair_norm/scale'] = np.zeros((c_z,), np.float32)
    out['msa_attention1/pair_norm/offset'] = np.zeros((c_z,), np.float32)
    out['msa_attention1/pair_logits/weights'] = np.zeros((c_z, msa_heads), np.float32)
    out['msa_attention1/v_projection/weights'] = np.zeros((c_m, msa_heads, dh), np.float32)
    out['msa_attention1/gating_query/weights'] = np.zeros((c_m, c_m), np.float32)
    out['msa_attention1/output_projection/weights'] = np.zeros((c_m, c_m), np.float32)
    for leaf, shape in (('input_layer_norm/scale', (c_m,)),
                        ('input_layer_norm/offset', (c_m,)),
                        ('transition1/weights', (c_m, 8 * c_m)),
                        ('transition2/weights', (4 * c_m, c_m))):
      out['msa_transition/%s' % leaf] = np.zeros(shape, np.float32)
  else:
    m = lambda leaf: sd['%s.msa_pair_weighted_averaging.%s' % (prefix, leaf)]
    out['msa_attention1/act_norm/scale'] = _arr(m('norm_single.weight'))
    out['msa_attention1/act_norm/offset'] = _arr(m('norm_single.bias'))
    out['msa_attention1/pair_norm/scale'] = _arr(m('compute_bias.0.weight'))
    out['msa_attention1/pair_norm/offset'] = _arr(m('compute_bias.0.bias'))
    out['msa_attention1/pair_logits/weights'] = t(m('compute_bias.1.weight'))
    out['msa_attention1/v_projection/weights'] = _arr(m('Wv.weight')).T.reshape(
        c_m, msa_heads, msa_head_width)
    out['msa_attention1/gating_query/weights'] = t(m('Wgate.weight'))
    out['msa_attention1/output_projection/weights'] = t(m('Wout.weight'))
    out.update(nest('msa_transition',
                    common.transition(sd, prefix + '.msa_transition', d)))
  out.update(zero_pair_attention(pair_only_block(sd, prefix), c_z, num_head))
  return out


def af3_diffusion_block(sd, prefix_attn, prefix_trans, c_token, num_head):
  """One ESMFold2 diffusion block as one AF3 diffusion-transformer block.

  adaLN and the gates line up directly:
      adaln.s_scale       -> single_cond_layer_norm/scale
      adaln.s_gate  (w,b) -> single_cond_scale/{weights,bias}
      adaln.s_shift.w     -> single_cond_bias/weights
      q_proj (w,b)        -> q_projection/{weights,bias}, reshaped to (H, D)
      kv_proj             -> k_projection + v_projection (the fused half split)
      g_proj              -> gating_query
      out_proj            -> transition2          (the ATTENTION output)
      out_gate (w,b)      -> adaptive_zero_cond/{weights,bias}
      lin_swish / lin_out -> ffw_transition1 / ffw_transition2
      output_gate (w,b)   -> ffw_adaptive_zero_cond/{weights,bias}

  The PER-BLOCK pair LayerNorm is folded, not carried: AF3 shares one
  pair_input_layer_norm across the stack, so the block's own LN scale multiplies
  into pair_logits_projection and its offset is dropped -- the offset enters
  every logit of a row equally and cancels in the softmax. The shared LN is then
  set to identity. Same treatment boltz2 and the protenix family already get.
  """
  a = lambda leaf: sd['%s.%s' % (prefix_attn, leaf)]
  tr = lambda leaf: sd['%s.%s' % (prefix_trans, leaf)]
  d_head = c_token // num_head
  kv = _arr(a('kv_proj.weight'))                       # (2*c, c)
  k_w, v_w = kv[:c_token], kv[c_token:]
  out = {
      'single_cond_layer_norm/scale': _arr(a('adaln.s_scale')),
      'single_cond_scale/weights': t(a('adaln.s_gate.weight')),
      'single_cond_scale/bias': _arr(a('adaln.s_gate.bias')),
      'single_cond_bias/weights': t(a('adaln.s_shift.weight')),
      'q_projection/weights': t(a('q_proj.weight')).reshape(c_token, num_head, d_head),
      'q_projection/bias': _arr(a('q_proj.bias')).reshape(num_head, d_head),
      'k_projection/weights': t(k_w).reshape(c_token, num_head, d_head),
      'v_projection/weights': t(v_w).reshape(c_token, num_head, d_head),
      'gating_query/weights': t(a('g_proj.weight')),
      'transition2/weights': t(a('out_proj.weight')),
      'adaptive_zero_cond/weights': t(a('out_gate.weight')),
      'adaptive_zero_cond/bias': _arr(a('out_gate.bias')),
      'ffw_single_cond_layer_norm/scale': _arr(tr('adaln.s_scale')),
      'ffw_single_cond_scale/weights': t(tr('adaln.s_gate.weight')),
      'ffw_single_cond_scale/bias': _arr(tr('adaln.s_gate.bias')),
      'ffw_single_cond_bias/weights': t(tr('adaln.s_shift.weight')),
      'ffw_transition1/weights': t(tr('lin_swish.weight')),
      'ffw_transition2/weights': t(tr('lin_out.weight')),
      'ffw_adaptive_zero_cond/weights': t(tr('output_gate.weight')),
      'ffw_adaptive_zero_cond/bias': _arr(tr('output_gate.bias')),
  }
  # per-block pair LN scale folded into the pair bias; offset dropped (cancels)
  pair_scale = _arr(a('pair_norm.weight'))
  out['__pair_logits'] = t(a('pair_bias_proj.weight')) * pair_scale[:, None]
  return out


def af3_diffusion_transformer(sd, prefix, n_blocks, n_super, c_token, num_head, c_z):
  """The 12 ESMFold2 token blocks as AF3's nested (n_super, inner) stack."""
  blocks = [af3_diffusion_block(sd, '%s.attn_blocks.%d' % (prefix, i),
                                '%s.transition_blocks.%d' % (prefix, i),
                                c_token, num_head)
            for i in range(n_blocks)]
  inner = n_blocks // n_super
  out = {}
  for k in blocks[0]:
    if k == '__pair_logits':
      continue
    flat = np.stack([b[k] for b in blocks], 0)
    out[k] = flat.reshape((n_super, inner) + flat.shape[1:])
  # pair_logits_projection is (n_super, c_z, inner, num_head): the SUPER axis
  # leads and the inner axis sits between the channel and head axes, not beside
  # the super axis.
  pl = np.stack([b['__pair_logits'] for b in blocks], 0)          # (n, c_z, H)
  pl = pl.reshape(n_super, inner, c_z, num_head).transpose(0, 2, 1, 3)
  return out, pl


def af3_diffusion_conditioning(sd, prefix):
  """ESMFold2's DiffusionConditioning onto AF3's diffusion-head scopes.

  Every piece has a home already; the only reason this needs anything from the
  graph is three LayerNorm OFFSETS, which AF3 creates only for the models that
  have them (boltz2, rosettafold3, chai1, and now esmfold2).

      z_input_norm / z_proj   -> pair_cond_initial_norm / _projection
      z_transitions x2        -> pair_transition_{0,1}
      s_input_norm / s_proj   -> single_cond_initial_norm / _projection
      s_transitions x2        -> single_cond transitions
      fourier.w / .b          -> fourier_embedding_weight / _bias (trained_fourier)
      noise_norm / noise_proj -> noise_embedding_initial_norm / _projection

  AF3's noise embedding is already ESMFold2's: (1/4)*log(sigma/sigma_data), then
  t*w + b, then the cosine -- the trained_fourier path that of3 and if2 use.
  """
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  d = DIALECT_ESMFOLD2_DIFF
  out = {
      'pair_cond_initial_norm/scale': _arr(g('z_input_norm.weight')),
      'pair_cond_initial_norm/offset': _arr(g('z_input_norm.bias')),
      'pair_cond_initial_projection/weights': t(g('z_proj.weight')),
      # s_inputs enters the diffusion head REORDERED but not narrowed: this
      # LayerNorm normalises over the width, so the four dead restype columns
      # have to stay (the graph pads them back in). Elsewhere in the port they
      # are dropped, because a zero column is free through a bias-free Linear.
      'single_cond_initial_norm/scale': _permute_vec(_arr(g('s_input_norm.weight'))),
      'single_cond_initial_norm/offset': _permute_vec(_arr(g('s_input_norm.bias'))),
      'single_cond_initial_projection/weights': permute_s_inputs(t(g('s_proj.weight'))),
      'fourier_embedding_weight': _arr(g('fourier.w')),
      'fourier_embedding_bias': _arr(g('fourier.b')),
      'noise_embedding_initial_norm/scale': _arr(g('noise_norm.weight')),
      'noise_embedding_initial_norm/offset': _arr(g('noise_norm.bias')),
      'noise_embedding_initial_projection/weights': t(g('noise_proj.weight')),
  }
  # AF3's transition_block concatenates its own name onto the leaf with NO
  # separator: `pair_transition_0ffw_layer_norm/scale`, not
  # `pair_transition_0/ffw_layer_norm/scale`. Same for the ffw_transitions.
  for i in range(2):
    for tag, src in (('pair_transition_%d' % i, 'z_transitions.%d' % i),
                     ('single_transition_%d' % i, 's_transitions.%d' % i)):
      tr = common.transition(sd, '%s.%s' % (prefix, src), d)
      out['%sffw_layer_norm/scale' % tag] = tr['input_layer_norm/scale']
      out['%sffw_layer_norm/offset' % tag] = tr['input_layer_norm/offset']
      out['%sffw_transition1/weights' % tag] = tr['transition1/weights']
      out['%sffw_transition2/weights' % tag] = tr['transition2/weights']
  return out


# ---------------------------------------------------------------------------
# full AF3-scope conversion
# ---------------------------------------------------------------------------

ESM_RESTYPE_OFFSET = 2          # ESMFold2 reserves classes 0 and 1


def _remap_restype_block(w, n_af3):
  """ESMFold2's 33-class residue block -> AF3's n_af3 classes.

  Both order residues the same way -- ALA, ARG, ASN, ASP, ... , VAL, UNK, then
  nucleic -- but ESMFold2 reserves two leading classes, so the whole block is
  shifted by two and the remap is a SLICE rather than a permutation. Verified on
  6MRR, whose res_type values run 3..21, i.e. inside the shifted protein range.
  """
  return w[ESM_RESTYPE_OFFSET:ESM_RESTYPE_OFFSET + n_af3]


def _remap_vec(v):
  """The s_inputs remap for a 1-D tensor (a LayerNorm scale or offset)."""
  return remap_s_inputs(v[:, None])[:, 0]


def _permute_vec(v):
  """The padded (451-wide) s_inputs reorder for a 1-D tensor."""
  return permute_s_inputs(v[:, None])[:, 0]


def remap_s_inputs(w, c_atom=384, n_esm=33, n_af3=31):
  """(451, out) -> (447, out): ESMFold2's s_inputs rows in AF3's order.

  The two models concatenate the SAME four blocks in different orders:

      ESMFold2   [atom 384 | restype 33 | profile 33 | deletion 1]  = 451
      AF3        [restype 31 | profile 31 | deletion 1 | atom 384]  = 447

  (model.create_target_feat_embedding: `concatenate([target_feat, token_act])`,
  and create_target_feat puts restype, profile, deletion in that order.) So this
  is a PERMUTATION as well as a slice -- reading it as a slice alone leaves
  every consumer multiplying the atom block by the restype weights, which is
  not a small error: z_init came out at corr 0.008 against the reference.
  """
  atom = w[:c_atom]
  rest = _remap_restype_block(w[c_atom:c_atom + n_esm], n_af3)
  prof = _remap_restype_block(w[c_atom + n_esm:c_atom + 2 * n_esm], n_af3)
  tail = w[c_atom + 2 * n_esm:]
  return np.concatenate([rest, prof, tail, atom], axis=0)


def permute_s_inputs(w, c_atom=384, n_esm=33):
  """(451, out) -> (451, out): AF3's block ORDER, ESM's block WIDTHS.

  For the diffusion conditioner the four dead restype columns cannot be dropped:
  its projection sits behind a LayerNorm, which normalises over the width, so
  447 channels is not the same function as 451 with four zeros. The graph pads
  the feature vector back to 451; these rows have to be in the padded order.
  """
  atom = w[:c_atom]
  rest = w[c_atom:c_atom + n_esm]
  prof = w[c_atom + n_esm:c_atom + 2 * n_esm]
  tail = w[c_atom + 2 * n_esm:]
  return np.concatenate([rest, prof, tail, atom], axis=0)


def remap_msa_feat(w, n_esm=33, gap_index=21):
  """(35, out) -> (34, out): [one-hot | has_deletion | deletion_value].

  Unlike s_inputs this is NOT a plain slice. AF3's MSA alphabet is
  POLYMER_TYPES_ORDER_WITH_ALL_UNKS_AND_GAP -- 32 classes with a GAP at index 21,
  between UNK and the nucleic acids -- and ESMFold2 has no gap class of its own.
  So the shifted block supplies AF3's 0..20 and 22..31, and index 21 is a ZERO
  row: a gap contributes nothing rather than aliasing onto a residue.

  Getting this wrong would not raise -- 33 columns land in a 34-wide slot only if
  something is inserted -- but every nucleic class would sit one slot low, which
  is exactly the rf3 G/C swap that broke RNA while every protein gate passed.
  """
  block = w[ESM_RESTYPE_OFFSET:n_esm]                       # AF3 0..20, then 22..
  head, tail = block[:gap_index], block[gap_index:]
  gap = np.zeros((1,) + w.shape[1:], w.dtype)
  return np.concatenate([head, gap, tail, w[n_esm:]], axis=0)


def af3_atom_block(sd, prefix, c_atom, num_head):
  """One SWAAtomBlock as one AF3 atom-transformer block.

  ESMFold2 fuses the whole modulation into a single Linear to 6*c:
      shift_a, scale_a, gate_a, shift_f, scale_f, gate_f
  AF3 keeps six separate projections, so the fused weight is split six ways.
  Note the ORDER -- shift before scale -- and that gate_a/gate_f are the
  adaptive_zero_cond output gates, not part of the adaLN.
  """
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  mod = t(g('adaln_modulation.1.weight'))                 # (c, 6c)
  sh_a, sc_a, g_a, sh_f, sc_f, g_f = np.split(mod, 6, axis=1)
  d_head = c_atom // num_head
  return {
      # cross_attention adaLNs the queries and the keys SEPARATELY, each with its
      # own projections; ESMFold2 has one adaln_modulation read by both, so the
      # same weights fill the q and k slots.
      'qsingle_cond_bias/weights': sh_a,
      'qsingle_cond_scale/weights': sc_a,
      'ksingle_cond_bias/weights': sh_a,
      'ksingle_cond_scale/weights': sc_a,
      # no bias: for these blocks the graph takes the gate RAW, so there is no
      # sigmoid to offset and no bias slot to fill
      'adaptive_zero_cond/weights': g_a,
      'ffw_single_cond_bias/weights': sh_f,
      'ffw_single_cond_scale/weights': sc_f,
      'ffw_adaptive_zero_cond/weights': g_f,
      'q_projection/weights': t(g('attn.Wqkv.weight'))[:, :c_atom].reshape(
          c_atom, num_head, d_head),
      'q_projection/bias': np.zeros((num_head, d_head), np.float32),
      'k_projection/weights': t(g('attn.Wqkv.weight'))[:, c_atom:2 * c_atom].reshape(
          c_atom, num_head, d_head),
      'v_projection/weights': t(g('attn.Wqkv.weight'))[:, 2 * c_atom:].reshape(
          c_atom, num_head, d_head),
      'gating_query/weights': t(g('attn.gate_proj.weight')),
      'transition2/weights': t(g('attn.out_proj.weight')),
      'ffw_transition1/weights': t(g('ffn.w_up.weight')),
      'ffw_transition2/weights': t(g('ffn.w_down.weight')),
  }


def map_esmfold2_to_af3_graph(sd, dims=None):
  """ESMFold2 -> the shared AF3 graph's own scopes. The whole port, in one map."""
  dims = dims or derive_dims(sd)
  c_z, c_m = dims['c_z'], dims['c_msa']
  pair_head = c_z // 64                      # AF3 tri-attention heads
  P = 'diffuser/evoformer'
  D = 'diffuser/~/diffusion_head'
  CH = 'diffuser/confidence_head'
  flat = {}

  def put(scope, local):
    for k, v in local.items():
      flat['%s/%s' % (scope, k) if scope else k] = v

  # ── trunk inputs, recycling, distogram ────────────────────────────────────
  a_vec, b_T = parcae_dynamics(sd)
  put(P, {
      'left_single/weights': remap_s_inputs(t(sd['z_init_1.weight'])),
      'right_single/weights': remap_s_inputs(t(sd['z_init_2.weight'])),
      'bond_embedding/weights': t(sd['token_bonds.weight']),
      'prev_embedding/weights': b_T,
      'prev_embedding_layer_norm/scale': _arr(sd['parcae_input_norm.weight']),
      'prev_embedding_layer_norm/offset': _arr(sd['parcae_input_norm.bias']),
      'recycle_decay': a_vec,
      '~_relative_encoding/position_activations/weights': t(sd['rel_pos.embed.weight']),
      'msa_activations/weights': remap_msa_feat(t(sd['msa_encoder.embed.weight'])),
      'extra_msa_target_feat/weights': remap_s_inputs(
          t(sd['msa_encoder.project_inputs.weight'])),
  })
  # ESMFold2 symmetrises BEFORE the head -- distogram_head(z + z^T) -- while AF3
  # computes half_logits(z) and then half + half^T. For the WEIGHTS those agree
  # (W(z + z^T) == Wz + (Wz)^T), but the BIAS is added once natively and twice
  # here, so it is HALVED. Dropping it entirely, which is what happens without
  # DISTOGRAM_BIAS membership, adds a constant per-bin offset to every logit and
  # flattens the contact map to ~0.96 EVERYWHERE -- the trunk looks destroyed
  # when only the head is wrong.
  flat['diffuser/distogram_head/half_logits/weights'] = t(sd['distogram_head.weight'])
  flat['diffuser/distogram_head/half_logits/bias'] = _arr(sd['distogram_head.bias']) / 2.0

  # ── trunk pairformer and MSA stack ────────────────────────────────────────
  put(P, af3_pair_stack(sd, 'folding_trunk', dims['n_trunk'], c_z, pair_head,
                        '__layer_stack_no_per_layer_1/trunk_pairformer'))
  # the post-trunk readout + coda: AF3 has no such stage, so nothing in the
  # scope diff asked for these until the graph gained them
  put(P, {'parcae_readout/weights': t(sd['parcae_readout.weight'])})
  put(P, af3_pair_stack(sd, 'parcae_coda', dims['n_coda'], c_z, pair_head,
                        '__layer_stack_no_per_layer_2/trunk_coda'))
  n_msa = dims['n_msa']
  msa = stack_blocks(
      lambda i: af3_msa_block(sd, 'msa_encoder.blocks.%d' % i, c_z, pair_head,
                              i == n_msa - 1, c_m), n_msa)
  put(P, nest('__layer_stack_no_per_layer/msa_stack', msa))

  # ── diffusion ─────────────────────────────────────────────────────────────
  put(D, af3_diffusion_conditioning(sd, 'structure_head.diffusion_module.conditioning'))
  n_tok = 12
  stk, pl = af3_diffusion_transformer(
      sd, 'structure_head.diffusion_module.token_transformer', n_tok, 3,
      dims['c_diff'], 16, c_z)
  # AF3 concatenates the stack's module name onto the leaf with NO separator:
  # `...diffusion_transformeradaptive_zero_cond/bias`, not `.../adaptive_zero_cond`.
  # The token stack is named `transformer`, and AF3 concatenates that name onto
  # each leaf with no separator.
  pre = ('%s/transformer/__layer_stack_with_per_layer/'
         '__layer_stack_with_per_layer/transformer' % D)
  for k, v in stk.items():
    flat[pre + k] = v
  flat['%s/transformer/__layer_stack_with_per_layer/pair_logits_projection/weights' % D] = pl
  # DIFFUSION_PROJECTED_RELPOS: ESMFold2 concatenates [z_trunk, PROJECTED relpos],
  # and the projection is the trunk's own rel_pos embedding reused.
  flat['%s/relpe_projection/weights' % D] = t(sd['rel_pos.embed.weight'])

  # ── the single track ESMFold2 does not have ──────────────────────────────
  # PAIR_ONLY_TRUNK removes the pairformer's single update, but AF3 still builds
  # the top-level single embedding and its recycle. Nothing reads them for this
  # model (the diffusion conditions on s_inputs alone), so they are zeros rather
  # than an invented mapping.
  c_s = 384
  put(P, {
      'single_activations/weights': np.zeros((447, c_s), np.float32),
      'prev_single_embedding/weights': np.zeros((c_s, c_s), np.float32),
      'prev_single_embedding_layer_norm/scale': np.ones((c_s,), np.float32),
      'prev_single_embedding_layer_norm/offset': np.zeros((c_s,), np.float32),
  })

  # ── atom encoders: the inputs embedder and the diffusion one ─────────────
  for prefix, name, root, c_out, n_at in (
      ('inputs_embedder.atom_attention_encoder', 'evoformer_conditioning',
       'diffuser', dims['c_diff'] // 2, dims.get('n_input_atom', 3)),
      ('structure_head.diffusion_module.atom_encoder', 'diffusion',
       D, dims['c_diff'], dims['n_diff_atom'])):
    o, stk = af3_atom_encoder(sd, prefix, name, dims['c_atom'], 4, n_at, c_out)
    put(root, o)
    pre_at = '%s/%s_atom_transformer_encoder' % (root, name)
    for k, v in stk.items():
      flat['%s/__layer_stack_with_per_layer/%s_atom_transformer_encoder%s' % (
          pre_at, name, k)] = v
    flat['%s/pair_input_layer_norm/scale' % pre_at] = np.ones((16,), np.float32)
    flat['%s/pair_logits_projection/weights' % pre_at] = np.zeros((16, n_at, 4), np.float32)

  # ── the diffusion atom decoder and the coordinate heads ─────────────────
  SH = 'structure_head.diffusion_module'
  AD = 'structure_head.diffusion_module.atom_decoder'
  dec = [af3_atom_block(sd, '%s.atom_transformer.blocks.%d' % (AD, i),
                        dims['c_atom'], 4) for i in range(dims['n_diff_atom'])]
  pre_dec = '%s/diffusion_atom_transformer_decoder' % D
  for k in dec[0]:
    flat['%s/__layer_stack_with_per_layer/diffusion_atom_transformer_decoder%s' % (
        pre_dec, k)] = np.stack([b[k] for b in dec], 0)
  flat['%s/pair_input_layer_norm/scale' % pre_dec] = np.ones((16,), np.float32)
  flat['%s/pair_logits_projection/weights' % pre_dec] = np.zeros(
      (16, dims['n_diff_atom'], 4), np.float32)
  put(D, {
      # coords_linear takes [r_l | pred_r1]; pred_r1 is zeros on the inference
      # path, so only the first three columns can contribute.
      'diffusion_atom_positions_to_features/weights':
          t(sd['structure_head.diffusion_module.atom_encoder.coords_linear.weight'])[:3],
      'diffusion_atom_features_to_position_update/weights': t(sd[AD + '.output_linear.weight']),
      'diffusion_atom_features_layer_norm/scale': _arr(sd[AD + '.norm.weight']),
      'diffusion_atom_features_layer_norm/offset': _arr(sd[AD + '.norm.bias']),
      'diffusion_project_token_features_for_broadcast/weights':
          t(sd[AD + '.token_to_atom_linear.weight']),
      # no trunk single or atom-pair conditioning in this model
      'diffusion_embed_trunk_single_cond/weights': np.zeros((c_s, dims['c_atom']), np.float32),
      'diffusion_lnorm_trunk_single_cond/scale': np.ones((c_s,), np.float32),
      'diffusion_embed_trunk_pair_cond/weights': np.zeros((c_z, 16), np.float32),
      'diffusion_lnorm_trunk_pair_cond/scale': np.ones((c_z,), np.float32),
      # ESMFold2's token_norm, NOT an identity: the previous mapping asserted
      # `a` needed no output normalisation and dropped both its scale and its
      # bias.
      'output_norm/scale': _arr(sd[SH + '.token_norm.weight']),
      'output_norm/offset': _arr(sd[SH + '.token_norm.bias']),
      # AF3 re-embeds the conditioned single before the token stack, which is
      # exactly ESMFold2's `a += s_to_token(LN(s))` -- s_step_norm carries a
      # BIAS, so this is not the identity the earlier mapping assumed.
      'single_cond_embedding_norm/scale': _arr(sd[SH + '.s_step_norm.weight']),
      'single_cond_embedding_norm/offset': _arr(sd[SH + '.s_step_norm.bias']),
      'single_cond_embedding_projection/weights': t(sd[SH + '.s_to_token.weight']),
  })
  # the token stack's SHARED pair LayerNorm is identity: ESMFold2's per-block
  # pair LN scales are folded into pair_logits_projection instead (its offsets
  # cancel in the softmax), so nothing is left for the shared one to do.
  flat['%s/transformer/pair_input_layer_norm/scale' % D] = np.ones((c_z,), np.float32)

  # ── confidence head ──────────────────────────────────────────────────────
  ch = 'confidence_head'
  n_at_tok = 24                     # AF3 slot count; ESMFold2 trains 23
  def pad_atoms(w):
    return np.concatenate([w, np.zeros((1,) + w.shape[1:], w.dtype)], 0)
  RE = CH + '/~_boltz2_reembed'
  put(RE, {
      's_inputs_norm/scale': _remap_vec(_arr(sd['%s.s_inputs_norm.weight' % ch])),
      's_inputs_norm/offset': _remap_vec(_arr(sd['%s.s_inputs_norm.bias' % ch])),
      's_norm/scale': _arr(sd['%s.s_norm.weight' % ch]),
      's_norm/offset': _arr(sd['%s.s_norm.bias' % ch]),
      's_input_to_s/weights': remap_s_inputs(t(sd['%s.s_input_to_s.weight' % ch])),
      'z_norm/scale': _arr(sd['%s.z_norm.weight' % ch]),
      'z_norm/offset': _arr(sd['%s.z_norm.bias' % ch]),
      's_to_z_prod_in1/weights': remap_s_inputs(t(sd['%s.s_to_z_prod_in1.weight' % ch])),
      's_to_z_prod_in2/weights': remap_s_inputs(t(sd['%s.s_to_z_prod_in2.weight' % ch])),
      's_to_z_prod_out/weights': t(sd['%s.s_to_z_prod_out.weight' % ch]),
      'left_target_feat_project/weights': remap_s_inputs(t(sd['%s.s_to_z.weight' % ch])),
      'right_target_feat_project/weights':
          remap_s_inputs(t(sd['%s.s_to_z_transpose.weight' % ch])),
      'rel_pos_project/weights': t(sd['rel_pos.embed.weight']),
      'token_bonds_project/weights': t(sd['token_bonds.weight'])[:1],
      # ESMFold2's distance-bin EMBEDDING is a Linear on the one-hot, so it maps
      # straight across -- but AF3 allots 64 bins where ESMFold2 trains 39, so the
      # tail is zero: a bin the model never learned contributes nothing.
      'distogram_feat_project/weights': np.concatenate([
          _arr(sd['%s.dist_bin_pairwise_embed.weight' % ch]),
          np.zeros((64 - 39, c_z), np.float32)], 0),
      # boltz2's contact conditioning; ESMFold2 has none
      'contact_encoder/weights': np.zeros((260, c_z), np.float32),
      'contact_encoder/bias': np.zeros((c_z,), np.float32),
      'contact_fourier/weights': np.zeros((1, c_z), np.float32),
      'contact_fourier/bias': np.zeros((c_z,), np.float32),
      'token_bonds_type_embed/weights': np.zeros((7, c_z), np.float32),
  })
  put(CH, {
      'contact_encoding_unselected': np.zeros((c_z,), np.float32),
      'contact_encoding_unspecified': np.zeros((c_z,), np.float32),
      'row_pool_attn/weights': t(sd['%s.row_attention_pooling.attn_proj.weight' % ch]),
      'row_pool_out/weights': t(sd['%s.row_attention_pooling.out_proj.weight' % ch]),
      'pae_logits_ln/scale': _arr(sd['%s.pae_ln.weight' % ch]),
      'pae_logits_ln/offset': _arr(sd['%s.pae_ln.bias' % ch]),
      'pae_logits/weights': t(sd['%s.pae_head.weight' % ch]),
      'logits_ln/scale': _arr(sd['%s.pde_ln.weight' % ch]),
      'logits_ln/offset': _arr(sd['%s.pde_ln.bias' % ch]),
      'left_half_distance_logits/weights': t(sd['%s.pde_head.weight' % ch]),
      'plddt_logits_ln/scale': _arr(sd['%s.plddt_ln.weight' % ch]),
      'plddt_logits_ln/offset': _arr(sd['%s.plddt_ln.bias' % ch]),
      'plddt_logits/weights': pad_atoms(_arr(sd['%s.plddt_weight' % ch])).transpose(1, 0, 2),
      'experimentally_resolved_ln/scale': _arr(sd['%s.resolved_ln.weight' % ch]),
      'experimentally_resolved_ln/offset': _arr(sd['%s.resolved_ln.bias' % ch]),
      'experimentally_resolved_logits/weights':
          pad_atoms(_arr(sd['%s.resolved_weight' % ch])).transpose(1, 0, 2),
  })
  # AF3's confidence pairformer carries a single track; ESMFold2's 4-block
  # confidence trunk is pair-only, so the single leaves are zeros -- a residual
  # with zero weights contributes exactly zero, the same identity the trunk uses.
  conf = stack_blocks(
      lambda i: zero_single_track(
          zero_pair_attention(pair_only_block(sd, '%s.folding_trunk.blocks.%d' % (ch, i)),
                              c_z, pair_head), c_z, c_s),
      dims['n_conf'])
  put(CH, nest('__layer_stack_no_per_layer/confidence_pairformer', conf))
  return flat


def af3_atom_encoder(sd, prefix, name, c_atom, num_head, n_blocks, c_out,
                     pair_channels=16):
  """ESMFold2's atom encoder onto AF3's per-atom conditioning scopes.

  The fused atom_linear (c_atom, 389) SPLITS into AF3's five bias-free
  per-feature embedders -- a Linear over a concatenation IS the sum of its
  column blocks, so this is a slice, not an approximation:

      [ref_pos 3 | charge 1 | mask 1 | element 128 | atom_name 256]

  Everything ESMFold2 does not have is zero-filled: the atom PAIR conditioning
  (offsets, distances, the three pair MLPs, the single->pair projections) and
  the attention's pair bias, because its atom attention is positional through
  rotary alone.

  Queries and keys get the SAME adaLN weights: cross_attention conditions each
  side separately, while ESMFold2 has one adaln_modulation read by both.
  """
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  w = t(g('atom_linear.weight'))                        # (389, c_atom)
  o = 0
  out = {}
  for leaf, width in (('embed_ref_pos', 3), ('embed_ref_charge', 1),
                      ('embed_ref_mask', 1), ('embed_ref_element', 128),
                      ('embed_ref_atom_name', 256)):
    out['%s_%s/weights' % (name, leaf)] = w[o:o + width]
    o += width
  assert o == w.shape[0], (o, w.shape)
  out['%s_atom_features_norm/scale' % name] = _arr(g('atom_norm.weight'))
  out['%s_atom_features_norm/offset' % name] = _arr(g('atom_norm.bias'))
  out['%s_project_atom_features_for_aggr/weights' % name] = t(g('atom_to_token_linear.weight'))
  for leaf, shape in (('embed_pair_offsets', (3, pair_channels)),
                      ('embed_pair_offsets_1', (3, pair_channels)),
                      ('embed_pair_distances', (1, pair_channels)),
                      ('embed_pair_distances_1', (1, pair_channels)),
                      ('embed_pair_offsets_valid', (1, pair_channels)),
                      ('pair_mlp_1', (pair_channels, pair_channels)),
                      ('pair_mlp_2', (pair_channels, pair_channels)),
                      ('pair_mlp_3', (pair_channels, pair_channels)),
                      ('single_to_pair_cond_row', (c_atom, pair_channels)),
                      ('single_to_pair_cond_row_1', (c_atom, pair_channels)),
                      ('single_to_pair_cond_col', (c_atom, pair_channels)),
                      ('single_to_pair_cond_col_1', (c_atom, pair_channels))):
    out['%s_%s/weights' % (name, leaf)] = np.zeros(shape, np.float32)

  blocks = [af3_atom_block(sd, '%s.atom_transformer.blocks.%d' % (prefix, i),
                           c_atom, num_head) for i in range(n_blocks)]
  stacked = {}
  for k in blocks[0]:
    stacked[k] = np.stack([b[k] for b in blocks], 0)
  stack_name = '%s_atom_transformer_%s' % (name, 'encoder' if 'encoder' in name or True else '')
  return out, stacked
