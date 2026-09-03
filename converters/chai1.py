"""chai-1 (chaidiscovery) -> AF3-haiku weights.

chai is the first family we port that publishes NO model source: the five
`models_v2/*.pt` archives are frozen, inlined TorchScript. Submodules there hold
parameters but have no forward methods, so nothing can be read off a class
definition and nothing can be hooked. Everything below was derived from two
places: the parameter shapes in `state_dict()`, and the traced graph itself
(`torch.jit.load(...).forward_256.code`), which is the authoritative spec.

Three of chai's primitives look, at first, like they need their own modules.
Two of them do not, because the algebra distributes:

  * The merged bidirectional triangle multiplication computes BOTH directions
    from one LayerNorm, sums the two products, and applies ONE output projection
    and ONE out-gate. Since `gate * W(x1 + x2) == gate*W(x1) + gate*W(x2)`, it
    maps onto AF3's two separate tri-mul scopes by copying the shared output
    projection and gate into both. chai's product norm is affine-free, so
    `center_norm` gets scale=1 / offset=0.

  * The fused two-direction triangle attention holds one qkvg projection per
    direction, so each maps to one of AF3's `pair_attention1/2`. Its
    `out_scalers` scales the ROWS of the shared output linear
    (`W * out_scalers[:, None]` in the trace), so it folds into the weight
    before the (256, 512) is split column-wise into the two directions.

What does NOT map is the outer product mean: chai's `weight_ab (2, 8, 8, 64)`
is EIGHT GROUPS of an 8x8 outer product (the group index is broadcast, not
contracted -- only the MSA-depth axis is summed), giving 8*8*8 = 512 channels
where AF3's single group gives 8*8 = 64. That one needs a chai-gated module.

See memory chai1-port.md for the full recon.
"""

from __future__ import annotations

import os

import numpy as np

from . import common as C

# AlphaFold 3's diffusion sigma_data (see model/network/diffusion_head.py).
SIGMA_DATA = 16.0

# chai's fused qkvg projections stack the four roles in this order (the trace
# unbinds them as `q, k, v, g = ...`).
_QKVG = ('q', 'k', 'v', 'g')

TRUNK_PAIRFORMER_BLOCKS = 48
TRI_ATTN_HEADS = 4          # trunk/msa/confidence tri-attention, head dim c_z//4
SINGLE_ATTN_HEADS = 16      # attention pair bias, head dim 24


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_chai1(model_dir):
  """Load chai's TorchScript archives -> {component: state_dict}.

  `model_dir` is a directory (not a file) because chai publishes five separate
  archives; see the `files` source form in runner.WEIGHTS.
  """
  import torch
  model_dir = os.path.expanduser(str(model_dir))
  base = os.path.join(model_dir, 'models_v2')
  if not os.path.isdir(base):
    base = model_dir
  out = {}
  for name in ('trunk', 'token_embedder', 'diffusion_module', 'confidence_head',
               'feature_embedding'):
    path = os.path.join(base, name + '.pt')
    out[name] = torch.jit.load(path, map_location='cpu').state_dict()
  dgram = os.path.join(base, 'distogram_head.pt')
  if os.path.isfile(dgram):
    # not a TorchScript archive -- a plain state_dict from sokrypton/chai-lab@dgram
    out['distogram_head'] = torch.load(dgram, map_location='cpu', weights_only=True)
  return out


# ---------------------------------------------------------------------------
# the fused-projection splits
# ---------------------------------------------------------------------------

def _split_qkvg_grid(w, heads):
  """chai tri-attention `pair2qkvg{1,2}` (H*4*D, in) -> {role: (H, D, in)}.

  The trace reshapes this to (..., H, 4, D) before permuting the role axis to
  the front, so the rows run head-major with the role axis SECOND -- not four
  contiguous q|k|v|g blocks.
  """
  w = C._arr(w)
  rows, c_in = w.shape
  d = rows // (heads * 4)
  w = w.reshape(heads, 4, d, c_in)
  return {role: w[:, i] for i, role in enumerate(_QKVG)}


def _split_qkvg_single(w):
  """chai `input2qkvg` (in, 4, H, D) -> {role: (in, H, D)}. Already unfused."""
  w = C._arr(w)
  return {role: w[:, i] for i, role in enumerate(_QKVG)}


def _tri_attention(sd, prefix, direction, heads, out_weight):
  """One direction of chai's fused triangle attention -> GridSelfAttention params.

  `out_weight` is the already-scaled, already-split (out, in) slice of
  `linear_out` belonging to this direction. chai's act_norm is affine-free, so
  the scale/offset are synthesised rather than read.
  """
  qkvg = _split_qkvg_grid(sd[f'{prefix}.pair2qkvg{direction + 1}.weight'], heads)
  bias = C._arr(sd[f'{prefix}.pair2b.weight'])          # (2*heads, c_z)
  c_z = bias.shape[1]
  return {
      'act_norm/scale': np.ones(c_z, np.float32),
      'act_norm/offset': np.zeros(c_z, np.float32),
      'pair_bias_projection/weights': bias[direction * heads:(direction + 1) * heads].T,
      'q_projection/weights': qkvg['q'],
      'k_projection/weights': qkvg['k'],
      'v_projection/weights': np.ascontiguousarray(
          qkvg['v'].reshape(-1, qkvg['v'].shape[-1]).T).reshape(-1, heads, qkvg['v'].shape[1]),
      'gating_query/weights': qkvg['g'].reshape(-1, qkvg['g'].shape[-1]),
      'output_projection/weights': out_weight.T,
  }


def _tri_attention_pair(sd, prefix, heads=TRI_ATTN_HEADS):
  """Both directions -> {'pair_attention1/...': v, 'pair_attention2/...': v}."""
  # out_scalers scales the rows of linear_out; fold it in, then split the input
  # axis, which is the concatenation [direction 1 | direction 2].
  w = C._arr(sd[f'{prefix}.linear_out.weight'])          # (c_z, 2*c_z)
  if f'{prefix}.out_scalers' in sd:
    w = w * C._arr(sd[f'{prefix}.out_scalers'])[:, None]
  half = w.shape[1] // 2
  out = {}
  for d, (tag, sl) in enumerate([('pair_attention1', slice(0, half)),
                                 ('pair_attention2', slice(half, None))]):
    for k, v in _tri_attention(sd, prefix, d, heads, w[:, sl]).items():
      out[f'{tag}/{k}'] = v
  return out


# ---------------------------------------------------------------------------
# merged bidirectional triangle multiplication
# ---------------------------------------------------------------------------

def _tri_attention_joint(sd, prefix, heads=TRI_ATTN_HEADS):
  """chai's CONFIDENCE triangle attention -> two dual-output GridSelfAttentions.

  Differs from the trunk's in three ways: q/k/v/g/bias for both directions are
  fused into one `pair2qkvgb`, there is no `out_scalers`, and `linear_out` is
  (2*c_z, 2*c_z) rather than (c_z, 2*c_z).

  That last one is the whole reason this needs its own recipe. The trunk's
  single shared output is `W @ [a1; a2]`, which splits along the INPUT axis into
  AF3's two per-direction projections that then sum. Here the output axis splits
  too: rows 0..c_z are kept and rows c_z..2*c_z are transposed, and each reads
  both directions. So the four blocks of W are four projections -- each
  direction gets a kept one and a transposed one -- which is what
  GridSelfAttention.dual_output provides. Both axes are direction-major
  (`d*c_z + head*dim + channel`), read off chai's own node-849 reshape to
  (..., 2, heads, dim).
  """
  qkvgb = C._arr(sd[f'{prefix}.pair2qkvgb.weight'])      # (2*4*c_z + 2*heads, c_z)
  c_z = qkvgb.shape[1]
  per = 4 * c_z                                          # q|k|v|g for one direction
  bias = qkvgb[2 * per:]                                 # (2*heads, c_z)
  w = C._arr(sd[f'{prefix}.linear_out.weight'])          # (2*c_z, 2*c_z)
  # this LayerNorm IS affine here, unlike the trunk's
  scale = C._arr(sd[f'{prefix}.pair_layer_norm.weight'])
  offset = C._arr(sd[f'{prefix}.pair_layer_norm.bias'])

  out = {}
  for d, tag in enumerate(('pair_attention1', 'pair_attention2')):
    # ROLE-MAJOR (4, heads, dim), NOT the trunk's head-major (heads, 4, dim).
    # The trunk holds a separate pair2qkvg per direction; this one fuses both
    # into pair2qkvgb, and the fused tensor orders roles outermost. Reading it
    # head-major does not error and does not look broken -- it silently yields
    # an attention that contributes almost nothing (block-0 pair 0.9832, and
    # zeroing the whole triangle attention scored 0.9839, i.e. BETTER). Getting
    # it right takes the same block to 0.9964.
    blk = qkvgb[d * per:(d + 1) * per]
    w4 = blk.reshape(4, heads, blk.shape[0] // (4 * heads), blk.shape[1])
    qkvg = {r: w4[i] for i, r in enumerate(_QKVG)}
    v = qkvg['v']
    out.update({
        f'{tag}/act_norm/scale': scale,
        f'{tag}/act_norm/offset': offset,
        f'{tag}/pair_bias_projection/weights':
            bias[d * heads:(d + 1) * heads].T,
        f'{tag}/q_projection/weights': qkvg['q'],
        f'{tag}/k_projection/weights': qkvg['k'],
        f'{tag}/v_projection/weights': np.ascontiguousarray(
            v.reshape(-1, v.shape[-1]).T).reshape(-1, heads, v.shape[1]),
        f'{tag}/gating_query/weights': qkvg['g'].reshape(-1, qkvg['g'].shape[-1]),
        # rows = which output half, columns = which direction's heads
        # input axis is direction-major too: swapping these halves costs pde
        # 0.997 -> 0.968 and pushes both rms ratios to 1.11/1.19.
        f'{tag}/output_projection/weights': w[:c_z, d * c_z:(d + 1) * c_z].T,
        f'{tag}/output_projection_transposed/weights':
            w[c_z:, d * c_z:(d + 1) * c_z].T,
    })
  return out


def _triangle_mul_pair(sd, prefix):
  """chai's ONE merged tri-mul -> AF3's outgoing + incoming scopes.

  `merged_linear_p` (4c, c) holds [a_out | b_out | a_in | b_in] and
  `merged_linear_g` (5c, c) the matching four gates plus the shared out-gate as
  its last c rows. The shared `linear_z_out` and out-gate go into BOTH scopes:
  each direction then applies them to its own product and the residuals sum to
  what chai computes in one step.
  """
  p = C._arr(sd[f'{prefix}.merged_linear_p.weight'])
  g = C._arr(sd[f'{prefix}.merged_linear_g.weight'])
  c = p.shape[0] // 4
  out_gate = g[4 * c:]
  z_out = C._arr(sd[f'{prefix}.linear_z_out.weight'])
  ln_w = C._arr(sd[f'{prefix}.layernorm_z_in.weight'])
  ln_b = C._arr(sd[f'{prefix}.layernorm_z_in.bias'])

  out = {}
  for tag, base, outgoing in [('triangle_multiplication_outgoing', 0, True),
                              ('triangle_multiplication_incoming', 2 * c, False)]:
    a_p, b_p = p[base:base + c], p[base + c:base + 2 * c]
    a_g, b_g = g[base:base + c], g[base + c:base + 2 * c]
    if not outgoing:                       # incoming swaps the a/b roles
      a_p, b_p = b_p, a_p
      a_g, b_g = b_g, a_g
    out.update({
        f'{tag}/left_norm_input/scale': ln_w,
        f'{tag}/left_norm_input/offset': ln_b,
        # chai's product norm is affine-free
        f'{tag}/center_norm/scale': np.ones(c, np.float32),
        f'{tag}/center_norm/offset': np.zeros(c, np.float32),
        f'{tag}/projection/weights': C.interleave(a_p, b_p),
        f'{tag}/gate/weights': C.interleave(a_g, b_g),
        f'{tag}/gating_linear/weights': out_gate.T,
        f'{tag}/output_projection/weights': z_out.T,
    })
  return out


# ---------------------------------------------------------------------------
# transitions and the single (attention pair bias) path
# ---------------------------------------------------------------------------

def _transition(sd, prefix):
  """SwiGLU with block-concat fusion: a, b = chunk(linear(LN(x)), 2, -1)."""
  w = C._arr(sd[f'{prefix}.linear_no_bias_ab.weight'])
  half = w.shape[0] // 2
  return {
      'input_layer_norm/scale': C._arr(sd[f'{prefix}.layer_norm.weight']),
      'input_layer_norm/offset': C._arr(sd[f'{prefix}.layer_norm.bias']),
      'transition1/weights': C.block_concat(w[:half], w[half:]),
      'transition2/weights': C.t(sd[f'{prefix}.linear_out.weight']),
  }


def _attention_pair_bias(sd, prefix):
  """chai AttentionPairBias -> the PairFormerIteration single_* params."""
  qkvg = _split_qkvg_single(sd[f'{prefix}.attention.input2qkvg.weight'])
  o = C._arr(sd[f'{prefix}.attention.output_proj.weight'])   # (H, D, c_s)
  g = qkvg['g']
  return {
      'single_pair_logits_norm/scale': C._arr(sd[f'{prefix}.pair_layer_norm.weight']),
      'single_pair_logits_norm/offset': C._arr(sd[f'{prefix}.pair_layer_norm.bias']),
      'single_pair_logits_projection/weights': C.t(sd[f'{prefix}.pair_linear.weight']),
      'single_attention_layer_norm/scale': C._arr(sd[f'{prefix}.single_layer_norm.weight']),
      'single_attention_layer_norm/offset': C._arr(sd[f'{prefix}.single_layer_norm.bias']),
      'single_attention_q_projection/weights': qkvg['q'],
      'single_attention_q_projection/bias': C._arr(sd[f'{prefix}.attention.query_bias']),
      'single_attention_k_projection/weights': qkvg['k'],
      'single_attention_v_projection/weights': qkvg['v'],
      'single_attention_gating_query/weights': g.reshape(g.shape[0], -1),
      'single_attention_transition2/weights': o.reshape(-1, o.shape[-1]),
  }


# ---------------------------------------------------------------------------
# the trunk pairformer stack
# ---------------------------------------------------------------------------

def _pairformer_block(sd, i, stack='pairformer_stack.blocks', tri_attn=True,
                      single=True):
  """One chai PairformerBlock. The confidence head's four blocks are the same
  module under a different prefix (`blocks.N`), so they share this recipe.

  `tri_attn=False` omits the triangle attention -- see map_confidence_head for
  why that variant cannot be expressed as weights alone.
  """
  prefix = f'{stack}.{i}'
  out = _triangle_mul_pair(sd, f'{prefix}.triangle_multiplication')
  if tri_attn == 'joint':
    out.update(_tri_attention_joint(sd, f'{prefix}.triangle_attention'))
  elif tri_attn:
    out.update(_tri_attention_pair(sd, f'{prefix}.triangle_attention'))
  for k, v in _transition(sd, f'{prefix}.transition_pair').items():
    out[f'pair_transition/{k}'] = v
  if single:
    out.update(_attention_pair_bias(sd, f'{prefix}.attention_pair_bias'))
    for k, v in _transition(sd, f'{prefix}.transition_single').items():
      out[f'single_transition/{k}'] = v
  return out


def map_trunk_pairformer(sd, params, n_blocks=TRUNK_PAIRFORMER_BLOCKS):
  scope = 'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer'
  C.populate(params, scope,
             C.stack_blocks(lambda i: _pairformer_block(sd, i), n_blocks))


# ---------------------------------------------------------------------------
# the MSA module
# ---------------------------------------------------------------------------

def _msa_pair_weighted_averaging(sd, prefix, heads=8):
  """chai MSAPairWeightedAveraging -> AF3 MSAAttention params.

  `linear_msa2vg` is one projection holding [v | g]; chai's v is 8 heads x 32
  (AF3 would derive 64 // 8 = 8, hence the msa_attention.value_dim widening).
  """
  vg = C._arr(sd[f'{prefix}.linear_msa2vg.weight'])       # (2*H*D, c_m)
  half = vg.shape[0] // 2
  v, g = vg[:half], vg[half:]
  # CONTIGUOUS HALVES IS CORRECT, read off chai's graph rather than guessed:
  # linear_msa2vg's output is reshaped to [..., 2, H, D] and permuted
  # [3,0,1,2,4,5] before unbind(0), so the leading split is the OUTER factor of
  # the feature axis -- v = rows [0:H*D], g = rows [H*D:]. unbind[0] feeds the
  # masked_fill (the VALUE) and unbind[1] the sigmoid (the GATE).
  # Reading it head-major instead scores z_trunk 0.9355 vs 0.9300, and BOTH
  # lose to ablating the module (0.9475) -- so that score was coincidence, not
  # a fix. The defect is elsewhere in this module.
  #
  # chai's algorithm, decoded from the graph (trunk forward_256, 10746-10845):
  #   w = softmax(masked_fill(permute(linear_pair(LN_pair(z))), ~pair_mask,
  #                           -10000), dim=-1)          # (b, H, i, j)
  #   v, g  from LN_msa(m) -> linear_msa2vg as above; v masked by the msa mask
  #   out = einsum('abcd,aedbf->aecbf', w, v) * sigmoid(g)
  #   out = linear_out_no_bias(reshape(out))
  # Compare each step against ours before changing any more weights.
  #
  # RULED OUT so far, all by measurement against the 0.9988 ceiling (ablating
  # the module scores 0.9475, computing it 0.9300):
  #   - the v/g split of linear_msa2vg (contiguous halves IS right; head-major
  #     scores 0.9355 and still loses to ablation)
  #   - missing weights (all 8 msa_attention params map; gating_query has no
  #     bias to strand at bias_init=1.0)
  #   - the algebra (its einsum 'abcd,aedbf->aecbf' IS our 'hqk,bkhc->bqhc';
  #     the gate flattens (H, D) identically)
  #   - the pair-logit mask and the v masking (matched to its graph; both are
  #     no-ops on a fully-covered MSA, z_trunk unchanged at 0.92997)
  #   - WHICH pair rep the msa update reads: chai's pair LayerNorm input has
  #     outer_product_mean.0.linear_out in its ancestry, so it reads the
  #     POST-OPM pair exactly as we do
  # Next approach should be numeric rather than structural: tap chai's msa
  # representation after block 0 and diff it against ours directly.
  return {
      'act_norm/scale': C._arr(sd[f'{prefix}.layernorm_msa.weight']),
      'act_norm/offset': C._arr(sd[f'{prefix}.layernorm_msa.bias']),
      'pair_norm/scale': C._arr(sd[f'{prefix}.layernorm_pair.weight']),
      'pair_norm/offset': C._arr(sd[f'{prefix}.layernorm_pair.bias']),
      'pair_logits/weights': C.t(sd[f'{prefix}.linear_pair.weight']),
      'v_projection/weights': C.v_std(v, heads, half // heads),
      'gating_query/weights': g.T,
      'output_projection/weights': C.t(sd[f'{prefix}.linear_out_no_bias.weight']),
  }


def _grouped_outer_product_mean(sd, prefix):
  """chai's grouped OPM. NOT AF3-shaped -- these names need the chai1 branch.

  `weight_ab (2, G, K, c_m)` is the left/right pair of projections for G groups
  of a K x K outer product: the trace einsums are
  `abc,defc->abdef` per side then `abcde,afcdg->cegabf`, which contracts ONLY
  the MSA-depth axis and BROADCASTS the group index `a`, giving G*K*K = 512
  channels. AF3's OPM is the G = 1 case. Both LayerNorms here are affine-free in
  the trace (`layer_norm(x, [c])` with no weight), so the input norm is
  synthesised; `ln_out` over the 512 product channels IS learned.
  """
  ab = C._arr(sd[f'{prefix}.weight_ab'])                  # (2, G, K, c_m)
  g, k, c_m = ab.shape[1], ab.shape[2], ab.shape[3]
  w_out = C._arr(sd[f'{prefix}.linear_out.weight'])       # (c_z, G*K*K)
  return {
      'layer_norm_input/scale': np.ones(c_m, np.float32),
      'layer_norm_input/offset': np.zeros(c_m, np.float32),
      # (G, K, c_m) -> (c_m, G*K): one Linear per side, split into groups in the
      # forward. Group axis before the within-group channel, matching the branch.
      'left_projection/weights': ab[0].transpose(2, 0, 1).reshape(c_m, g * k),
      'right_projection/weights': ab[1].transpose(2, 0, 1).reshape(c_m, g * k),
      'product_norm/scale': C._arr(sd[f'{prefix}.ln_out.weight']),
      'product_norm/offset': C._arr(sd[f'{prefix}.ln_out.bias']),
      '::output_w': w_out.T.reshape(g, k, k, w_out.shape[0]),
      '::output_b': C._arr(sd[f'{prefix}.linear_out.bias']),
  }


def _msa_block(sd, i):
  """One MSA-module block.

  chai runs 4 of these but holds msa-side weights for only the first 3: the last
  block never updates `m`, since nothing reads it again. stack_blocks zero-fills
  the missing slots, the same convention IF2's dead final block uses.
  """
  out = {}
  for k, v in C.nest('outer_product_mean',
                     _grouped_outer_product_mean(sd, f'msa_module.outer_product_mean.{i}')).items():
    out[k] = v
  pwa = f'msa_module.msa_pair_weighted_averaging.{i}'
  if f'{pwa}.linear_msa2vg.weight' in sd:
    for k, v in _msa_pair_weighted_averaging(sd, pwa).items():
      out[f'msa_attention1/{k}'] = v
    for k, v in _transition(sd, f'msa_module.msa_transition.{i}').items():
      out[f'msa_transition/{k}'] = v
  out.update(_triangle_mul_pair(sd, f'msa_module.triangular_multiplication.{i}'))
  out.update(_tri_attention_pair(sd, f'msa_module.triangular_attention.{i}'))
  for k, v in _transition(sd, f'msa_module.pair_transition.{i}').items():
    out[f'pair_transition/{k}'] = v
  return out


def map_confidence_head(sd, params, n_blocks=4, n_atom=24):
  """chai's ConfidenceHead -> AF3's.

  The four blocks are ordinary chai PairformerBlocks, so they reuse
  _pairformer_block; PairFormerIteration's chai1 branch already makes them
  parallel. What is specific to this head is the pair init and three logit
  projections.

  Three things fall out as pure weight surgery rather than forward branches:

  * The head LayerNorms are AFFINE-FREE in the graph (`layer_norm(x, [c], None,
    None)`), which is exactly a LayerNorm with scale 1 and offset 0 -- so they
    are synthesised as identity rather than gated out of the graph.
  * `pde = pde_projection(LN(z) + LN(z)^T)` and ours is
    `left(LN(z)) + swapaxes(left(LN(z)))`. A Linear commutes with the transpose
    of the two token axes, so the two are the same function.
  * pLDDT is projected to 37 atom slots x 50 bins and then gathered per atom by
    the atom's within-token index. Our dense layout already carries the slot
    axis, so the gather IS our layout and the mapping is a reshape and a slice
    to `n_atom` -- chai's 37 is padding for the largest residue.

  The triangle attention needs its own recipe (_tri_attention_joint) and a
  forward branch (GridSelfAttention.dual_output). Where the trunk's `linear_out` is (c_z, 2*c_z) -- one shared output
  over both directions, so `W @ [a1;a2]` splits into AF3's two independent
  per-direction projections that then SUM -- the confidence head's is
  (2*c_z, 2*c_z): each direction's output reads BOTH directions' heads, and the
  two halves are combined as `out1 + transpose(out2)`. The cross-direction
  blocks are not a rounding detail; they are the same magnitude as the diagonal
  ones (Frobenius 24-30 either way) and dropping them costs corr 0.831 / rms
  0.597 against chai's own node-824 output.
  """
  scope = 'diffuser/confidence_head'
  C.populate(params,
             f'{scope}/__layer_stack_no_per_layer/confidence_pairformer',
             C.stack_blocks(
                 lambda i: _pairformer_block(sd, i, 'blocks',
                                            tri_attn='joint'),
                 n_blocks))

  # pair init: z_trunk + col(s_inputs) + row(s_inputs) + dgram(reference atoms)
  s2p = C._arr(sd['single_to_pair_proj.weight'])            # (2*c_z, c_s)
  half = s2p.shape[0] // 2
  emb = f'{scope}/~_embed_features'
  # which half is the row term and which the column: swapping them moves the
  # head's logits by 0.0002 (pae 0.9773 vs 0.9775), so this is not load-bearing
  # and the pair init lands at corr 0.999839 either way.
  params[f'{emb}/left_target_feat_project'] = {'weights': s2p[:half].T}
  params[f'{emb}/right_target_feat_project'] = {'weights': s2p[half:].T}
  params[f'{emb}/distogram_feat_project'] = {
      'weights': C._arr(sd['atom_distance_bins_projection.weight']).T}

  ln = lambda c: {'scale': np.ones(c, np.float32),
                  'offset': np.zeros(c, np.float32)}
  pae = C._arr(sd['pae_projection.weight'])                  # (bins, c_z)
  pde = C._arr(sd['pde_projection.weight'])
  plddt = C._arr(sd['plddt_projection.weight'])              # (slots*bins, c_s)
  c_z, c_s = pae.shape[1], plddt.shape[1]
  n_bins = plddt.shape[0] // 37
  params[f'{scope}/pae_logits'] = {'weights': pae.T}
  params[f'{scope}/pae_logits_ln'] = ln(c_z)
  params[f'{scope}/left_half_distance_logits'] = {'weights': pde.T}
  params[f'{scope}/logits_ln'] = ln(c_z)
  # all 37 ATOM37 slots: the forward gathers by atom name, so no slice here
  params[f'{scope}/plddt_logits'] = {'weights': plddt.T.reshape(c_s, 37, n_bins)}
  params[f'{scope}/plddt_logits_ln'] = ln(c_s)


def map_template_embedder(sd, params, n_blocks=2, n_dgram=39):
  """chai's template embedder (inside trunk.pt) -> AF3's TemplateEmbedding.

  The stack is two PAIR-ONLY chai Pairformer blocks at c_z=64 -- trunk naming
  (pair2qkvg1/2, pair2b, out_scalers, so head-major) but with no single track.

  The features arrive through one 76-column projection where AF3 has nine
  separate ones, so the columns split into them. Layout verified against chai's
  own `Trunk/in/template_input_feats` at corr 0.99999838 for all four templates
  (tools/oracles/chai1/cmp_template_stream.py):

      Distogram(39) | Mask(2) | ResType(32) | UnitVector(3)     alphabetical

  ResType is the interesting one. chai holds ONE 33x32 embedding and SUMS the
  row and column embeddings, where AF3 has two separate 31-class projections.
  Linearity makes that the same function, so both of AF3's get the same matrix,
  E @ W_rt^T with the 33->31 restype permutation. Reading it one-sided instead
  scores 0.938, so it was worth testing rather than assuming.

  The two mask channels are RESOLVED (they were long guessed, because they are
  identical on every captured template -- all four 1STP hits -- so no capture
  could distinguish them). chai_lab's TemplateMaskGenerator._generate builds
  `cat([bij_backbone, bij_pseudo_beta])`, so column 0 is the BACKBONE frame mask
  and column 1 is the PSEUDO-BETA mask. AF3's to_concat is the other way round
  (index 1 pseudo-beta, index 7 backbone), so they cross over below. This only
  changes anything for a template residue with a complete backbone but no CB,
  or the reverse.

  KNOWN DIVERGENCE, single-chain templates unaffected: chai masks all four
  template features to WITHIN-CHAIN pairs (allow_inter_chain=False), and its
  distogram assigns cross-chain pairs the one-hot MASK class rather than zeroing
  them, where AF3 multiplies the distogram by the pseudo-beta mask and applies
  multichain_mask_2d. For a single-chain template these coincide; a multi-chain
  template would need a chai1 branch in template_modules.construct_input.
  """
  scope = 'diffuser/evoformer/template_embedding'
  single = f'{scope}/single_template_embedding'
  C.populate(params,
             f'{single}/__layer_stack_no_per_layer/template_embedding_iteration',
             C.stack_blocks(
                 lambda i: _pairformer_block(sd, i, 'template_embedder.pairformer.blocks',
                                             single=False),
                 n_blocks))

  w = C._arr(sd['template_embedder.proj_in.1.weight'])        # (64, 256)
  params[f'{single}/query_embedding_norm'] = {
      'scale': C._arr(sd['template_embedder.proj_in.0.weight']),
      'offset': C._arr(sd['template_embedder.proj_in.0.bias'])}
  params[f'{single}/output_layer_norm'] = {
      'scale': C._arr(sd['template_embedder.template_layernorm.weight']),
      'offset': C._arr(sd['template_embedder.template_layernorm.bias'])}
  params[f'{scope}/output_linear'] = {
      'weights': C._arr(sd['template_embedder.proj_out.1.weight']).T}
  params[f'{single}/template_pair_embedding_8'] = {'weights': w.T}
  return params


def map_template_features(fe, params, n_dgram=39):
  """The TEMPLATES stream's 76 columns -> AF3's template_pair_embedding_0..7.

  Lives in feature_embedding.pt rather than the trunk, hence a second function.
  See map_template_embedder for the layout and how it was verified.
  """
  single = 'diffuser/evoformer/template_embedding/single_template_embedding'
  w = C._arr(fe['input_projs.TEMPLATES.0.weight'])            # (64, 76)
  emb = C._arr(
      fe['feature_embeddings.TEMPLATES.TemplateResType.embedding.weight'])
  lo = 0
  dgram, lo = w[:, lo:lo + n_dgram], lo + n_dgram
  mask, lo = w[:, lo:lo + 2], lo + 2
  restype, lo = w[:, lo:lo + emb.shape[1]], lo + emb.shape[1]
  unit = w[:, lo:lo + 3]
  # (33, 32) @ (32, 64) -> (33, 64), then reorder chai's 33 classes to our 31
  rt = (emb @ restype.T)[CHAI1_RESTYPE_PERM]
  out = {
      'template_pair_embedding_0': dgram.T,      # distogram
      'template_pair_embedding_1': mask[:, 1],   # AF3 idx 1 = pseudo-beta = chai col 1
      'template_pair_embedding_2': rt,           # restype, column side
      'template_pair_embedding_3': rt,           # restype, row side
      'template_pair_embedding_4': unit[:, 0],   # unit vector x
      'template_pair_embedding_5': unit[:, 1],   # y
      'template_pair_embedding_6': unit[:, 2],   # z
      'template_pair_embedding_7': mask[:, 0],   # AF3 idx 7 = backbone = chai col 0
  }
  for k, v in out.items():
    params[f'{single}/{k}'] = {'weights': v}
  return params


def map_recycling(sd, params):
  """chai's recycle projections are AF3's, name for name.

  `token_{single,pair}_recycle_proj` is Sequential(affine LayerNorm, bias-free
  Linear) and the result is ADDED to the initial representation -- exactly
  AF3's `prev_embedding_layer_norm` + `prev_embedding`. Nothing to reshape.
  """
  for chai, af3 in (('token_pair_recycle_proj', 'prev_embedding'),
                    ('token_single_recycle_proj', 'prev_single_embedding')):
    params[f'diffuser/evoformer/{af3}_layer_norm'] = {
        'scale': np.asarray(sd[f'{chai}.0.weight']),
        'offset': np.asarray(sd[f'{chai}.0.bias'])}
    params[f'diffuser/evoformer/{af3}'] = {
        'weights': C.t(sd[f'{chai}.1.weight'])}


def map_msa_input(fe, params, n_src=6):
  """chai's 42-column MSA stream -> our msa_activations.

  chai has no target-feat term on the MSA and no msa parameters in the trunk
  outside msa_module, so its whole MSA embedding is this one projection --
  which, unlike ours, carries a BIAS (hence use_bias for chai1 in the graph,
  and extra_msa_target_feat skipped rather than left at random init).

  Column order is alphabetical and was derived, not guessed: input_projs.MSA is
  (64, 42) and therefore overdetermined, so the input is recoverable as
  x = (ref - b) @ pinv(W.T) and can be read off directly. Guessing had scored
  0.84 and 0.57. See tools/oracles/chai1/cmp_msa_stream.py.

      IsPairedMSA(1) | MSADataSource(6) | MSADeletionValue(1)
      | MSAHasDeletion(1) | MSAOneHot(33)

  The forward emits the same order but keeps OUR 32-class one-hot, so chai's 33
  weight columns are permuted onto it here: our 31 polymer classes through
  CHAI1_RESTYPE_PERM, then our trailing class onto chai's 32 (its mask row).

  KNOWN WRONG SOMEWHERE -- do not treat this as finished. With a real 2144-row
  MSA from chai's own server, MSA HELPS chai and HURTS us on 1STP+BTN:

      native ESM only 0.627 -> ESM+MSA 0.525   (MSA gains 0.10 A)
      ours   ESM only 0.707 -> ESM+MSA 0.822   (MSA costs 0.12 A)

  A wrong sign, not a small residual, so it is a defect and not the target
  being insensitive. Prime suspect is this very permutation: if our MSA row
  vocabulary is not the restype vocabulary CHAI1_RESTYPE_PERM was built for,
  every MSA row becomes garbage residues, which is actively harmful rather than
  merely uninformative. Next step is the injection gate -- feed chai's own
  captured MSA features through this projection and compare against its
  `Trunk/in/msa_input_feats`, exactly as cmp_msa_stream.py does for the raw
  layout. Until then featurise_chai1 keeps msa=False.
  """
  w = C._arr(fe['input_projs.MSA.0.weight'])                 # (64, 42)
  b = C._arr(fe['input_projs.MSA.0.bias'])
  lo = 0
  paired, lo = w[:, lo:lo + 1], lo + 1
  src, lo = w[:, lo:lo + n_src], lo + n_src
  delval, lo = w[:, lo:lo + 1], lo + 1
  hasdel, lo = w[:, lo:lo + 1], lo + 1
  onehot = w[:, lo:]                                         # (64, 33)
  ours = onehot[:, list(CHAI1_RESTYPE_PERM) + [onehot.shape[1] - 1]]
  stacked = np.concatenate([paired, src, delval, hasdel, ours], axis=1)
  params['diffuser/evoformer/msa_activations'] = {
      'weights': stacked.T, 'bias': b}
  return params


def map_msa_single_to_msa(sd, params):
  """chai's `msa_module.linear_s2m` -> AF3's extra_msa_target_feat.

  chai DOES add a single-representation term to the MSA, and it lives INSIDE
  msa_module, which is how it was missed: the search that concluded "no msa
  parameters outside msa_module" excluded the very module holding it. Skipping
  it left the MSA stack missing a whole input.
  """
  params['diffuser/evoformer/extra_msa_target_feat'] = {
      'weights': C.t(sd['msa_module.linear_s2m.weight'])}
  return params


def map_msa_module(sd, params, n_blocks=4):
  scope = 'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack'
  C.populate(params, scope, C.stack_blocks(lambda i: _msa_block(sd, i), n_blocks))


# ---------------------------------------------------------------------------
# distogram head
# ---------------------------------------------------------------------------

def map_distogram_head(sd, params):
  """The post-hoc distogram head from sokrypton/chai-lab@dgram.

  `nn.Sequential(LayerNorm(c_z), Linear(c_z, 2*c_z), GELU, Linear(2*c_z, bins))`,
  so the state dict keys are m.0 (norm), m.1 (hidden), m.3 (logits).

  Its bins are AF3's own: fitting the native 6MRR distogram against the
  structure that same trunk produced gives Pearson 0.971 / RMSE 0.35 A over
  pseudo-beta distances at first_break 2.3125, last_break 21.6875, 64 bins
  (0-32 gives RMSE 5.6, CA instead of CB gives 1.14). Bin 63 is a catch-all.
  So heads.distogram needs no chai-specific config and the con / i_con losses
  transfer unchanged.
  """
  scope = 'diffuser/distogram_head'
  C.populate(params, scope, {
      'input_layer_norm/scale': C._arr(sd['m.0.weight']),
      'input_layer_norm/offset': C._arr(sd['m.0.bias']),
      'hidden/weights': C.t(sd['m.1.weight']),
      'hidden/bias': C._arr(sd['m.1.bias']),
      'half_logits/weights': C.t(sd['m.3.weight']),
      'half_logits/bias': C._arr(sd['m.3.bias']),
  })


# ---------------------------------------------------------------------------
# the diffusion module: three atom stacks and the token transformer
# ---------------------------------------------------------------------------

DIFF_TOKEN_BLOCKS = 16
DIFF_TOKEN_SUPER = 4
ATOM_BLOCKS = 3


def _adaln_pair(w):
  """chai fuses adaLN's scale and shift into one linear; AF3 keeps two.

  `lin_s_merged` is (2 * c_out, c_in) and the trace chunks its OUTPUT in two,
  scale first. AF3 wants (c_in, c_out) each, and no bias -- chai has none, which
  is why the chai1 branch builds single_cond_scale with use_bias=False.
  """
  c_out = w.shape[0] // 2
  return C.t(w[:c_out]), C.t(w[c_out:])


def _split_to_qkv_fused(w, heads):
  """Token transformer: one (heads * 3 * dim, c_in) projection -> AF3's three.

  The trace reshapes to (..., heads, 3 * dim) and only then chunks in three, so
  q, k and v are contiguous WITHIN a head -- not three stacked head-blocks. Get
  this backwards and every head reads a third of the wrong projection.
  """
  d3 = w.shape[0] // heads
  dim = d3 // 3
  g = w.reshape(heads, d3, w.shape[1])
  return [np.transpose(g[:, i * dim:(i + 1) * dim], (2, 0, 1)) for i in range(3)]


def _atom_block(sd, prefix, name, i):
  """One block of a chai atom transformer -> AF3 cross-attention params."""
  P = f'{prefix}.atom_transformer.local_diffn_transformer'
  L, T = f'{P}.local_attentions.{i}', f'{P}.transitions.{i}'
  out = {}

  # to_qkv is already (3, heads, dim, c_in) here -- the role axis is FIRST,
  # unlike the token transformer's fused layout.
  qkv = sd[f'{L}.to_qkv.weight']
  for j, role in enumerate(('q', 'k', 'v')):
    out[f'{name}{role}_projection/weights'] = np.transpose(qkv[j], (2, 0, 1))
  out[f'{name}q_projection/bias'] = sd[f'{L}.q_bias']

  # chai adaLN's the single repr ONCE and gathers k/v from the result, while AF3
  # adaLN's the query and key layouts separately -- so both of AF3's scopes get
  # the same chai parameters.
  scale, shift = _adaln_pair(sd[f'{L}.single_layer_norm.lin_s_merged.weight'])
  for side in ('q', 'k'):
    out[f'{name}{side}single_cond_scale/weights'] = scale
    out[f'{name}{side}single_cond_bias/weights'] = shift

  out[f'{name}adaptive_zero_cond/weights'] = C.t(sd[f'{L}.out_proj.weight'])
  out[f'{name}adaptive_zero_cond/bias'] = sd[f'{L}.out_proj.bias']

  scale, shift = _adaln_pair(sd[f'{T}.ada_ln.lin_s_merged.weight'])
  out[f'{name}ffw_single_cond_scale/weights'] = scale
  out[f'{name}ffw_single_cond_bias/weights'] = shift
  out[f'{name}ffw_transition1/weights'] = C.t(sd[f'{T}.linear_a_nobias_double.weight'])
  out[f'{name}ffw_transition2/weights'] = C.t(sd[f'{T}.linear_b_nobias.weight'])
  out[f'{name}ffw_adaptive_zero_cond/weights'] = C.t(sd[f'{T}.linear_s_biasinit_m2.weight'])
  out[f'{name}ffw_adaptive_zero_cond/bias'] = sd[f'{T}.linear_s_biasinit_m2.bias']
  return out


def map_atom_transformer(sd, params, prefix, scope, name):
  """A 3-block chai atom stack -> one AF3 CrossAttTransformer scope."""
  C.populate(params, f'{scope}/__layer_stack_with_per_layer',
             C.stack_blocks(lambda i: _atom_block(sd, prefix, name, i), ATOM_BLOCKS))
  # The atom-pair bias LayerNorm is SHARED across the stack (one
  # blocked_pairs2blocked_bias.0) with a per-block slice in .1, which is exactly
  # AF3's shared-LN layout -- but chai's LayerNorm is affine on both.
  P = f'{prefix}.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias'
  params[f'{scope}/pair_input_layer_norm'] = {
      'scale': np.asarray(sd[f'{P}.0.weight']),
      'offset': np.asarray(sd[f'{P}.0.bias']),
  }
  # (blocks, heads, c_pair) -> AF3's (c_pair, blocks, heads)
  params[f'{scope}/pair_logits_projection'] = {
      'weights': np.transpose(np.asarray(sd[f'{P}.1.weight']), (2, 0, 1))}


def _token_block(sd, i, heads=16):
  """One block of chai's 16-block diffusion token transformer."""
  B = f'diffusion_transformer.blocks.{i}'
  n = 'transformer'
  out = {}
  q, k, v = _split_to_qkv_fused(sd[f'{B}.to_qkv.weight'], heads)
  out[f'{n}q_projection/weights'] = q
  out[f'{n}k_projection/weights'] = k
  out[f'{n}v_projection/weights'] = v
  out[f'{n}q_projection/bias'] = sd[f'{B}.q_bias']

  scale, shift = _adaln_pair(sd[f'{B}.norm_in.lin_s_merged.weight'])
  out[f'{n}single_cond_scale/weights'] = scale
  out[f'{n}single_cond_bias/weights'] = shift

  # chai DOES project the token attention output (`to_out`); only its atom
  # stacks skip that projection.
  out[f'{n}transition2/weights'] = C.t(sd[f'{B}.to_out.weight'])
  out[f'{n}adaptive_zero_cond/weights'] = C.t(sd[f'{B}.gate_proj.0.weight'])
  out[f'{n}adaptive_zero_cond/bias'] = sd[f'{B}.gate_proj.0.bias']

  T = f'{B}.transition'
  scale, shift = _adaln_pair(sd[f'{T}.ada_ln.lin_s_merged.weight'])
  out[f'{n}ffw_single_cond_scale/weights'] = scale
  out[f'{n}ffw_single_cond_bias/weights'] = shift
  out[f'{n}ffw_transition1/weights'] = C.t(sd[f'{T}.linear_a_nobias_double.weight'])
  out[f'{n}ffw_transition2/weights'] = C.t(sd[f'{T}.linear_b_nobias.weight'])
  out[f'{n}ffw_adaptive_zero_cond/weights'] = C.t(sd[f'{T}.linear_s_biasinit_m2.weight'])
  out[f'{n}ffw_adaptive_zero_cond/bias'] = sd[f'{T}.linear_s_biasinit_m2.bias']

  # per-block pair bias: affine LayerNorm then a (heads, c_pair) projection
  out['pair_input_layer_norm/scale'] = np.asarray(sd[f'{B}.pair_layer_norm.weight'])
  out['pair_input_layer_norm/offset'] = np.asarray(sd[f'{B}.pair_layer_norm.bias'])
  out['pair_logits_projection/weights'] = C.t(sd[f'{B}.pair_linear.weight'])
  return out


def map_diffusion_token_transformer(sd, params):
  scope = ('diffuser/~/diffusion_head/transformer'
           '/__layer_stack_no_per_layer/__layer_stack_no_per_layer')
  C.populate(params, scope, C.stack_super(
      lambda i: _token_block(sd, i), DIFF_TOKEN_BLOCKS, DIFF_TOKEN_SUPER))


def _cond_transition(sd, prefix):
  """chai's `*_trans{1,2}`: affine LayerNorm then SwiGLU -> AF3's ffw_* names."""
  return {
      'ffw_layer_norm/scale': np.asarray(sd[f'{prefix}.layer_norm.weight']),
      'ffw_layer_norm/offset': np.asarray(sd[f'{prefix}.layer_norm.bias']),
      'ffw_transition1/weights': C.t(sd[f'{prefix}.linear_no_bias_ab.weight']),
      'ffw_transition2/weights': C.t(sd[f'{prefix}.linear_out.weight']),
  }


def map_diffusion_conditioning(sd, params):
  """chai's diffusion_conditioning + the atom encoder/decoder one-offs.

  Deliberately NOT mapped here: `single_cond_initial_projection` (831, 384) and
  `pair_cond_initial_projection` (395, 256). Those widths come from AF3's
  447-d target_feat, while chai concatenates its own 384-d token_single_initial
  and 256-d token_pair_initial -- so they only resolve once the input embedder
  feeds chai's s_init/z_init in as target_feat. Same dependency the Boltz-2 port
  hit at its Stage 3.
  """
  scope = 'diffuser/~/diffusion_head'
  D = 'diffusion_conditioning'
  for i, which in enumerate(('1', '2')):
    C.populate(params, scope, {f'pair_transition_{i}{k}': v for k, v in
                               _cond_transition(sd, f'{D}.pair_trans{which}').items()})
    C.populate(params, scope, {f'single_transition_{i}{k}': v for k, v in
                               _cond_transition(sd, f'{D}.single_trans{which}').items()})

  # The single conditioning's initial projection. This one only became mappable
  # once the input embedder landed: AF3's width here is 1 + its 447-d
  # target_feat, and it collapses to chai's 768 (s_init 384 + s_trunk 384) as
  # soon as target_feat IS chai's s_init.
  # THE CONCAT ORDERS ARE OPPOSITE BETWEEN THE TWO TRACKS. chai builds the
  # single conditioning from cat[s_init, s_trunk] but the pair one from
  # cat[z_trunk, z_init]; AF3 puts the trunk first on both. So the single
  # weight's two input halves have to be swapped, and the pair one must not be.
  # Shapes are identical either way, so nothing catches this but reading the trace.
  tin_w = np.asarray(sd[f'{D}.token_in_proj.1.weight'])          # (384, 768)
  tin_n = np.asarray(sd[f'{D}.token_in_proj.0.weight'])          # (768,)
  h = tin_w.shape[1] // 2
  swap_cols = lambda w: np.concatenate([w[:, h:], w[:, :h]], axis=1)
  swap_vec = lambda v: np.concatenate([v[h:], v[:h]])
  # AFFINE: chai's token_in_proj.0 carries a bias (absmax 0.20) and it must be
  # swapped with the scale, since our features_1d is the other concat order.
  params[f'{scope}/single_cond_initial_norm'] = {
      'scale': swap_vec(tin_n),
      'offset': swap_vec(np.asarray(sd[f'{D}.token_in_proj.0.bias']))}
  params[f'{scope}/single_cond_initial_projection'] = {
      'weights': C.t(swap_cols(tin_w))}

  # the pair side keeps chai's order (trunk first), so it maps straight across
  tp_w = np.asarray(sd[f'{D}.token_pair_proj.1.weight'])
  # AFFINE too (bias absmax 0.31); the pair side keeps chai's order so no swap
  params[f'{scope}/pair_cond_initial_norm'] = {
      'scale': np.asarray(sd[f'{D}.token_pair_proj.0.weight']),
      'offset': np.asarray(sd[f'{D}.token_pair_proj.0.bias'])}
  params[f'{scope}/pair_cond_initial_projection'] = {'weights': C.t(tp_w)}

  # The Fourier embedding itself. chai's is TRAINED where stock AF3 hardcodes
  # constants, and its bias has absorbed the log(sigma_data)/4 offset that AF3
  # applies by dividing sigma first. Leaving this unmapped left the denoiser
  # reading AF3's constants off a wrongly-scaled sigma.
  params.setdefault(scope, {})
  # chai does not divide the noise level by sigma_data before the Fourier
  # embedding; AlphaFold 3 does. The embedding is cos(2pi(0.25*log(s)*w + b)),
  # and 0.25*log(sigma) = 0.25*log(sigma/16) + 0.25*log(16), so feeding AF3's
  # scaled input and adding w * 0.25*log(16) to the BIAS is the same function --
  # exactly, to float round-off. That keeps the forward graph uniform instead of
  # branching on the model to decide what to divide by.
  fourier_weight = np.asarray(sd[f'{D}.fourier_embedding.weights'])
  params[scope]['fourier_embedding_weight'] = fourier_weight
  params[scope]['fourier_embedding_bias'] = (
      np.asarray(sd[f'{D}.fourier_embedding.bias'])
      + fourier_weight * 0.25 * np.log(SIGMA_DATA))

  # chai closes each conditioning track with an affine LayerNorm that AF3 has
  # no counterpart for -- single_ln / pair_ln
  params[f'{scope}/single_cond_final_norm'] = {
      'scale': np.asarray(sd[f'{D}.single_ln.weight']),
      'offset': np.asarray(sd[f'{D}.single_ln.bias'])}
  params[f'{scope}/pair_cond_final_norm'] = {
      'scale': np.asarray(sd[f'{D}.pair_ln.weight']),
      'offset': np.asarray(sd[f'{D}.pair_ln.bias'])}

  # the noise embedding: chai's fourier_proj is Seq(LayerNorm, Linear-no-bias)
  params[f'{scope}/noise_embedding_initial_norm'] = {
      'scale': np.asarray(sd[f'{D}.fourier_proj.0.weight']),
      'offset': np.asarray(sd[f'{D}.fourier_proj.0.bias'])}
  params[f'{scope}/noise_embedding_initial_projection'] = {
      'weights': C.t(sd[f'{D}.fourier_proj.1.weight'])}

  E, DEC = 'atom_attention_encoder', 'atom_attention_decoder'
  one_offs = {
      # atom <-> token traffic
      'diffusion_atom_positions_to_features': C.t(sd[f'{E}.prev_pos_embed.weight']),
      'diffusion_project_atom_features_for_aggr': C.t(sd[f'{E}.to_token_single.0.weight']),
      'diffusion_project_token_features_for_broadcast': C.t(sd[f'{DEC}.token_to_atom.weight']),
      'diffusion_atom_features_to_position_update': C.t(sd[f'{DEC}.to_pos_updates.1.weight']),
      # chai's structure_cond_to_token_structure_proj: the s_cond term added to
      # the pooled atom features before the token transformer. Unmapped, this
      # sits at final_init (zeros) and contributes NOTHING -- caught because our
      # tokens_in magnitude equalled `pooled` exactly while chai's was larger.
      'single_cond_embedding_projection': C.t(
          sd['structure_cond_to_token_structure_proj.weight']),
      # trunk conditioning gathered onto atoms / atom pairs
      'diffusion_embed_trunk_single_cond': C.t(sd[f'{E}.token_to_atom_single.1.weight']),
      'diffusion_embed_trunk_pair_cond': C.t(sd[f'{E}.token_pair_to_atom_pair.1.weight']),
  }
  for name, w in one_offs.items():
    params[f'{scope}/{name}'] = {'weights': np.asarray(w)}

  # chai's post_attn_layernorm is affine on BOTH scale and offset; AF3's
  # output_norm is scale-only everywhere else
  params[f'{scope}/output_norm'] = {
      'scale': np.asarray(sd['post_attn_layernorm.weight']),
      'offset': np.asarray(sd['post_attn_layernorm.bias'])}

  # chai's to_pos_updates LayerNorm is affine, and its decoder conditions on a
  # second affine LayerNorm of the encoder's atom conditioning
  params[f'{scope}/diffusion_atom_features_layer_norm'] = {
      'scale': np.asarray(sd[f'{DEC}.to_pos_updates.0.weight']),
      'offset': np.asarray(sd[f'{DEC}.to_pos_updates.0.bias'])}
  params[f'{scope}/diffusion_post_atom_cond_layer_norm'] = {
      'scale': np.asarray(sd['post_atom_cond_layernorm.weight']),
      'offset': np.asarray(sd['post_atom_cond_layernorm.bias'])}

  # Both are AFFINE LayerNorms in chai -- weight AND bias. Mapping only the
  # scale drops the bias, and the graph has to create the offset for it to land
  # (see create_offset on these two norms in atom_cross_attention).
  norms = {
      'diffusion_lnorm_trunk_single_cond': f'{E}.token_to_atom_single.0',
      'diffusion_lnorm_trunk_pair_cond': f'{E}.token_pair_to_atom_pair.0',
  }
  for name, key in norms.items():
    params[f'{scope}/{name}'] = {'scale': np.asarray(sd[f'{key}.weight']),
                                 'offset': np.asarray(sd[f'{key}.bias'])}

  # the atom-pair projections. AF3 makes a second copy of each (`_1`) because it
  # builds the pair conditioning twice; chai has one set, so both copies get it.
  U = f'{E}.pair_update_block'
  for af3, chai in (('diffusion_single_to_pair_cond_row', 'atom_single_to_atom_pair_proj_h'),
                    ('diffusion_single_to_pair_cond_col', 'atom_single_to_atom_pair_proj_w')):
    w = C.t(sd[f'{U}.{chai}.1.weight'])
    params[f'{scope}/{af3}'] = {'weights': np.asarray(w)}
    params[f'{scope}/{af3}_1'] = {'weights': np.asarray(w)}


# ---------------------------------------------------------------------------
# the input embedder (chai's token_embedder archive)
# ---------------------------------------------------------------------------

# chai's 33-entry residue vocabulary -> our 31-class
# POLYMER_TYPES_WITH_UNKNOWN_AND_GAP. PERM[our_idx] = chai_idx, so a one-hot
# weight's input columns reorder as w[:, PERM]. chai's first 20 are already our
# 3-letter order; our '-'(21) is chai's '-'(31); RNA A/G/C/U come from
# RA/RG/RC/RU (21/23/22/24) and DNA DA/DG/DC/DT from 26/28/27/29. chai has BOTH
# an RNA and a DNA unknown where we have one nucleic 'N'; picking RX(25) matches
# what the boltz2 port chose.
CHAI1_RESTYPE_PERM = (list(range(21)) + [31] + [21, 23, 22, 24]
                      + [26, 28, 27, 29] + [25])

# cumulative offsets of the TOKEN stream's alphabetical concat
TOKEN_COLS = {
    'ChainIsCropped': (0, 1), 'ESMEmbeddings': (1, 2561),
    'IsDistillation': (2561, 2563), 'MSADeletionMean': (2563, 2564),
    'MSAProfile': (2564, 2597), 'MissingChainContact': (2597, 2598),
    'ResidueType': (2598, 2631), 'TokenBFactor': (2631, 2634),
    'TokenPLDDT': (2634, 2638),
}


# cumulative offsets of the TOKEN_PAIR stream's alphabetical concat
PAIR_COLS = {
    'DockingConstraintGenerator': (0, 6), 'RelativeChain': (6, 12),
    'RelativeEntity': (12, 15), 'RelativeSequenceSeparation': (15, 82),
    'RelativeTokenSeparation': (82, 149), 'TokenDistanceRestraint': (149, 156),
    'TokenPairPocketRestraint': (156, 163),
}


def map_pair_init(sds, params):
  """chai's pair initialisation -> AF3's left/right_single + position_activations.

  chai computes `z_init = W_p @ (token_pair_feats + outer_sum(s_init))`, with the
  final projection applied AFTER the sum. AF3 has no such trailing projection --
  but W_p is linear, so it distributes: fold it into every term instead of
  adding a graph op.

    left_single  = (W_p @ W_outer[:256]).T      outer-sum row half
    right_single = (W_p @ W_outer[256:]).T      outer-sum column half
    position_activations = (W_p @ P[:, rel_cols]).T, bias = W_p @ (P @ const + b)

  where P and b are the TRUNK half (rows 0:256) of the TOKEN_PAIR projection --
  chai chunks that 512-wide output into a trunk and a structure copy.
  """
  fe, te = sds['feature_embedding'], sds['token_embedder']
  Wp = np.asarray(te['token_pair_proj_in_trunk.weight'])          # (256, 256)
  Wo = np.asarray(te['token_single_to_token_pair_outer_sum_proj.weight'])  # (512, 384)
  half = Wo.shape[0] // 2
  params['diffuser/evoformer/left_single'] = {'weights': C.t(Wp @ Wo[:half])}
  params['diffuser/evoformer/right_single'] = {'weights': C.t(Wp @ Wo[half:])}

  P = np.asarray(fe['input_projs.TOKEN_PAIR.0.weight'])[:half]    # trunk half
  b = np.asarray(fe['input_projs.TOKEN_PAIR.0.bias'])[:half]
  rel = np.concatenate([
      P[:, slice(*PAIR_COLS['RelativeSequenceSeparation'])],
      P[:, slice(*PAIR_COLS['RelativeTokenSeparation'])]], axis=1)
  # the constant members of the stream, as a single input vector
  const = np.zeros((P.shape[1],), np.float32)
  for name, idx in (('DockingConstraintGenerator', 5), ('RelativeChain', 2),
                    ('RelativeEntity', 1)):
    const[PAIR_COLS[name][0] + idx] = 1.0
  for name in ('TokenDistanceRestraint', 'TokenPairPocketRestraint'):
    const[PAIR_COLS[name][1] - 1] = 1.0     # the -1 sentinel's mask column
  # `_relative_encoding` is not @hk.transparent, so haiku injects the method
  # name into the scope
  params['diffuser/evoformer/~_relative_encoding/position_activations'] = {
      'weights': C.t(Wp @ rel),
      'bias': Wp @ (P @ const + b),
  }


# cumulative offsets of the ATOM stream's alphabetical concat
ATOM_COLS = {
    'AtomNameOneHot': (0, 260), 'AtomRefCharge': (260, 261),
    'AtomRefElement': (261, 391), 'AtomRefMask': (391, 392),
    'AtomRefPos': (392, 395),
}


def map_atom_conditioning(sds, params, to_atom_cond_w, scope, prefix, half,
                          sds_enc=None, enc_prefix=None):
  """chai's pre-embedded atom features -> AF3's five split ref embeddings.

  chai does not embed reference features per block: its feature embedder emits
  ONE 128-d atom vector and `to_atom_cond` re-projects it. But both stages are
  linear over the same concatenated ref features, so the composition
  `to_atom_cond @ ATOM_proj` IS AF3's split embedding -- just written as one
  matrix. Slice its input columns back apart.

  `half` picks chai's chunk: the token embedder takes the TRUNK half of the
  512-wide ATOM projection and the diffusion module the STRUCTURE half.

  Three conventions have to be undone on the way:
    * chai divides ref_pos by 10 and AF3 does not, so the position columns are
      scaled by 1/10 here rather than branching the graph.
    * chai one-hots atom-name characters into 65 classes and elements into 130;
      AF3 uses 64 and 128. The extra classes are chai's mask class and two
      unused elements, never set for a real atom, so dropping them is exact.
    * chai's ATOM projection carries a BIAS and AF3's per-feature embeddings do
      not. It folds into embed_ref_mask, whose input is 1.0 for every real atom
      and 0 for padding -- exactly the right gate.
  """
  fe = sds['feature_embedding']
  P = np.asarray(fe['input_projs.ATOM.0.weight'])       # (256, 395)
  pb = np.asarray(fe['input_projs.ATOM.0.bias'])
  c = P.shape[0] // 2
  sl = slice(0, c) if half == 'trunk' else slice(c, 2 * c)
  W = np.asarray(to_atom_cond_w) @ P[sl]                # (128, 395)
  bias = np.asarray(to_atom_cond_w) @ pb[sl]            # (128,)

  col = lambda n: W[:, slice(*ATOM_COLS[n])]
  # chai's 65 name classes per character -> AF3's 64; 130 elements -> 128
  name = np.concatenate([col('AtomNameOneHot')[:, i * 65:i * 65 + 64]
                         for i in range(4)], axis=1)
  put = lambda n, w: params.__setitem__(f'{scope}/{prefix}{n}',
                                        {'weights': C.t(np.asarray(w))})
  put('embed_ref_pos', col('AtomRefPos') / 10.0)
  put('embed_ref_element', col('AtomRefElement')[:, :128])
  put('embed_ref_charge', col('AtomRefCharge'))
  put('embed_ref_atom_name', name)
  put('embed_ref_mask', col('AtomRefMask') + bias[:, None])

  # the atom PAIR feature: [distogram one-hot(12), inv-sq, mask] -> 16.
  # This one genuinely could not be folded into AF3's offset/distance linears,
  # so the graph grows a chai branch and the weight maps straight across.
  Pp = np.asarray(fe['input_projs.ATOM_PAIR.0.weight'])          # (32, 14)
  pbp = np.asarray(fe['input_projs.ATOM_PAIR.0.bias'])
  cp = Pp.shape[0] // 2
  slp = slice(0, cp) if half == 'trunk' else slice(cp, 2 * cp)
  params[f'{scope}/{prefix}embed_atom_pair_feat'] = {
      'weights': C.t(Pp[slp]), 'bias': pbp[slp]}

  # chai's atom_pair_mlp is two layers (Linear, ReLU, Linear); AF3's third has
  # no counterpart and the chai branch does not create it.
  U = f'{enc_prefix}.pair_update_block'
  put('pair_mlp_1', np.asarray(sds_enc[f'{U}.atom_pair_mlp.0.weight']))
  put('pair_mlp_2', np.asarray(sds_enc[f'{U}.atom_pair_mlp.2.weight']))


def map_input_embedder(sds, params):
  """chai's TOKEN feature projection + the trunk single projection.

  The 2638-wide TOKEN stream collapses, for a de novo protein, to a Linear over
  the residue one-hot plus a constant: everything else is either identically
  zero (ChainIsCropped, ESMEmbeddings, MSADeletionMean, MSAProfile,
  MissingChainContact) or a constant one-hot (IsDistillation 0, TokenBFactor 2,
  TokenPLDDT 3 -- the "not included" sentinel classes, include_prob=0.0). So the
  constant folds into the bias and the residue columns become the weight.

  FOLDING ESM AWAY IS THE PORT'S BIGGEST REMAINING LIMITATION. It is only valid
  because we feed ESMEmbeddings zeros, and that costs almost everything on a
  natural protein: native chai on 1STP+BTN scores 0.642 A with ESM and 4.899 A
  without (our port, 5.697 A). 6MRR is designed and therefore predictable from
  sequence alone, which is why it never showed.

  To support it, do NOT fold the ESM columns into the bias; emit them as their
  own projection instead:
      W[:, slice(*TOKEN_COLS['ESMEmbeddings'])]   -> (384, 2560), transpose
  and add Linear(esm_embeddings) to the token embedding, with a new
  featurise_spec input defaulting to zeros so every existing caller is
  unchanged. The embeddings come from chai's own TRACED TorchScript ESM at
  ~/chai1_weights/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt (5.7 GB, already
  downloaded) -- a self-contained archive that runs under ~/chai_venv exactly
  like the other five, so this needs no transformers/HuggingFace dependency.
  Per-token, 2560-dim, TOKEN feature, can_mask=False.
  """
  fe, te = sds['feature_embedding'], sds['token_embedder']
  W = np.asarray(fe['input_projs.TOKEN.0.weight'])       # (384, 2638)
  b = np.asarray(fe['input_projs.TOKEN.0.bias'])
  col = lambda n: W[:, slice(*TOKEN_COLS[n])]
  const = (b + col('IsDistillation')[:, 0] + col('TokenBFactor')[:, 2]
           + col('TokenPLDDT')[:, 3])
  # ESM gets its OWN projection rather than being folded away. Harmless when
  # unused: the forward only builds this module when esm_embeddings is present,
  # so the name simply has no home in init and is ignored.
  params['diffuser/chai1_esm_embedding'] = {
      'weights': col('ESMEmbeddings').T}
  # MSAProfile + MSADeletionMean get their OWN projection rather than being
  # folded away. Folding them was correct ONLY because chai feeds an all-gap MSA
  # when there is none, making both columns identically zero (verified in its
  # captured features: |x| 0.0000 without an MSA, 0.0162 / 0.0184 with one). The
  # moment a real MSA exists they carry the profile signal, and dropping it is
  # why MSA made our prediction WORSE while it made chai's better.
  # Our profile is 31-wide, so chai's 33 columns come through CHAI1_RESTYPE_PERM;
  # deletion mean is appended as the 32nd input.
  prof = col('MSAProfile')[:, list(CHAI1_RESTYPE_PERM)]        # (384, 31)
  params['diffuser/chai1_msa_profile_embedding'] = {
      'weights': np.concatenate([prof, col('MSADeletionMean')], axis=1).T}
  params['diffuser/chai1_token_feature_embedding'] = {
      'weights': C.t(col('ResidueType')[:, CHAI1_RESTYPE_PERM]),
      'bias': const,
  }
  params['diffuser/chai1_single_proj_in_trunk'] = {
      'weights': C.t(np.asarray(te['token_single_proj_in_trunk.weight']))}

  # AF3 re-projects target_feat into the trunk's single track; chai's s_init IS
  # that track, so the projection is the identity.
  c = np.asarray(te['token_single_proj_in_trunk.weight']).shape[0]
  params['diffuser/evoformer/single_activations'] = {'weights': np.eye(c, dtype=np.float32)}

  # the token embedder's own atom encoder one-offs (the diffusion module's
  # equivalents are handled in map_diffusion_conditioning)
  E = 'token_single_input_emb.atom_encoder'
  params['diffuser/evoformer_conditioning_project_atom_features_for_aggr'] = {
      'weights': C.t(np.asarray(te[f'{E}.to_token_single.0.weight']))}
  U = f'{E}.pair_update_block'
  for af3, chai in (('single_to_pair_cond_row', 'atom_single_to_atom_pair_proj_h'),
                    ('single_to_pair_cond_col', 'atom_single_to_atom_pair_proj_w')):
    w = C.t(np.asarray(te[f'{U}.{chai}.1.weight']))
    params[f'diffuser/evoformer_conditioning_{af3}'] = {'weights': np.asarray(w)}
    params[f'diffuser/evoformer_conditioning_{af3}_1'] = {'weights': np.asarray(w)}


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def map_chai1_to_af3(sds):
  """{component: state_dict} -> {haiku scope: {name: array}}.

  Work in progress: the trunk pairformer stack and the MSA module are mapped;
  the template embedder, token embedder, diffusion module, confidence head and
  distogram head are not yet. The MSA module's grouped OPM emits names that only
  exist once the chai1 forward branch does, so it reads as EXTRA in the gate
  until then.
  """
  params = {}
  map_trunk_pairformer(sds['trunk'], params)
  map_msa_module(sds['trunk'], params)
  if 'feature_embedding' in sds:
    map_msa_input(sds['feature_embedding'], params)
  map_msa_single_to_msa(sds['trunk'], params)
  map_recycling(sds['trunk'], params)
  if 'diffusion_module' in sds:
    dm = sds['diffusion_module']
    map_diffusion_token_transformer(dm, params)
    map_atom_transformer(
        dm, params, 'atom_attention_encoder',
        'diffuser/~/diffusion_head/diffusion_atom_transformer_encoder',
        'diffusion_atom_transformer_encoder')
    map_atom_transformer(
        dm, params, 'atom_attention_decoder',
        'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder',
        'diffusion_atom_transformer_decoder')
    map_diffusion_conditioning(dm, params)
  if 'token_embedder' in sds and 'feature_embedding' in sds:
    map_input_embedder(sds, params)
    map_pair_init(sds, params)
    map_atom_conditioning(
        sds, params,
        sds['token_embedder']['token_single_input_emb.atom_encoder.to_atom_cond.weight'],
        'diffuser', 'evoformer_conditioning_', 'trunk',
        sds_enc=sds['token_embedder'],
        enc_prefix='token_single_input_emb.atom_encoder')
  if 'diffusion_module' in sds and 'feature_embedding' in sds:
    map_atom_conditioning(
        sds, params,
        sds['diffusion_module']['atom_attention_encoder.to_atom_cond.weight'],
        'diffuser/~/diffusion_head', 'diffusion_', 'structure',
        sds_enc=sds['diffusion_module'], enc_prefix='atom_attention_encoder')
  if 'token_embedder' in sds:
    map_atom_transformer(
        sds['token_embedder'], params, 'token_single_input_emb.atom_encoder',
        'diffuser/evoformer_conditioning_atom_transformer_encoder',
        'evoformer_conditioning_atom_transformer_encoder')
  if 'confidence_head' in sds:
    map_confidence_head(sds['confidence_head'], params)
  if 'feature_embedding' in sds:
    map_template_embedder(sds['trunk'], params)
    map_template_features(sds['feature_embedding'], params)
  if 'distogram_head' in sds:
    map_distogram_head(sds['distogram_head'], params)
  return params


def convert_chai1_weights(model_dir, out_dir=None):
  """ensure_weights entry point. `model_dir` is a DIRECTORY (multi-file source)."""
  params = map_chai1_to_af3(load_chai1(model_dir))
  return C.write_params_blob(out_dir or model_dir, 'chai1.bin.zst', params)
