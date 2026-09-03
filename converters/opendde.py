"""Convert OpenDDE (PyTorch) weights into colabdesign2's vendored AF3 haiku layout.

OpenDDE (github aurekaresearch/OpenDDE, Apache-2.0) is an independent PyTorch
reimplementation in the AlphaFold3 family (its own pairformer/primitives, like
Protenix), NOT DeepMind-AF3-arch. So this is an OF3-style port: a per-module weight
converter into our AF3 graph, run with the `model='opendde'` config profile (widths +
heads, see runner.make_config / OPENDDE_SETTINGS), which is in OPENFOLD3_LINEAGE, with
trained_fourier ON.

Conventions
-----------
- PyTorch `nn.Linear.weight` is (out, in); haiku `hm.Linear` param 'weights' is
  (in, out) -- every Linear transposes.  LayerNorm: torch weight/bias -> haiku
  scale/offset.
- haiku params are a FLAT dict keyed by full '/'-path -> {param_name: array}.
- The trunk pairformer / diffusion blocks are collapsed by hk.experimental.layer_stack
  into a single param with a leading block axis; the converter stacks per-block arrays
  along axis 0 in block order (see stack_blocks).

Validation
----------
Each module converter is checked by numerical parity against the real OpenDDE module
run on CPU (its 'torch' triangle path; LayerNorm falls back to F.layer_norm off-CUDA):
convert its state_dict, run our haiku module on the same input, compare. fp32 residual
is ~1e-3 from our LayerNorm's use_fast_variance=True (vs torch's stable two-pass) -- a
known numerical difference, not a mapping error; it vanishes in bf16 inference. A wrong
mapping is orders of magnitude off (e.g. the fused-projection block-vs-interleave layout
below: interleave 1.8e-3, block 4.6).
"""
import numpy as np

from . import common as C

# OpenDDE dialect for the shared primitives: separate a/b tri-mul (interleave-fused),
# block-fused transitions named 'layernorm1'/'linear_no_bias_*', 'linear' pair-bias
# leaf, MSA/single leaves with the OpenDDE spellings, OPM output from linear_out.
DIALECT_OPENDDE = C.Dialect(
    tr_mode='block', tr_ln='layernorm1',
    tr_a='linear_no_bias_a', tr_b='linear_no_bias_b', tr_out='linear_no_bias',
    tm_fused=False, ga_bias='linear',
    sa_ln_a='layernorm_a', sa_ln_z='layernorm_z', sa_z='linear_nobias_z', sa_mha='attention',
    msa_ln_m='layernorm_m', msa_ln_z='layernorm_z', msa_z='linear_no_bias_z',
    msa_v='linear_no_bias_mv', msa_g='linear_no_bias_mg', msa_o='linear_no_bias_out',
    opm_out_direct=False,
)


def _regroup(dst_scope, flat):
  """common's flat {'sub/param' | '::param'} -> OpenDDE's {full_scope: {param: arr}}."""
  out = {}
  for k, v in flat.items():
    if k.startswith('::'):
      out.setdefault(dst_scope, {})[k[2:]] = v
    else:
      sub, name = k.rsplit('/', 1)
      out.setdefault(f'{dst_scope}/{sub}', {})[name] = v
  return out


def T(w):
  """torch Linear weight (out, in) -> haiku (in, out)."""
  return np.asarray(w).T


def fuse_glu(a, b):
  """Fuse two torch Linears (each (out=c, in=c)) into our GLU 'projection'/'gate'
  weight (in=c, 2c). Our TriangleMultiplication/TransitionBlock reads the 2c axis
  as reshape(c, 2) -- an INTERLEAVE (a=even, b=odd), NOT a block [a|b] (validated:
  block is 4.6 off, interleave 1.8e-3)."""
  a, b = np.asarray(a), np.asarray(b)
  cin = a.shape[1]
  w = np.zeros((cin, a.shape[0], 2), np.float32)
  w[:, :, 0] = a.T
  w[:, :, 1] = b.T
  return w.reshape(cin, 2 * a.shape[0])


def _k(prefix, name):
  """join a torch state_dict prefix with a name (prefix='' -> bare name)."""
  return f'{prefix}.{name}' if prefix else name


def fuse_block(a, b):
  """Fuse two torch Linears (each (out, in)) into a BLOCK-layout GLU weight
  (in, 2*out) = [a | b]. TransitionBlock reshapes transition1 as (c, 2, n_int) --
  a block split (first n_int = a, second = b), UNLIKE tri-mul's reshape(c, 2)
  interleave (validated separately per module)."""
  a, b = np.asarray(a), np.asarray(b)
  return np.concatenate([a.T, b.T], axis=1)


def ln(sd, prefix):
  """torch LayerNorm {prefix}.weight/.bias -> haiku {scale, offset}."""
  return {'scale': np.asarray(sd[_k(prefix, 'weight')]),
          'offset': np.asarray(sd[_k(prefix, 'bias')])}


# These module converters now delegate to the shared primitives in common.py (via the
# OpenDDE dialect); the reshape/transpose math is identical across families -- only the
# torch leaf names and fusion mode differ. Each keeps the OpenDDE signature (returning
# {full_haiku_path: {param: array}}, adapted by _regroup) so the block composers and the
# component tests read unchanged.

def convert_tri_mul(sd, src_prefix, dst_scope, outgoing=True):
  """One triangle-multiplication. VALIDATED (outgoing + incoming)."""
  return _regroup(dst_scope, C.triangle_mul(sd, src_prefix, DIALECT_OPENDDE, outgoing=outgoing))


def convert_tri_att(sd, src_prefix, dst_scope):
  """One triangle-attention (start->pair_attention1, end->pair_attention2). VALIDATED."""
  g = lambda n: np.asarray(sd[_k(src_prefix, n)])
  H = g('linear.weight').shape[0]
  D = g('mha.linear_q.weight').shape[0] // H
  return _regroup(dst_scope, C.grid_attention(sd, src_prefix, DIALECT_OPENDDE, H, D))


def convert_transition(sd, src_prefix, dst_scope):
  """OpenDDE Transition -> our TransitionBlock. VALIDATED (block-fused a|b)."""
  return _regroup(dst_scope, C.transition(sd, src_prefix, DIALECT_OPENDDE))


def convert_msa(sd, src_prefix, dst_scope):
  """OpenDDE MSAPairWeightedAveraging -> our MSAAttention. VALIDATED."""
  g = lambda n: np.asarray(sd[_k(src_prefix, n)])
  H = g('linear_no_bias_z.weight').shape[0]
  vd = g('linear_no_bias_mv.weight').shape[0] // H
  return _regroup(dst_scope, C.msa_attention(sd, src_prefix, DIALECT_OPENDDE, H, vd))


def convert_opm(sd, src_prefix, dst_scope):
  """OpenDDE OuterProductMean -> our OuterProductMean. VALIDATED."""
  ch = np.asarray(sd[_k(src_prefix, 'linear_1.weight')]).shape[0]
  return _regroup(dst_scope, C.outer_product_mean(sd, src_prefix, DIALECT_OPENDDE, c_hidden=ch))


def convert_single_attn(sd, src_prefix, dst_scope):
  """OpenDDE AttentionPairBias(has_s=False) -> single-attn + single_pair_logits. VALIDATED."""
  g = lambda n: np.asarray(sd[_k(src_prefix, n)])
  H = g('linear_nobias_z.weight').shape[0]
  D = g('attention.linear_q.weight').shape[0] // H
  return _regroup(dst_scope, C.single_attention(sd, src_prefix, DIALECT_OPENDDE, H, D))


def convert_pairformer_block(sd, dst_scope, with_single=True):
  """Assemble one OpenDDE PairformerBlock -> our PairFormerIteration params. sd keys
  are prefixed per submodule (tri_mul_out/in, tri_att_start/end, pair_transition, and
  when with_single: attention_pair_bias, single_transition). VALIDATED whole-block.
  with_single=False = a pair-only block (c_s=0), e.g. the template pairformer stack."""
  out = {}
  out.update(convert_tri_mul(sd, 'tri_mul_out', f'{dst_scope}/triangle_multiplication_outgoing', outgoing=True))
  out.update(convert_tri_mul(sd, 'tri_mul_in', f'{dst_scope}/triangle_multiplication_incoming', outgoing=False))
  out.update(convert_tri_att(sd, 'tri_att_start', f'{dst_scope}/pair_attention1'))
  out.update(convert_tri_att(sd, 'tri_att_end', f'{dst_scope}/pair_attention2'))
  out.update(convert_transition(sd, 'pair_transition', f'{dst_scope}/pair_transition'))
  if with_single:
    out.update(convert_single_attn(sd, 'attention_pair_bias', dst_scope))
    out.update(convert_transition(sd, 'single_transition', f'{dst_scope}/single_transition'))
  return out


def convert_evoformer_block(sd, dst_scope):
  """One OpenDDE MSABlock -> our EvoformerIteration (the MSA stack). Reuses the
  validated pair/msa/opm converters; everything sits under an extra 'msa_stack/'
  namespace. Boltz op-order (MSA update then OPM) is a forward-graph branch, not a
  weight difference, so the converter is order-agnostic."""
  b = f'{dst_scope}/msa_stack'
  out = {}
  out.update(convert_msa(sd, 'msa_stack.msa_pair_weighted_averaging', f'{b}/msa_attention1'))
  out.update(convert_transition(sd, 'msa_stack.transition_m', f'{b}/msa_transition'))
  out.update(convert_opm(sd, 'outer_product_mean_msa', f'{b}/outer_product_mean'))
  out.update(convert_tri_mul(sd, 'pair_stack.tri_mul_out', f'{b}/triangle_multiplication_outgoing', outgoing=True))
  out.update(convert_tri_mul(sd, 'pair_stack.tri_mul_in', f'{b}/triangle_multiplication_incoming', outgoing=False))
  out.update(convert_tri_att(sd, 'pair_stack.tri_att_start', f'{b}/pair_attention1'))
  out.update(convert_tri_att(sd, 'pair_stack.tri_att_end', f'{b}/pair_attention2'))
  out.update(convert_transition(sd, 'pair_stack.pair_transition', f'{b}/pair_transition'))
  return out


def convert_atom_encoder(sd, src_prefix, dst_prefix):
  """OpenDDE AtomAttentionEncoder single/pair conditioning linears -> our
  evoformer_conditioning_* flat-named params (the atom-attention encoder inside the
  InputFeatureEmbedder). Non-block only (the atom_transformer converts separately).

  The fused linear_no_bias_f (128,385) splits by size into ref_mask(1)+ref_element
  (128)+ref_atom_name(256) (unambiguous).

  CRITICAL (from data-flow, atom_cross_attention.py:144): our encoder calls
  _per_atom_conditioning and DISCARDS its pair output (`, _ =`), then RE-embeds the
  pair conditioning itself -> those recomputed linears get haiku's _1 suffix and are
  the ones ACTUALLY USED. So OpenDDE's single pair set (cl/cm/d/invd) maps to the
  _1 names; the base single_to_pair_cond_row/col + embed_pair_offsets/distances are
  DEAD params (their output is discarded) and are zero-filled just to satisfy loading.
  """
  g = lambda n: np.asarray(sd[_k(src_prefix, n)])
  f = g('linear_no_bias_f.weight')                 # (128, 1+128+256)
  cl, cm = g('linear_no_bias_cl.weight'), g('linear_no_bias_cm.weight')
  d, invd = g('linear_no_bias_d.weight'), g('linear_no_bias_invd.weight')
  P = dst_prefix
  out = {
      f'{P}_embed_ref_pos':    {'weights': T(g('linear_no_bias_ref_pos.weight'))},
      f'{P}_embed_ref_charge': {'weights': T(g('linear_no_bias_ref_charge.weight'))},
      f'{P}_embed_ref_mask':      {'weights': T(f[:, 0:1])},
      f'{P}_embed_ref_element':   {'weights': C.fold_element_index_shift(T(f[:, 1:129]))},
      f'{P}_embed_ref_atom_name': {'weights': T(f[:, 129:385])},
      # the USED pair conditioning is the _1 set (see docstring)
      f'{P}_single_to_pair_cond_row_1': {'weights': T(cl)},
      f'{P}_single_to_pair_cond_col_1': {'weights': T(cm)},
      f'{P}_embed_pair_offsets_1':      {'weights': T(d)},
      f'{P}_embed_pair_distances_1':    {'weights': T(invd)},
      f'{P}_embed_pair_offsets_valid':  {'weights': T(g('linear_no_bias_v.weight'))},
      f'{P}_pair_mlp_1': {'weights': T(g('small_mlp.1.weight'))},
      f'{P}_pair_mlp_2': {'weights': T(g('small_mlp.3.weight'))},
      f'{P}_pair_mlp_3': {'weights': T(g('small_mlp.5.weight'))},
      f'{P}_project_atom_features_for_aggr': {'weights': T(g('linear_no_bias_q.weight'))},
      # DEAD base pair linears (output discarded at _per_atom_conditioning caller) --
      # zero-filled so the param set loads; they do not affect the forward.
      f'{P}_single_to_pair_cond_row': {'weights': np.zeros_like(T(cl))},
      f'{P}_single_to_pair_cond_col': {'weights': np.zeros_like(T(cm))},
      f'{P}_embed_pair_offsets':      {'weights': np.zeros_like(T(d))},
      f'{P}_embed_pair_distances':    {'weights': np.zeros_like(T(invd))},
  }
  return out


def _adaln(sd, src, dst):
  """OpenDDE AdaptiveLayerNorm (layernorm_s + linear_s scale-gate + linear_nobias_s
  bias) -> our adaptive_layernorm params ({dst}single_cond_layer_norm/scale/bias).
  Our forward: sigmoid(single_cond_scale(LN_s(s)))*x + single_cond_bias(LN_s(s))."""
  g = lambda n: np.asarray(sd[_k(src, n)])
  return {
      f'{dst}single_cond_layer_norm': {'scale': g('layernorm_s.weight')},
      f'{dst}single_cond_scale': {'weights': T(g('linear_s.weight')),
                                  'bias': g('linear_s.bias')},
      f'{dst}single_cond_bias': {'weights': T(g('linear_nobias_s.weight'))},
  }


def convert_atom_transformer_block(sd, name_prefix):
  """One OpenDDE atom-transformer block (cross-attn DiffusionTransformer + adaLN) ->
  our CrossAttTransformer block params (relative to the layer_stack scope). Reuses the
  self_attention q/k/v (c,H,dh) reshape + block-GLU + adaLN patterns already validated.
  Emits the of3-style per-block pair LN/projection under bare names (opendde branch)."""
  P = name_prefix                                  # 'evoformer_conditioning_atom_transformer_encoder'
  ab = 'attention_pair_bias'
  g = lambda n: np.asarray(sd[_k(ab, n)])
  lq = g('attention.linear_q.weight')
  H = g('linear_nobias_z.weight').shape[0]         # (H, c_z)
  dh = lq.shape[0] // H
  c = lq.shape[1]
  proj = lambda w: np.asarray(w).reshape(H, dh, c).transpose(2, 0, 1)   # (c,H,dh)
  out = {
      # of3-style per-block pair LN + projection (bare names, opendde branch)
      'pair_input_layer_norm': {'scale': g('layernorm_z.weight')},
      'pair_logits_projection': {'weights': T(g('linear_nobias_z.weight'))},
      # cross-attention (self.name-prefixed)
      f'{P}q_projection': {'weights': proj(lq),
                           'bias': g('attention.linear_q.bias').reshape(H, dh)},
      f'{P}k_projection': {'weights': proj(g('attention.linear_k.weight'))},
      f'{P}v_projection': {'weights': proj(g('attention.linear_v.weight'))},
      f'{P}gating_query': {'weights': T(g('attention.linear_g.weight'))},
      f'{P}transition2':  {'weights': T(g('attention.linear_o.weight'))},
      f'{P}adaptive_zero_cond': {'weights': T(g('linear_a_last.weight')),
                                 'bias': g('linear_a_last.bias')},
  }
  out.update(_adaln(sd, _k(ab, 'layernorm_a'),  f'{P}q'))    # queries adaLN
  out.update(_adaln(sd, _k(ab, 'layernorm_kv'), f'{P}k'))    # keys adaLN
  # conditioned transition (ffw)
  ct = 'conditioned_transition_block'
  gc_ = lambda n: np.asarray(sd[_k(ct, n)])
  out.update(_adaln(sd, _k(ct, 'adaln'), f'{P}ffw_'))
  out[f'{P}ffw_transition1'] = {'weights': fuse_block(gc_('linear_nobias_a1.weight'),
                                                      gc_('linear_nobias_a2.weight'))}
  out[f'{P}ffw_transition2'] = {'weights': T(gc_('linear_nobias_b.weight'))}
  out[f'{P}ffw_adaptive_zero_cond'] = {'weights': T(gc_('linear_s.weight')),
                                       'bias': gc_('linear_s.bias')}
  return out


# OpenDDE residue vocab (STD_RESIDUES_WITH_GAP, 32) -> ours
# (POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, 31). Protein ALA..UNK (0-20) align
# identically; nucleic acids reorder; OpenDDE's DNA-unknown 'DN'(30) has no ours slot
# (dropped). RESTYPE_PERM[our_idx] = opendde_idx, so a one-hot weight column reorders
# W_ours[:, our] = W_opendde[:, opendde]. (Protein-only design touches just 0-20.)
RESTYPE_PERM = list(range(21)) + [31, 21, 22, 23, 24, 26, 27, 28, 29, 25]


def _restype_cols(w, start):
  """select the 31 ours-order columns from a 32-wide restype/profile block of an
  OpenDDE (out, in) weight starting at column `start`."""
  return np.asarray(w)[:, [start + i for i in RESTYPE_PERM]]


def _s_inputs_adapter(w):
  """adapt an OpenDDE linear over s_inputs (out, 449) with order [a(384), restype(32),
  profile(32), del(1)] to our target_feat order [restype(31), profile(31), del(1),
  token_act(384)] (out, 447), remapping the 32->31 residue vocab. Returns our (in=447,
  out) haiku weight. Shared by single_activations (sinit) + extra_msa_target_feat."""
  w = np.asarray(w)
  restype = _restype_cols(w, 384)                   # (out, 31)
  profile = _restype_cols(w, 416)                   # (out, 31)
  deletion = w[:, 448:449]                          # (out, 1)
  a = w[:, 0:384]                                   # (out, 384)
  return np.concatenate([restype, profile, deletion, a], axis=1).T   # (447, out)


def convert_residue_features(sd):
  """OpenDDE loose residue-level linears -> our evoformer loose params.

  single_activations (sinit) is a FEATURE ADAPTER via _s_inputs_adapter (block reorder
  + 32->31 vocab remap). Recycle/bond/rel-pos linears are plain .T. NOTE: left/right_
  single (pair init) are NOT here -- OpenDDE builds z_init from s_init(384), our graph
  from target_feat(447); that needs an opendde forward branch, handled separately.
  """
  g = lambda n: np.asarray(sd[n])
  return {
      'single_activations': {'weights': _s_inputs_adapter(g('linear_no_bias_sinit.weight'))},
      # pair init from s_init(384) (opendde _seq_pair_embedding branch): plain .T
      'left_single':  {'weights': T(g('linear_no_bias_zinit1.weight'))},
      'right_single': {'weights': T(g('linear_no_bias_zinit2.weight'))},
      'bond_embedding': {'weights': T(g('linear_no_bias_token_bond.weight'))},
      'prev_embedding': {'weights': T(g('linear_no_bias_z_cycle.weight'))},
      'prev_single_embedding': {'weights': T(g('linear_no_bias_s.weight'))},
      'prev_embedding_layer_norm': ln(sd, 'layernorm_z_cycle'),
      'prev_single_embedding_layer_norm': ln(sd, 'layernorm_s'),
      '~_relative_encoding/position_activations':
          {'weights': T(g('relative_position_encoding.linear_no_bias.weight'))},
  }


def _remap_s_inputs_vec(v):
  """reorder a length-449 s_inputs vector [a(384), restype(32), profile(32), del(1)]
  to ours [restype(31), profile(31), del(1), a(384)] (449->447), remapping the 32->31
  vocab. For 1D features (e.g. a LayerNorm scale)."""
  v = np.asarray(v)
  return np.concatenate([v[384:416][RESTYPE_PERM], v[416:448][RESTYPE_PERM],
                         v[448:449], v[0:384]])


def convert_diffusion_conditioning(sd):
  """OpenDDE diffusion_module.diffusion_conditioning + loose -> our diffusion_head
  conditioning params. Pair side is clean (.T); the diffusion relpe branch (added to
  diffusion_head) matches OpenDDE's separate compress+concat. single_cond_initial uses
  the s_inputs adapter-combine (s_trunk identity + s_inputs 449->447). Fourier folds in
  like IF2 (trained_fourier). NOTE: single_cond_initial_norm LN spans 831 (ours) vs 833
  (OpenDDE); the 2 dropped vocab features are 0 for protein so the normalization count
  differs only slightly (minor; confirm via e2e)."""
  C = 'diffusion_module.diffusion_conditioning'
  g = lambda n: np.asarray(sd[_k(C, n)])
  # single_cond_initial adapter-combine (833 -> 831): [s_trunk(384) | s_inputs(449->447)]
  scale = g('layernorm_s.weight')                              # (833,)
  sci_scale = np.concatenate([scale[0:384], _remap_s_inputs_vec(scale[384:833])])
  sw = g('linear_no_bias_s.weight')                            # (384, 833) out,in
  sci_w = np.concatenate([sw[:, 0:384],
                          _s_inputs_adapter(sw[:, 384:833]).T], axis=1)   # (384, 831)

  def tr(src, dst):   # OpenDDE Transition -> our transition_block(single_cond=None) ffw_
    gg = lambda n: np.asarray(sd[_k(_k(C, src), n)])
    return {f'{dst}ffw_layer_norm': {'scale': gg('layernorm1.weight'),
                                     'offset': gg('layernorm1.bias')},
            f'{dst}ffw_transition1': {'weights': fuse_block(gg('linear_no_bias_a.weight'),
                                                            gg('linear_no_bias_b.weight'))},
            f'{dst}ffw_transition2': {'weights': T(gg('linear_no_bias.weight'))}}

  DH = 'diffuser/~/diffusion_head/'
  out = {
      DH + 'z_trunk_norm': {'scale': g('layernorm_z_trunk.weight')},
      DH + 'z_trunk_projection': {'weights': T(g('linear_no_bias_z_trunk.weight'))},
      DH + 'relpe_projection': {'weights': T(g('relpe.linear_no_bias.weight'))},
      DH + 'pair_cond_initial_norm': {'scale': g('layernorm_z.weight')},
      DH + 'pair_cond_initial_projection': {'weights': T(g('linear_no_bias_z.weight'))},
      DH + 'single_cond_initial_norm': {'scale': sci_scale},
      DH + 'single_cond_initial_projection': {'weights': sci_w.T},
      DH + 'noise_embedding_initial_norm': {'scale': g('layernorm_n.weight')},
      DH + 'noise_embedding_initial_projection': {'weights': T(g('linear_no_bias_n.weight'))},
      # trained Fourier folded in at the diffusion_head scope (trained_fourier reads
      # fourier_embedding_weight/bias via hk.get_parameter there; like IF2)
      DH.rstrip('/'): {'fourier_embedding_weight': np.ravel(g('fourier_embedding.w')),
                       'fourier_embedding_bias': np.ravel(g('fourier_embedding.b'))},
      # loose diffusion single-cond embedding + output norm
      DH + 'single_cond_embedding_norm': {'scale': np.asarray(sd['diffusion_module.layernorm_s.weight'])},
      DH + 'single_cond_embedding_projection': {'weights': T(np.asarray(sd['diffusion_module.linear_no_bias_s.weight']))},
      DH + 'output_norm': {'scale': np.asarray(sd['diffusion_module.layernorm_a.weight'])},
  }
  for src, dst in (('transition_z1', 'pair_transition_0'), ('transition_z2', 'pair_transition_1'),
                   ('transition_s1', 'single_transition_0'), ('transition_s2', 'single_transition_1')):
    out.update({DH + k: v for k, v in tr(src, dst).items()})
  return out


def convert_diffusion_atom_encoder(sd):
  """OpenDDE diffusion atom encoder (has_coords) + decoder non-block linears -> our
  diffusion_* params. Reuses convert_atom_encoder for the shared single/pair-cond (base
  +_1+dead + project + ref_*), then adds the has_coords/trunk-cond extras + decoder."""
  P = 'diffusion'
  ae = 'diffusion_module.atom_attention_encoder'
  aesd = {k[len(ae) + 1:]: v for k, v in sd.items()
          if k.startswith(ae + '.') and 'atom_transformer' not in k}
  out = dict(convert_atom_encoder(aesd, '', P))
  g = lambda n: np.asarray(aesd[n])
  out.update({
      f'{P}_atom_positions_to_features': {'weights': T(g('linear_no_bias_r.weight'))},
      f'{P}_lnorm_trunk_single_cond': {'scale': g('layernorm_s.weight')},
      f'{P}_embed_trunk_single_cond': {'weights': T(g('linear_no_bias_s.weight'))},
      f'{P}_lnorm_trunk_pair_cond': {'scale': g('layernorm_z.weight')},
      f'{P}_embed_trunk_pair_cond': {'weights': T(g('linear_no_bias_z.weight'))},
  })
  de = 'diffusion_module.atom_attention_decoder'
  dg = lambda n: np.asarray(sd[f'{de}.{n}'])
  out.update({
      f'{P}_project_token_features_for_broadcast': {'weights': T(dg('linear_no_bias_a.weight'))},
      f'{P}_atom_features_layer_norm': {'scale': dg('layernorm_q.weight')},
      f'{P}_atom_features_to_position_update': {'weights': T(dg('linear_no_bias_out.weight'))},
  })
  return out


def convert_structural_token_expander(ck, scope='structural_token_expander',
                                      remap_s_inputs=False):
  """OpenDDE StructuralTokenExpander -> our haiku params (flat {scope: {name: arr}}).

  pair_projection_mode='full' (49 = 7x7 per-role-pair linears). Embedding tables and
  learned scalars copy straight through; the split-MLP + pair projections transpose
  (torch (out,in) -> haiku (in,out)). `scope` is the module's haiku path; the standalone
  module is 'structural_token_expander', but when embedded it will be prefixed.
  Module-scope params (embeddings, pair_block_proj, attn_bias_*) sit under `scope`;
  the split-MLP linears/LN sit under `scope/single_split_*`.

  remap_s_inputs: OpenDDE's single_input_role_embedding is in the 449-dim s_inputs order
  [a(384),restype32,profile32,del]; our diffusion consumes our 447-dim target_feat order
  [restype31,profile31,del,a]. When True, remap the embedding rows to 447 so the expander
  runs on our target_feat (s_inputs_res = target_feat) and produces target_feat_struct.
  """
  P = 'structural_token_expander.'
  g = lambda n: np.asarray(ck[P + n])
  proj = np.stack([T(g(f'pair_block_proj.{k}.weight')) for k in range(49)], axis=0)
  sire = g('single_input_role_embedding.weight')      # (7, 449)
  if remap_s_inputs:
    sire = np.stack([_remap_s_inputs_vec(row) for row in sire], axis=0)   # (7, 447)
  return {
      scope: {
          'single_input_role_embedding': sire,
          'single_role_embedding': g('single_role_embedding.weight'),
          'same_parent_embedding': g('same_parent_embedding.weight'),
          'same_residue_twin_embedding': g('same_residue_twin_embedding.weight'),
          'prev_bb_chain_embedding': g('prev_bb_chain_embedding.weight'),
          'next_bb_chain_embedding': g('next_bb_chain_embedding.weight'),
          'role_pair_type_embedding': g('role_pair_type_embedding.weight'),
          'pair_block_proj': proj,
          'attn_bias_same_parent': np.asarray(g('attn_bias_same_parent')),
          'attn_bias_same_residue_twin': np.asarray(g('attn_bias_same_residue_twin')),
          'attn_bias_prev_bb_chain': np.asarray(g('attn_bias_prev_bb_chain')),
          'attn_bias_next_bb_chain': np.asarray(g('attn_bias_next_bb_chain')),
          'attn_bias_role_pair_type': np.asarray(g('attn_bias_role_pair_type')),
      },
      f'{scope}/single_split_norm': {'scale': g('single_split_mlp.0.weight'),
                                     'offset': g('single_split_mlp.0.bias')},
      f'{scope}/single_split_1': {'weights': T(g('single_split_mlp.1.weight'))},
      f'{scope}/single_split_2': {'weights': T(g('single_split_mlp.3.weight'))},
  }


def convert_opendde_confidence(ck, scope='confidence_head', n_blocks=4):
  """OpenDDE confidence head -> our haiku params (flat {scope: {name: arr}}).

  s1/s2 project the 449-dim s_inputs -> c_z; remapped to our 447 order via the same
  s_inputs adapter as extra_msa_target_feat. The 4-block pairformer reuses
  convert_pairformer_block. plddt/resolved weight tensors [24,c_s,bins] copy straight
  through (einsum, no transpose). PAE/PDE/d linears transpose."""
  if not all(isinstance(v, np.ndarray) for v in ck.values()):
    ck = _strip_module(ck)
  P = 'confidence_head.'
  g = lambda n: np.asarray(ck[P + n])
  out = {
      scope: {
          'plddt_weight': g('plddt_weight'),
          'resolved_weight': g('resolved_weight'),
      },
      f'{scope}/input_strunk_ln': ln(ck, P + 'input_strunk_ln'),
      f'{scope}/linear_no_bias_s1': {'weights': _s_inputs_adapter(g('linear_no_bias_s1.weight'))},
      f'{scope}/linear_no_bias_s2': {'weights': _s_inputs_adapter(g('linear_no_bias_s2.weight'))},
      f'{scope}/linear_no_bias_d': {'weights': T(g('linear_no_bias_d.weight'))},
      f'{scope}/linear_no_bias_d_wo_onehot': {'weights': T(g('linear_no_bias_d_wo_onehot.weight'))},
      f'{scope}/pae_ln': ln(ck, P + 'pae_ln'),
      f'{scope}/pde_ln': ln(ck, P + 'pde_ln'),
      f'{scope}/plddt_ln': ln(ck, P + 'plddt_ln'),
      f'{scope}/resolved_ln': ln(ck, P + 'resolved_ln'),
      f'{scope}/linear_no_bias_pae': {'weights': T(g('linear_no_bias_pae.weight'))},
      f'{scope}/linear_no_bias_pde': {'weights': T(g('linear_no_bias_pde.weight'))},
  }
  # 4-block pairformer (identical block to the trunk pairformer).
  per = []
  for i in range(n_blocks):
    p = f'{P}pairformer_stack.blocks.{i}.'
    per.append(convert_pairformer_block(
        {k[len(p):]: v for k, v in ck.items() if k.startswith(p)}, dst_scope=''))
  stacked = {k.lstrip('/'): v for k, v in stack_blocks(per).items()}
  pf_scope = f'{scope}/pairformer_stack/trunk_pairformer'
  out.update({f'{pf_scope}/{k}': v for k, v in stacked.items()})
  return out


def convert_structural_token_refiner(ck, n_blocks=4, scope='structural_token_refiner'):
  """OpenDDE structural_token_refiner (a PairformerStack on structural tokens) ->
  our haiku params. Block structure is IDENTICAL to the trunk pairformer, so each
  block reuses convert_pairformer_block; the n_blocks blocks are collapsed onto the
  layer_stack leading axis via stack_blocks. `scope` is the haiku layer_stack path."""
  if not all(isinstance(v, np.ndarray) for v in ck.values()):
    ck = _strip_module(ck)
  per = []
  for i in range(n_blocks):
    p = f'structural_token_refiner.blocks.{i}.'
    per.append(convert_pairformer_block(
        {k[len(p):]: v for k, v in ck.items() if k.startswith(p)}, dst_scope=''))
  stacked = {k.lstrip('/'): v for k, v in stack_blocks(per).items()}
  return {f'{scope}/{k}': v for k, v in stacked.items()}


def convert_msa_init(sd):
  """OpenDDE msa_module input projections -> our evoformer msa-init loose params.
  linear_no_bias_m (c_m, 34) -> msa_activations; linear_no_bias_s (c_m, 449) ->
  extra_msa_target_feat (447, c_m) via the s_inputs adapter (same 32->31 vocab +
  order remap as sinit).

  linear_no_bias_m's input is [restype one-hot(32), has_deletion, deletion_value],
  so its 32-class block is restype-indexed and takes the SAME RESTYPE_PERM as
  everything else here. It used to be a plain .T. That is silent -- 34 rows either
  way, and protein indices (0-20) align between the two vocabularies -- so every
  protein and ligand gate passed while nucleic acids read the wrong classes and
  MSA gaps were embedded as an RNA base. The identical omission was found in
  of3, protenix2 and rf3; opendde was the fourth.
  """
  # AF3's MSA one-hot is 32 classes, not the 31 of target_feat: the extra class
  # (our index 31) is OpenDDE's DN(30), which RESTYPE_PERM drops. Same +[30] tail
  # of3's _AF3_TO_OF3_MSA appends for exactly this reason.
  m = np.asarray(sd['msa_module.linear_no_bias_m.weight'])          # (c_m, 34)
  perm = list(RESTYPE_PERM) + [30]                                  # 32 classes
  m = np.concatenate([m[:, perm], m[:, 32:]], axis=1)               # (c_m, 34)
  return {
      'msa_activations': {'weights': T(m)},
      'extra_msa_target_feat':
          {'weights': _s_inputs_adapter(sd['msa_module.linear_no_bias_s.weight'])},
  }


def _strip_module(ck):
  """raw opendde.pt state_dict (ck['model']) -> {name: fp32 ndarray}, DDP prefix off."""
  return {(k[7:] if k.startswith('module.') else k):
          (v.detach().cpu().float().numpy() if hasattr(v, 'detach') else np.asarray(v))
          for k, v in ck.items()}


def convert_opendde_trunk(ck):
  """Assemble the FULL trunk->distogram haiku param dict from an OpenDDE state_dict
  (ck = the 'module.'-prefixed torch dict, or already-stripped). Covers the input
  embedder (atom encoder + atom transformer), evoformer (residue/msa init, trunk
  pairformer x48, MSA stack x4, template embedder + its pairformer x2), rel-pos, and
  the distogram head -- i.e. everything the structure=False model needs. Diffusion +
  confidence heads are separate (Phases 3/4). VALIDATED shape-exact (213 params / 173
  scopes) on the real checkpoint."""
  if not all(isinstance(v, np.ndarray) for v in ck.values()):
    ck = _strip_module(ck)                            # torch tensors -> stripped fp32 numpy

  def blk(prefix, n, conv, **kw):
    per = []
    for i in range(n):
      p = f'{prefix}.blocks.{i}.'
      per.append(conv({k[len(p):]: v for k, v in ck.items() if k.startswith(p)},
                      dst_scope='', **kw))
    return {k.lstrip('/'): v for k, v in stack_blocks(per).items()}

  def pref(d, p):
    return {p + k: v for k, v in d.items()}

  out = {}
  E = 'diffuser/evoformer/'
  out.update(pref(convert_residue_features(ck), E))
  out.update(pref(convert_msa_init(ck), E))
  out.update(pref(blk('pairformer_stack', 48, convert_pairformer_block),
                  E + '__layer_stack_no_per_layer_1/trunk_pairformer/'))
  out.update(pref(blk('msa_module', 4, convert_evoformer_block),
                  E + '__layer_stack_no_per_layer/'))
  out.update(pref(blk('template_embedder.pairformer_stack', 2,
                      convert_pairformer_block, with_single=False),
                  E + 'template_embedding/single_template_embedding/'
                  '__layer_stack_no_per_layer/template_embedding_iteration/'))
  ae = 'input_embedder.atom_attention_encoder'
  aesd = {k[len(ae) + 1:]: v for k, v in ck.items()
          if k.startswith(ae + '.') and 'atom_transformer' not in k}
  out.update(pref(convert_atom_encoder(aesd, '', 'evoformer_conditioning'), 'diffuser/'))
  atp = f'{ae}.atom_transformer.diffusion_transformer'
  per = []
  for i in range(3):
    p = f'{atp}.blocks.{i}.'
    per.append(convert_atom_transformer_block(
        {k[len(p):]: v for k, v in ck.items() if k.startswith(p)},
        'evoformer_conditioning_atom_transformer_encoder'))
  out.update(pref(stack_blocks(per),
                  'diffuser/evoformer_conditioning_atom_transformer_encoder/'
                  '__layer_stack_no_per_layer/'))
  out.update(pref(convert_template(ck), 'diffuser/'))
  out.update(pref(convert_distogram(ck), 'diffuser/'))
  return out


def convert_opendde(ck):
  """Assemble the FULL OpenDDE model haiku param dict (trunk->distogram + diffusion)
  from a state_dict. Everything except the confidence head (pLDDT/PAE; separate new
  code). VALIDATED shape-exact (343 params) on the real checkpoint. Load with
  AF3Runner(..., opendde config); a structure-predicting OpenDDE in our graph."""
  if not all(isinstance(v, np.ndarray) for v in ck.values()):
    ck = _strip_module(ck)
  DH = 'diffuser/~/diffusion_head/'
  pref = lambda d, p: {p + k: v for k, v in d.items()}

  def blk(prefix, n, conv, **kw):
    per = []
    for i in range(n):
      p = f'{prefix}.blocks.{i}.'
      s = {k[len(p):]: v for k, v in ck.items() if k.startswith(p)}
      per.append(conv(s, **kw))
    return per

  out = convert_opendde_trunk(ck)
  out.update(convert_diffusion_conditioning(ck))
  out.update(pref(convert_diffusion_atom_encoder(ck), DH))
  # main diffusion transformer (24 self-attn blocks -> 6x4 super-stack)
  main = blk('diffusion_module.diffusion_transformer', 24,
             convert_diffusion_transformer_block, name_prefix='transformer')
  out.update(pref(stack_super(main, 4),
                  DH + 'transformer/__layer_stack_no_per_layer/__layer_stack_no_per_layer/'))
  # diffusion atom encoder/decoder cross-att transformers (3 blocks each)
  for x in ('encoder', 'decoder'):
    P = f'diffusion_atom_transformer_{x}'
    per = blk(f'diffusion_module.atom_attention_{x}.atom_transformer.diffusion_transformer',
              3, convert_atom_transformer_block, name_prefix=P)
    out.update(pref(stack_blocks(per), f'{DH}{P}/__layer_stack_no_per_layer/'))
  # OpenDDE structural-token expansion (opendde-only; the Model instantiates these
  # on the has_structural path). Expander in our target_feat order (remap 449->447);
  # refiner = 4-block pairformer reusing convert_pairformer_block.
  if any(k.startswith('structural_token_expander.') for k in ck):
    out.update(convert_structural_token_expander(
        ck, scope='diffuser/structural_token_expander', remap_s_inputs=True))
    out.update(convert_structural_token_refiner(
        ck, n_blocks=4, scope='diffuser/structural_token_refiner/trunk_pairformer'))
  if any(k.startswith('confidence_head.') for k in ck):
    out.update(convert_opendde_confidence(ck, scope='diffuser/confidence_head'))
  return out


def convert_distogram(sd, src_prefix='distogram_head', dst_scope='distogram_head'):
  """OpenDDE DistogramHead.linear (num_bins, c_z)+bias -> our half_logits (c_z, num_bins).

  The bias maps straight across. OpenDDE symmetrises AFTER the linear
  (`logits = self.linear(z); logits + logits.transpose(-2, -3)`, head.py:44),
  which is what our graph does, so it is doubled on both sides. This USED to be
  dropped, because stock AF3's half_logits is bias-free -- see
  model_config.DISTOGRAM_BIAS, which now gates the parameter into existence."""
  return {f'{dst_scope}/half_logits':
          {'weights': T(np.asarray(sd[_k(src_prefix, 'linear.weight')])),
           'bias': np.asarray(sd[_k(src_prefix, 'linear.bias')])}}


def convert_template(sd, src_prefix='template_embedder',
                     dst='evoformer/template_embedding'):
  """OpenDDE TemplateEmbedder -> our SingleTemplateEmbedding + output. The fused
  linear_no_bias_a (64,108) splits per-feature into AF3's 9 template_pair_embedding_*:
  [dgram(39), pbmask(1), restype_col(31<-32 vocab), restype_row(31<-32), uvec x/y/z(1
  each), bbmask(1)], and linear_no_bias_z (query) -> template_pair_embedding_8. restype
  cols map SEMANTICALLY: AF3 _2 = aatype[None] (column/j) <- OpenDDE restype_j; _3 =
  aatype[:,None] (row/i) <- restype_i. OpenDDE's single_template_forward (pairformer.py
  concats [dgram, pbmask, restype_j (expand dim=-3), restype_i (expand dim=-2), uvec,
  bbmask]) so restype_j = cols 40:72 and restype_i = cols 72:104 (verified against source;
  swapping these is what makes the template functional -- 5CAJ fold 18 A -> 1.1 A). Scalars
  (num_input_dims=0) are 1D (64,) weights = the single OpenDDE column."""
  g = lambda n: np.asarray(sd[_k(src_prefix, n)])
  a = g('linear_no_bias_a.weight')                  # (64, 108)
  ste = f'{dst}/single_template_embedding'
  return {
      f'{ste}/template_pair_embedding_0': {'weights': T(a[:, 0:39])},        # dgram
      f'{ste}/template_pair_embedding_1': {'weights': a[:, 39]},             # pbmask
      f'{ste}/template_pair_embedding_2': {'weights': T(a[:, 40:72][:, RESTYPE_PERM])},   # col/j <- restype_j (OpenDDE cols 40:72)
      f'{ste}/template_pair_embedding_3': {'weights': T(a[:, 72:104][:, RESTYPE_PERM])},  # row/i <- restype_i (OpenDDE cols 72:104)
      f'{ste}/template_pair_embedding_4': {'weights': a[:, 104]},            # uvec x
      f'{ste}/template_pair_embedding_5': {'weights': a[:, 105]},            # uvec y
      f'{ste}/template_pair_embedding_6': {'weights': a[:, 106]},            # uvec z
      f'{ste}/template_pair_embedding_7': {'weights': a[:, 107]},            # bbmask
      f'{ste}/template_pair_embedding_8': {'weights': T(g('linear_no_bias_z.weight'))},   # query
      f'{ste}/query_embedding_norm': {'scale': g('layernorm_z.weight'),
                                      'offset': g('layernorm_z.bias')},
      f'{ste}/output_layer_norm': {'scale': g('layernorm_v.weight'),
                                   'offset': g('layernorm_v.bias')},
      f'{dst}/output_linear': {'weights': T(g('linear_no_bias_u.weight'))},
  }


def convert_diffusion_transformer_block(sd, name_prefix):
  """One OpenDDE diffusion_transformer block (SELF-attn + adaLN) -> our Transformer
  block params. Like convert_atom_transformer_block but self-attention: a SINGLE adaLN
  (layernorm_a -> {P}single_cond_*, no q/k split). c=768, 16 heads x 48. of3 per-block
  pair (layernorm_z/linear_nobias_z -> pair_input_layer_norm/pair_logits_projection)."""
  P = name_prefix
  ab = 'attention_pair_bias'
  g = lambda n: np.asarray(sd[_k(ab, n)])
  lq = g('attention.linear_q.weight')
  H = g('linear_nobias_z.weight').shape[0]
  dh = lq.shape[0] // H
  c = lq.shape[1]
  proj = lambda w: np.asarray(w).reshape(H, dh, c).transpose(2, 0, 1)
  out = {
      'pair_input_layer_norm': {'scale': g('layernorm_z.weight')},
      'pair_logits_projection': {'weights': T(g('linear_nobias_z.weight'))},
      f'{P}q_projection': {'weights': proj(lq),
                           'bias': g('attention.linear_q.bias').reshape(H, dh)},
      f'{P}k_projection': {'weights': proj(g('attention.linear_k.weight'))},
      f'{P}v_projection': {'weights': proj(g('attention.linear_v.weight'))},
      f'{P}gating_query': {'weights': T(g('attention.linear_g.weight'))},
      f'{P}transition2':  {'weights': T(g('attention.linear_o.weight'))},
      f'{P}adaptive_zero_cond': {'weights': T(g('linear_a_last.weight')),
                                 'bias': g('linear_a_last.bias')},
  }
  out.update(_adaln(sd, _k(ab, 'layernorm_a'), P))          # single adaLN (self-attn)
  ct = 'conditioned_transition_block'
  gc_ = lambda n: np.asarray(sd[_k(ct, n)])
  out.update(_adaln(sd, _k(ct, 'adaln'), f'{P}ffw_'))
  out[f'{P}ffw_transition1'] = {'weights': fuse_block(gc_('linear_nobias_a1.weight'),
                                                      gc_('linear_nobias_a2.weight'))}
  out[f'{P}ffw_transition2'] = {'weights': T(gc_('linear_nobias_b.weight'))}
  out[f'{P}ffw_adaptive_zero_cond'] = {'weights': T(gc_('linear_s.weight')),
                                       'bias': gc_('linear_s.bias')}
  return out


def stack_super(per_block, super_size):
  """stack per-block dicts into leading (num_super, super_size, ...) axes -- for our
  nested layer_stack(num_super)(layer_stack(super_size)) diffusion transformer. Block
  i -> [i//super_size, i % super_size]."""
  stacked = stack_blocks(per_block)                        # (N, ...)
  n = len(per_block)
  ns = n // super_size
  out = {}
  for path, params in stacked.items():
    out[path] = {p: v.reshape((ns, super_size) + v.shape[1:]) for p, v in params.items()}
  return out


def stack_blocks(per_block):
  """[{path: {param: arr}}, ...] (one dict per block, same keys) -> {path: {param:
  stacked}} with a leading block axis, for a hk.experimental.layer_stack scope."""
  out = {}
  keys = per_block[0]
  for path in keys:
    out[path] = {p: np.stack([blk[path][p] for blk in per_block], axis=0)
                 for p in keys[path]}
  return out


# TODO (Phase 1 cont.): convert_tri_att, convert_transition (GLU), convert_single_attn
# + single_pair_logits, convert_msa (value_dim), convert_opm, convert_template; then
# assemble the trunk under diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer
# via stack_blocks. Phase 2: input embedder + atom attention. Phase 3: diffusion.


def convert_opendde_weights(checkpoint, output_dir):
  """Convert an OpenDDE PyTorch checkpoint to a loadable AF3-haiku dir.

  Runs convert_opendde (trunk + diffusion + structural token expander/refiner +
  confidence) and writes opendde.bin.zst via the shared blob writer. Returns output_dir.
  """
  import os
  import torch
  ck = torch.load(os.path.expanduser(str(checkpoint)), map_location='cpu',
                  weights_only=False)['model']
  C.write_params_blob(output_dir, 'opendde.bin.zst', convert_opendde(ck))
  return output_dir
