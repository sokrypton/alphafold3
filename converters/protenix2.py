"""Protenix-v2 (ByteDance, Apache-2.0) -> AF3 haiku weight converter.

Protenix-v2 is an OpenFold-derived AF3 reproduction, structurally identical to the
already-ported OF3 graph (converters/of3.py). Its ONLY architectural divergence is the
"hidden_scale_up" widening of the trunk pair channel to c_z=256 (stock/of3 is 128);
everything derived from c_z follows (triangle-mul hidden = c_z = 256, triangle-att heads
= c_z//32 = 8). The diffusion stack is UNWIDENED (token 768, single cond 384). Block
counts are AF3 defaults (pairformer 48, msa 4, template 2, confidence 4, distogram 64).

So the port is a dialect (leaf-name map) + a section map onto the SAME AF3 haiku target
scopes of3.py emits, at the wider dims. The state-dict layout was dumped and verified
against the checkpoint (memory: protenix-v2-port.md); the naming diverges from of3 in:
  - flat SwiGLU transition (linear_no_bias_a/b/(out), layernorm1) -- not nested `swiglu.`
  - pairformer block has NO `pair_stack` wrapper (tri_* direct); the MSA block DOES
  - single attn = `attention_pair_bias` (layernorm_a, attention.*, layernorm_z, linear_nobias_z)
  - MSA pair-weighted-averaging leaf names (linear_no_bias_mv/mg/z/out), not nested `mha.`
  - top-level linears (linear_no_bias_sinit/zinit1/zinit2/token_bond/z_cycle/s)
  - template = fused-feature embedder (linear_no_bias_a 108->64) + full pairformer + aggregate
  - diffusion-conditioning leaf names (transition_z1/z2 as separate modules, flat swiglu)

Status: trunk + heads + template + top-level embeddings implemented and coverage-gated.
The diffusion module (conditioning + atom enc/dec + token transformer) is the next chunk.
"""

from pathlib import Path

import numpy as np

from . import common as C
from .common import Dialect

# ─── Protenix dialect: leaf names differing from the IF2/AF3 default Dialect() ────
# Unlisted fields keep their defaults (which already describe Protenix's tri-mul --
# unfused linear_{a,b}_{p,g} -- and its grid attention -- layer_norm/linear/mha).
DIALECT_PROTENIX = Dialect(
    # transition: flat SwiGLU (silu(a)*b -> out), NOT nested under `swiglu.`
    tr_ln='layernorm1', tr_mode='swiglu',
    tr_a='linear_no_bias_a', tr_b='linear_no_bias_b', tr_out='linear_no_bias',
    # triangle multiplication: UNFUSED, default leaf names (linear_a_p/b_p/a_g/b_g/g/z)
    tm_fused=False,
    # single attention (`attention_pair_bias`)
    sa_ln_a='layernorm_a', sa_ln_z='layernorm_z', sa_z='linear_nobias_z',
    sa_mha='attention',  # q/k/v/g/o keep default linear_q.. names; q carries a bias
    # MSA pair-weighted averaging: flat leaf names (not nested `mha.`)
    msa_ln_m='layernorm_m', msa_ln_z='layernorm_z', msa_z='linear_no_bias_z',
    msa_v='linear_no_bias_mv', msa_g='linear_no_bias_mg', msa_o='linear_no_bias_out',
    # outer product mean: output from linear_out (reshaped), not stored direct
    opm_out_direct=False, opm_out='linear_out',
)
D = DIALECT_PROTENIX


# ─── small helpers (mirror of3.py) ───────────────────────────────────────────
def _t(w):
  return C.t(w)


def _get(sd, key):
  if key not in sd:
    raise KeyError(f'protenix converter: missing source key {key!r}')
  return C._arr(sd[key])


def _has(sd, key):
  return key in sd


def _set(params, scope, name, arr):
  params.setdefault(scope, {})[name] = np.asarray(arr)


def _populate_scope(params, scope, local_dict):
  """Spread a {'sub/param': arr} dict into params under `scope`.

  A key with no '/' lands on the module scope itself (e.g. outer_product_mean's
  '::output_w' -> a param named 'output_w' on the stack scope).
  """
  for k, v in local_dict.items():
    if k.startswith('::'):
      _set(params, scope, k[2:], v)
    elif '/' in k:
      sub, name = k.rsplit('/', 1)
      _set(params, f'{scope}/{sub}', name, v)
    else:
      _set(params, scope, k, v)


def _stack_blocks(per_block_fn, n_blocks):
  """Stack per-block dicts along a new leading axis (haiku layer_stack).

  Blocks may differ in which keys they carry (e.g. the last MSA block drops the MSA
  update). Missing keys are zero-filled from the first block that defines them, so
  the stack stays rectangular -- matching of3's converter.
  """
  all_dicts = [per_block_fn(i) for i in range(n_blocks)]
  ref = {}
  for d in all_dicts:
    for k, v in d.items():
      ref.setdefault(k, v)
  result = {}
  for key, refv in ref.items():
    result[key] = np.stack([d[key] if key in d else np.zeros_like(refv)
                            for d in all_dicts], axis=0)
  return result


# ─── trunk: pairformer + msa stacks ──────────────────────────────────────────
def _pairformer_block(sd, i, pair_H, pair_D, single_H, single_D):
  """Protenix pairformer block: tri_* DIRECTLY on the block (no pair_stack wrapper)."""
  prefix = f'pairformer_stack.blocks.{i}'
  out = C.pair_block(sd, prefix, D, pair_H, pair_D)          # tri_mul/att + pair_transition
  out.update(C.single_attention(sd, f'{prefix}.attention_pair_bias', D, single_H, single_D))
  for k, v in C.transition(sd, f'{prefix}.single_transition', D).items():
    out[f'single_transition/{k}'] = v
  return out


def map_pairformer_stack(sd, params, n_blocks=48,
                         pair_H=8, pair_D=32, single_H=16, single_D=24):
  scope = 'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer'
  stacked = _stack_blocks(
      lambda i: _pairformer_block(sd, i, pair_H, pair_D, single_H, single_D), n_blocks)
  _populate_scope(params, scope, stacked)


def _msa_block(sd, i, msa_H, msa_D, pair_H, pair_D):
  """Protenix MSA block: msa ops under `msa_stack.`, pair ops under `pair_stack.`."""
  prefix = f'msa_module.blocks.{i}'
  out = {}
  # the last MSA block drops the MSA update (only OPM + pair stack remain)
  if _has(sd, f'{prefix}.msa_stack.msa_pair_weighted_averaging.layernorm_m.weight'):
    for k, v in C.msa_attention(sd, f'{prefix}.msa_stack.msa_pair_weighted_averaging',
                                D, msa_H, msa_D).items():
      out[f'msa_attention1/{k}'] = v
    for k, v in C.transition(sd, f'{prefix}.msa_stack.transition_m', D).items():
      out[f'msa_transition/{k}'] = v
  for k, v in C.outer_product_mean(sd, f'{prefix}.outer_product_mean_msa', D,
                                   c_hidden=32).items():
    if k.startswith('::'):
      out[f'outer_product_mean:{k[2:]}'] = v
    else:
      out[f'outer_product_mean/{k}'] = v
  for k, v in C.pair_block(sd, f'{prefix}.pair_stack', D, pair_H, pair_D).items():
    out[k] = v
  return out


def map_msa_stack(sd, params, n_blocks=4, msa_H=8, msa_D=8, pair_H=8, pair_D=32):
  scope = 'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack'
  # the ':' regroup: outer_product_mean:output_w -> module-scope param on the opm scope
  stacked = _stack_blocks(lambda i: _msa_block(sd, i, msa_H, msa_D, pair_H, pair_D), n_blocks)
  regrouped = {}
  for k, v in stacked.items():
    if ':' in k and '/' not in k:
      sub, name = k.split(':', 1)
      regrouped[f'{sub}/{name}'] = v
    else:
      regrouped[k] = v
  _populate_scope(params, scope, regrouped)


# ─── top-level input embeddings ──────────────────────────────────────────────
def map_input_embeddings(sd, params):
  """Top-level linears + recycling norms.

  s_inputs (target_feat) layout is [atom(384), restype(32), profile(32), del_mean(1)]
  = 449, IDENTICAL to OF3 -> reuse of3's 449->447 remap (drops 1 aatype + 1 profile
  class, reorders to AF3's [aatype(31), profile(31), del_mean(1), atom(384)] = 447).

  Protenix computes z_init from s_init, not target_feat directly: s_init=sinit(tf),
  z_init=zinit1(s_init)+zinit2(s_init). AF3's graph wants left/right_single(tf)->z, and
  its single_activations == protenix's s_init, so left_single = zinit1 @ sinit (composed),
  remapped 449->447. (right_single = zinit2 @ sinit.)
  """
  from .openfold3 import _reorder_target_feat_weights as _rtf
  scope = 'diffuser/evoformer'
  # recycling: prev pair + prev single embedding (z_cycle / s)
  _set(params, f'{scope}/prev_embedding_layer_norm', 'scale', _get(sd, 'layernorm_z_cycle.weight'))
  _set(params, f'{scope}/prev_embedding_layer_norm', 'offset', _get(sd, 'layernorm_z_cycle.bias'))
  _set(params, f'{scope}/prev_embedding', 'weights', _t(_get(sd, 'linear_no_bias_z_cycle.weight')))
  _set(params, f'{scope}/prev_single_embedding_layer_norm', 'scale', _get(sd, 'layernorm_s.weight'))
  _set(params, f'{scope}/prev_single_embedding_layer_norm', 'offset', _get(sd, 'layernorm_s.bias'))
  _set(params, f'{scope}/prev_single_embedding', 'weights', _t(_get(sd, 'linear_no_bias_s.weight')))
  # single init: sinit(target_feat)->s_init, remapped 449->447
  sinit = _get(sd, 'linear_no_bias_sinit.weight')                  # torch (384, 449)
  _set(params, f'{scope}/single_activations', 'weights', _rtf(_t(sinit)))   # (447, 384)
  # pair init: compose zinit @ sinit (target_feat->z), remapped 449->447
  z1 = _get(sd, 'linear_no_bias_zinit1.weight')                    # (256, 384)
  z2 = _get(sd, 'linear_no_bias_zinit2.weight')
  _set(params, f'{scope}/left_single', 'weights', _rtf((z1 @ sinit).T))     # (447, 256)
  _set(params, f'{scope}/right_single', 'weights', _rtf((z2 @ sinit).T))
  # relative position encoding + token bond
  _set(params, f'{scope}/~_relative_encoding/position_activations', 'weights',
       _t(_get(sd, 'relative_position_encoding.linear_no_bias.weight')))
  _set(params, f'{scope}/bond_embedding', 'weights', _t(_get(sd, 'linear_no_bias_token_bond.weight')))
  # MSA module embedder: msa activations (34-d MSA feat) + s_inputs->MSA (449->447).
  # The 32-class block IS restype-indexed and needs the same alphabet permutation as
  # everything else here -- protenix's STD_RESIDUES_WITH_GAP is OF3's ordering exactly
  # (21-25 A/G/C/U/N, 26-30 DA..DN, 31 GAP) against AF3's (21 GAP, 22-25 A/G/C/U, 30 N).
  # It used to be left unpermuted ("no remap"), which is silent: 34 rows either way, and
  # protein and DNA indices coincide, so only RNA and the gap class are wrong.
  from .openfold3 import _reorder_msa_weights as _rmsa
  _set(params, f'{scope}/msa_activations', 'weights',
       _rmsa(_t(_get(sd, 'msa_module.linear_no_bias_m.weight'))))
  _set(params, f'{scope}/extra_msa_target_feat', 'weights',
       _rtf(_t(_get(sd, 'msa_module.linear_no_bias_s.weight'))))


# ─── distogram head ──────────────────────────────────────────────────────────
def map_distogram_head(sd, params):
  _set(params, 'diffuser/distogram_head/half_logits', 'weights',
       _t(_get(sd, 'distogram_head.linear.weight')))
  # Protenix symmetrises AFTER the linear (`logits + logits.transpose(-2, -3)`,
  # head.py:55), exactly as our graph does, so its bias is doubled on both sides
  # and maps straight across. See model_config.DISTOGRAM_BIAS.
  _set(params, 'diffuser/distogram_head/half_logits', 'bias',
       _get(sd, 'distogram_head.linear.bias'))


# ─── confidence head ─────────────────────────────────────────────────────────
def _conf_pairformer_block(sd, i, pair_H, pair_D, single_H, single_D):
  prefix = f'confidence_head.pairformer_stack.blocks.{i}'
  out = C.pair_block(sd, prefix, D, pair_H, pair_D)
  out.update(C.single_attention(sd, f'{prefix}.attention_pair_bias', D, single_H, single_D))
  for k, v in C.transition(sd, f'{prefix}.single_transition', D).items():
    out[f'single_transition/{k}'] = v
  return out


def map_confidence_head(sd, params, n_layers=4, pair_H=8, pair_D=32,
                        single_H=16, single_D=24, c_s=384, max_atoms_per_token=24):
  scope_base = 'diffuser/confidence_head'
  stack_scope = f'{scope_base}/__layer_stack_no_per_layer/confidence_pairformer'
  _populate_scope(params, stack_scope,
                  _stack_blocks(lambda i: _conf_pairformer_block(sd, i, pair_H, pair_D,
                                                                 single_H, single_D), n_layers))
  # feature embedders: s_inputs -> left/right, distance one-hot -> pair
  embed_scope = f'{scope_base}/~_embed_features'
  from .openfold3 import _reorder_target_feat_weights as _rtf
  _set(params, f'{embed_scope}/left_target_feat_project', 'weights',
       _rtf(_t(_get(sd, 'confidence_head.linear_no_bias_s1.weight'))))
  _set(params, f'{embed_scope}/right_target_feat_project', 'weights',
       _rtf(_t(_get(sd, 'confidence_head.linear_no_bias_s2.weight'))))
  _set(params, f'{embed_scope}/distogram_feat_project', 'weights',
       _t(_get(sd, 'confidence_head.linear_no_bias_d.weight')))
  # the second, UNBINNED distance term (protenix-gated branch in _embed_features)
  _set(params, f'{embed_scope}/distance_feat_project', 'weights',
       _t(_get(sd, 'confidence_head.linear_no_bias_d_wo_onehot.weight')))
  # LayerNorm on the trunk single before the head (protenix-gated branch in __call__)
  _set(params, f'{scope_base}/input_single_norm', 'scale',
       _get(sd, 'confidence_head.input_strunk_ln.weight'))
  _set(params, f'{scope_base}/input_single_norm', 'offset',
       _get(sd, 'confidence_head.input_strunk_ln.bias'))
  # pde / pae logits (+ their input LNs)
  _set(params, f'{scope_base}/logits_ln', 'scale', _get(sd, 'confidence_head.pde_ln.weight'))
  _set(params, f'{scope_base}/logits_ln', 'offset', _get(sd, 'confidence_head.pde_ln.bias'))
  _set(params, f'{scope_base}/left_half_distance_logits', 'weights',
       _t(_get(sd, 'confidence_head.linear_no_bias_pde.weight')))
  _set(params, f'{scope_base}/pae_logits_ln', 'scale', _get(sd, 'confidence_head.pae_ln.weight'))
  _set(params, f'{scope_base}/pae_logits_ln', 'offset', _get(sd, 'confidence_head.pae_ln.bias'))
  _set(params, f'{scope_base}/pae_logits', 'weights',
       _t(_get(sd, 'confidence_head.linear_no_bias_pae.weight')))
  # plddt / experimentally-resolved: per-atom-type heads stored as (atoms, c_s, bins)
  _set(params, f'{scope_base}/plddt_logits_ln', 'scale', _get(sd, 'confidence_head.plddt_ln.weight'))
  _set(params, f'{scope_base}/plddt_logits_ln', 'offset', _get(sd, 'confidence_head.plddt_ln.bias'))
  _set(params, f'{scope_base}/plddt_logits', 'weights',
       _plddt_logits(_get(sd, 'confidence_head.plddt_weight'), c_s, 50, max_atoms_per_token))
  _set(params, f'{scope_base}/experimentally_resolved_ln', 'scale',
       _get(sd, 'confidence_head.resolved_ln.weight'))
  _set(params, f'{scope_base}/experimentally_resolved_ln', 'offset',
       _get(sd, 'confidence_head.resolved_ln.bias'))
  _set(params, f'{scope_base}/experimentally_resolved_logits', 'weights',
       _plddt_logits(_get(sd, 'confidence_head.resolved_weight'), c_s, 2, max_atoms_per_token))


def _plddt_logits(w, c_s, bins, max_atoms):
  """(atoms, c_s, bins) torch weight -> (c_s, max_atoms, bins) haiku, atom-padded."""
  atoms = w.shape[0]
  logits = np.transpose(w, (1, 0, 2))              # (c_s, atoms, bins)
  if atoms < max_atoms:
    pad = np.zeros((c_s, max_atoms - atoms, bins), dtype=logits.dtype)
    logits = np.concatenate([logits, pad], axis=1)
  return logits


# ─── template embedder (fused-feature embedder + full pairformer + aggregate) ──
def _template_pair_block(sd, i, templ_H, templ_D):
  """Template pairformer block -- pair-only (tri_* + pair_transition), c_z=64."""
  return C.pair_block(sd, f'template_embedder.pairformer_stack.blocks.{i}', D, templ_H, templ_D)


def map_template_embedder(sd, params, n_blocks=2, templ_H=2, templ_D=32):
  """Protenix template = boltz2-style FUSED embedder -> Boltz2TemplateEmbedding scopes.

  Maps onto diffuser/evoformer/template_embedding (the boltz2 template branch), NOT
  of3's per-feature embedder. Weight orientation matches boltz2's map_template:
    z_norm/v_norm carry offsets; z/a/u_proj are no-bias linears (torch (out,in)->C.t);
    the 2-block pair-only pairformer (protenix leaf names) stacks under tmpl_pairformer.
  NOTE: the FORWARD feature builder differs -- protenix a_proj is 108-d vs boltz2's
  109-d, so Boltz2TemplateEmbedding._features (109-d) can't be reused as-is; a protenix
  108-d feature builder + forward gate is the remaining template work (task #31).
  """
  te = 'template_embedder'
  TE = 'diffuser/evoformer/template_embedding'
  _set(params, f'{TE}/z_norm', 'scale', _get(sd, f'{te}.layernorm_z.weight'))
  _set(params, f'{TE}/z_norm', 'offset', _get(sd, f'{te}.layernorm_z.bias'))
  _set(params, f'{TE}/v_norm', 'scale', _get(sd, f'{te}.layernorm_v.weight'))
  _set(params, f'{TE}/v_norm', 'offset', _get(sd, f'{te}.layernorm_v.bias'))
  _set(params, f'{TE}/z_proj', 'weights', _t(_get(sd, f'{te}.linear_no_bias_z.weight')))
  _set(params, f'{TE}/a_proj', 'weights', _t(_get(sd, f'{te}.linear_no_bias_a.weight')))
  _set(params, f'{TE}/u_proj', 'weights', _t(_get(sd, f'{te}.linear_no_bias_u.weight')))
  stacked = _stack_blocks(lambda i: {f'tmpl_pairformer/{k}': v for k, v in
                                     C.pair_block(sd, f'{te}.pairformer_stack.blocks.{i}',
                                                  D, templ_H, templ_D).items()}, n_blocks)
  _populate_scope(params, f'{TE}/__layer_stack_no_per_layer', stacked)


# ─── top-level map ───────────────────────────────────────────────────────────
def derive_dims(sd):
  """Read a Protenix checkpoint's own shape -- block counts and widths -- from it.

  Protenix publishes nine model types (configs_model_type.py) and they differ from
  one another ONLY in counts and widths: mini is 16 pairformer blocks where v2 has
  48, tiny 8, v0.5.0 is templateless (0 template blocks), v2 alone widens c_z to
  256. Nothing about the module tree changes, so one converter serves all of them
  as long as it stops assuming v2's numbers.

  Derived, never passed, for the reason D36 records in ChoongHwanLee's port (see
  README): a literal that is right for one release is wrong the moment another
  ships, and the failure is silent -- shapes agree, the model loads, every number
  is wrong. The artefact survives a release; a default argument does not.

  Note the two atom encoders are counted SEPARATELY. mini drops the diffusion
  atom encoder to 1 block while leaving the input embedder's at 3, so a single
  `n_atom` would be wrong for one of them.
  """
  import re

  def n_blocks(prefix):
    idx = {int(m.group(1)) for k in sd
           if (m := re.match(rf'{re.escape(prefix)}\.(\d+)\.', k))}
    return (max(idx) + 1) if idx else 0

  dims = dict(
      n_msa=n_blocks('msa_module.blocks'),
      n_pairformer=n_blocks('pairformer_stack.blocks'),
      n_template=n_blocks('template_embedder.pairformer_stack.blocks'),
      n_confidence=n_blocks('confidence_head.pairformer_stack.blocks'),
      n_diff_token=n_blocks('diffusion_module.diffusion_transformer.blocks'),
      n_diff_atom_enc=n_blocks('diffusion_module.atom_attention_encoder'
                               '.atom_transformer.diffusion_transformer.blocks'),
      n_diff_atom_dec=n_blocks('diffusion_module.atom_attention_decoder'
                               '.atom_transformer.diffusion_transformer.blocks'),
      n_input_atom_enc=n_blocks('input_embedder.atom_attention_encoder'
                                '.atom_transformer.diffusion_transformer.blocks'),
  )
  # widths, off a tensor that every release carries
  dims['c_z'] = int(_get(sd, 'pairformer_stack.blocks.0.tri_mul_out.linear_a_p.weight').shape[0])
  dims['c_s'] = int(_get(sd, 'linear_no_bias_sinit.weight').shape[0])
  dims['num_bins'] = int(_get(sd, 'distogram_head.linear.weight').shape[0])
  # head count stated by the pair-bias projection, not computed from c_z // 32
  dims['pair_H'] = int(_get(sd, 'pairformer_stack.blocks.0.tri_att_start.linear.weight').shape[0])
  # the template stack has its OWN head count and its own triangle-multiplication
  # width, and v1 differs from v2 in both: 4 heads against 2, and a 128-wide
  # tri-mul on a 64-channel pair against 64. Absent on the templateless releases.
  tb = 'template_embedder.pairformer_stack.blocks.0'
  if _has(sd, f'{tb}.tri_att_start.linear.weight'):
    dims['templ_H'] = int(_get(sd, f'{tb}.tri_att_start.linear.weight').shape[0])
    dims['templ_hidden'] = int(_get(sd, f'{tb}.tri_mul_out.linear_a_p.weight').shape[0])
  return dims


def map_protenix2_to_af3(sd, **overrides):
  """Convert ANY Protenix state dict (module.* stripped) to AF3 haiku params.

  Named for protenix2 because that is the first release it served, but the shape
  is read from the checkpoint (`derive_dims`), so the same function converts the
  mini and tiny model types -- and any future one that changes only counts and
  widths, which is every Protenix release so far.

  Pass an override only to test a deliberate mismatch; the derived value is
  right for the checkpoint in hand by construction.
  """
  d = derive_dims(sd)
  d.update(overrides)
  params = {}
  map_input_embeddings(sd, params)
  map_msa_stack(sd, params, n_blocks=d['n_msa'], pair_H=d['pair_H'])
  map_pairformer_stack(sd, params, n_blocks=d['n_pairformer'], pair_H=d['pair_H'])
  map_confidence_head(sd, params, n_layers=d['n_confidence'], pair_H=d['pair_H'])
  map_distogram_head(sd, params)
  if d['n_template']:
    # v0.5.0-lineage checkpoints (mini, tiny) are TEMPLATELESS and carry no
    # template_embedder tensors at all; asking for them raises rather than
    # silently emitting zeros.
    map_template_embedder(sd, params, n_blocks=d['n_template'],
                          templ_H=d['templ_H'])
  map_evoformer_conditioning(sd, params, n_atom=d['n_input_atom_enc'])
  # super_block_size is 4 in the graph, so the token transformer nests as
  # (num_blocks // 4, 4); 24 -> 6 supers for protenix2, 8 -> 2 for mini/tiny.
  map_diffusion(sd, params, n_token=d['n_diff_token'],
                n_super=max(1, d['n_diff_token'] // 4),
                n_atom=d['n_diff_atom_enc'])
  return params


# ─── checkpoint I/O ──────────────────────────────────────────────────────────
def load_protenix_checkpoint(ckpt_path):
  """Load protenix-v2.pt -> state dict with the `module.` prefix stripped."""
  import torch
  ck = torch.load(str(Path(ckpt_path).expanduser()), map_location='cpu', weights_only=False)
  sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
  return {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in sd.items()}


def _convert_protenix(checkpoint, output_dir, model_name):
  """Shared body for every Protenix model type.

  Refuses a checkpoint whose derived shape does not match the name asked for.
  The graph is built from the registry's static settings while the blob is built
  from the checkpoint, so a mismatch between the two is exactly the silent kind:
  the blob would load and every block count would be wrong.
  """
  from alphafold3.model import model_registry
  sd = load_protenix_checkpoint(checkpoint)
  d = derive_dims(sd)
  cfg = model_registry.get(model_name).configure(_af3_config())
  want = {
      'n_pairformer': cfg.evoformer.pairformer.num_layer,
      'n_msa': cfg.evoformer.msa_stack.num_layer,
      'n_diff_token': cfg.heads.diffusion.transformer.num_blocks,
      'c_z': cfg.evoformer.pair_channel,
  }
  bad = {k: (d[k], v) for k, v in want.items() if d[k] != v}
  if bad:
    raise ValueError(
        f'--model {model_name} does not match this checkpoint: '
        + ', '.join(f'{k} is {got} but the graph wants {exp}'
                    for k, (got, exp) in bad.items())
        + '. Protenix model types differ only in counts and widths, so the wrong'
          ' pairing loads silently and is wrong everywhere.')
  return Path(C.write_params_blob(output_dir, f'{model_name}.bin.zst',
                                  map_protenix2_to_af3(sd), add_meta=True))


def _af3_config():
  from alphafold3.model import model
  return model.Model.Config()


def convert_protenix1_weights(checkpoint, output_dir):
  """Convert Protenix's `protenix_base_default_v1.0.0` checkpoint (368 M)."""
  return _convert_protenix(checkpoint, output_dir, 'protenix1')


def convert_protenix_mini_weights(checkpoint, output_dir):
  """Convert Protenix's `protenix_mini_default_v0.5.0` checkpoint."""
  return _convert_protenix(checkpoint, output_dir, 'protenix_mini')


def convert_protenix_tiny_weights(checkpoint, output_dir):
  """Convert Protenix's `protenix_tiny_default_v0.5.0` checkpoint."""
  return _convert_protenix(checkpoint, output_dir, 'protenix_tiny')


def convert_protenix2_weights(checkpoint, output_dir):
  """Convert protenix-v2.pt to a loadable AF3-haiku blob dir."""
  sd = load_protenix_checkpoint(checkpoint)
  params = map_protenix2_to_af3(sd)
  return Path(C.write_params_blob(output_dir, 'protenix2.bin.zst',
                                  params, add_meta=True))


# ─── diffusion module ────────────────────────────────────────────────────────
# Mirrors of3.map_diffusion_head / map_evoformer_conditioning onto the SAME AF3
# target scopes, with Protenix leaf names. adaLN (primitives.AdaptiveLayerNorm):
#   a = sigmoid(linear_s(s)) * layernorm_a(a) + linear_nobias_s(s)
# so single_cond_scale <- linear_s (+bias, the sigmoid gate), single_cond_bias <-
# linear_nobias_s, single_cond_layer_norm <- layernorm_s. ConditionedTransitionBlock:
#   b = silu(a1(a))*a2(a);  a = sigmoid(linear_s(s)) * b_proj(b)
# so ffw_transition1 <- concat[a1,a2], ffw_transition2 <- linear_nobias_b,
# ffw_adaptive_zero_cond <- linear_s (+bias). The fused SwiGLU Transition modules of
# diffusion_conditioning (transition_z1/2, transition_s1/2) use the standard dialect.

def _p_adaln(sd, prefix, name):
  """Protenix AdaptiveLayerNorm -> of3 adaln params under `name`."""
  return {
      f'{name}single_cond_layer_norm/scale': _get(sd, f'{prefix}.layernorm_s.weight'),
      f'{name}single_cond_scale/weights': _t(_get(sd, f'{prefix}.linear_s.weight')),
      f'{name}single_cond_scale/bias': _get(sd, f'{prefix}.linear_s.bias'),
      f'{name}single_cond_bias/weights': _t(_get(sd, f'{prefix}.linear_nobias_s.weight')),
  }


def _p_cond_transition(sd, prefix, name):
  """Protenix ConditionedTransitionBlock -> of3 ffw params under `name`."""
  d = _p_adaln(sd, f'{prefix}.adaln', f'{name}ffw_')
  a1 = _get(sd, f'{prefix}.linear_nobias_a1.weight')
  a2 = _get(sd, f'{prefix}.linear_nobias_a2.weight')
  d[f'{name}ffw_transition1/weights'] = np.concatenate([a1.T, a2.T], axis=-1)
  d[f'{name}ffw_transition2/weights'] = _t(_get(sd, f'{prefix}.linear_nobias_b.weight'))
  d[f'{name}ffw_adaptive_zero_cond/weights'] = _t(_get(sd, f'{prefix}.linear_s.weight'))
  d[f'{name}ffw_adaptive_zero_cond/bias'] = _get(sd, f'{prefix}.linear_s.bias')
  return d


def _p_diff_block(sd, i, prefix_base, name, H, D, cross=False):
  """One diffusion transformer block (self- or cross-attn). Returns flat {tag/param}."""
  pa = f'{prefix_base}.blocks.{i}.attention_pair_bias'
  ct = f'{prefix_base}.blocks.{i}.conditioned_transition_block'
  d = {}
  if cross:
    d.update(_p_adaln(sd, f'{pa}.layernorm_a', f'{name}q'))
    d.update(_p_adaln(sd, f'{pa}.layernorm_kv', f'{name}k'))
  else:
    d.update(_p_adaln(sd, f'{pa}.layernorm_a', name))
    d['pair_input_layer_norm/scale'] = _get(sd, f'{pa}.layernorm_z.weight')
    d['pair_logits_projection/weights'] = _t(_get(sd, f'{pa}.linear_nobias_z.weight'))
  qw = _get(sd, f'{pa}.attention.linear_q.weight')
  d[f'{name}q_projection/weights'] = qw.T.reshape(-1, H, D)
  d[f'{name}q_projection/bias'] = _get(sd, f'{pa}.attention.linear_q.bias').reshape(H, D)
  d[f'{name}k_projection/weights'] = _get(sd, f'{pa}.attention.linear_k.weight').T.reshape(-1, H, D)
  d[f'{name}v_projection/weights'] = _get(sd, f'{pa}.attention.linear_v.weight').T.reshape(-1, H, D)
  d[f'{name}gating_query/weights'] = _t(_get(sd, f'{pa}.attention.linear_g.weight'))
  d[f'{name}transition2/weights'] = _t(_get(sd, f'{pa}.attention.linear_o.weight'))
  d[f'{name}adaptive_zero_cond/weights'] = _t(_get(sd, f'{pa}.linear_a_last.weight'))
  d[f'{name}adaptive_zero_cond/bias'] = _get(sd, f'{pa}.linear_a_last.bias')
  d.update(_p_cond_transition(sd, ct, name))
  return d


def _p_pair_logits_flat(sd, prefix_base, n):
  return np.stack([_t(_get(sd, f'{prefix_base}.blocks.{j}.attention_pair_bias.linear_nobias_z.weight'))
                   for j in range(n)], axis=1)


def _p_atom_transformer(sd, params, scope, base, name, n_blocks, H, D):
  """A cross-attn atom transformer (encoder/decoder).

  Protenix applies the atom-pair LayerNorm PER BLOCK (its per-block layernorm_z
  differ), exactly like OpenDDE -- so this rides the opendde forward branch (per-block
  pair LN/projection baked into each block, plain `__layer_stack_no_per_layer`), NOT
  of3's shared-LN `__layer_stack_with_per_layer`. Reuse OpenDDE's validated per-block
  converter + stacker (identical leaf names + adaLN convention).
  """
  from . import opendde as OD
  per = []
  for i in range(n_blocks):
    p = f'{base}.blocks.{i}.'
    blk_sd = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    per.append(OD.convert_atom_transformer_block(blk_sd, name))
  stacked = OD.stack_blocks(per)                       # {path: {param: stacked_arr}}
  stack_scope = f'{scope}/__layer_stack_no_per_layer'
  for path, pdict in stacked.items():
    for param, arr in pdict.items():
      _set(params, f'{stack_scope}/{path}', param, arr)


def _p_ref_embedders(sd, params, scope, enc, pfx):
  """Atom ref-feature embedders: fused-f split + ref_pos/charge + pair feats + q + trunk cond.

  pfx is the target-name prefix ('evoformer_conditioning_' or 'diffusion_').
  """
  f = _get(sd, f'{enc}.linear_no_bias_f.weight')          # (128, 385) = [mask1, elem128, name256]
  _set(params, f'{scope}/{pfx}embed_ref_mask', 'weights', f[:, 0:1].T)
  _set(params, f'{scope}/{pfx}embed_ref_element', 'weights',
       C.fold_element_index_shift(f[:, 1:129].T))
  _set(params, f'{scope}/{pfx}embed_ref_atom_name', 'weights', f[:, 129:385].T)
  _set(params, f'{scope}/{pfx}embed_ref_pos', 'weights', _t(_get(sd, f'{enc}.linear_no_bias_ref_pos.weight')))
  _set(params, f'{scope}/{pfx}embed_ref_charge', 'weights', _t(_get(sd, f'{enc}.linear_no_bias_ref_charge.weight')))
  # pair features (+ the of3 "_1" duplicates that share the weight)
  for suf, leaf, dup in [('embed_pair_offsets', 'linear_no_bias_d', True),
                         ('embed_pair_distances', 'linear_no_bias_invd', True),
                         ('embed_pair_offsets_valid', 'linear_no_bias_v', False)]:
    w = _t(_get(sd, f'{enc}.{leaf}.weight'))
    _set(params, f'{scope}/{pfx}{suf}', 'weights', w)
    if dup:
      _set(params, f'{scope}/{pfx}{suf}_1', 'weights', w)
  for suf, leaf in [('single_to_pair_cond_row', 'linear_no_bias_cl'),
                    ('single_to_pair_cond_col', 'linear_no_bias_cm')]:
    w = _t(_get(sd, f'{enc}.{leaf}.weight'))
    _set(params, f'{scope}/{pfx}{suf}', 'weights', w)
    _set(params, f'{scope}/{pfx}{suf}_1', 'weights', w)
  _set(params, f'{scope}/{pfx}pair_mlp_1', 'weights', _t(_get(sd, f'{enc}.small_mlp.1.weight')))
  _set(params, f'{scope}/{pfx}pair_mlp_2', 'weights', _t(_get(sd, f'{enc}.small_mlp.3.weight')))
  _set(params, f'{scope}/{pfx}pair_mlp_3', 'weights', _t(_get(sd, f'{enc}.small_mlp.5.weight')))
  _set(params, f'{scope}/{pfx}project_atom_features_for_aggr', 'weights',
       _t(_get(sd, f'{enc}.linear_no_bias_q.weight')))


def _p_flat_swiglu(sd, params, scope, prefix, name):
  """A flat SwiGLU Transition (layernorm1 + linear_no_bias_a/b + linear_no_bias) -> ffw_*."""
  _set(params, f'{scope}/{name}ffw_layer_norm', 'scale', _get(sd, f'{prefix}.layernorm1.weight'))
  _set(params, f'{scope}/{name}ffw_layer_norm', 'offset', _get(sd, f'{prefix}.layernorm1.bias'))
  a = _get(sd, f'{prefix}.linear_no_bias_a.weight')
  b = _get(sd, f'{prefix}.linear_no_bias_b.weight')
  _set(params, f'{scope}/{name}ffw_transition1', 'weights', np.concatenate([a.T, b.T], axis=-1))
  _set(params, f'{scope}/{name}ffw_transition2', 'weights', _t(_get(sd, f'{prefix}.linear_no_bias.weight')))


def map_diffusion(sd, params, *, n_token=24, n_super=6, diff_H=16, diff_D=48,
                  n_atom=3, atom_H=4, atom_D=32):
  """Map the whole diffusion module onto AF3 diffusion_head + evoformer_conditioning scopes."""
  dm = 'diffusion_module'
  dc = f'{dm}.diffusion_conditioning'
  scope = 'diffuser/~/diffusion_head'

  # --- diffusion conditioning: pair path ---
  _set(params, f'{scope}/relpe_projection', 'weights', _t(_get(sd, f'{dc}.relpe.linear_no_bias.weight')))
  _set(params, f'{scope}/pair_cond_initial_norm', 'scale', _get(sd, f'{dc}.layernorm_z.weight'))
  _set(params, f'{scope}/pair_cond_initial_projection', 'weights', _t(_get(sd, f'{dc}.linear_no_bias_z.weight')))
  _p_flat_swiglu(sd, params, scope, f'{dc}.transition_z1', 'pair_transition_0')
  _p_flat_swiglu(sd, params, scope, f'{dc}.transition_z2', 'pair_transition_1')
  # --- diffusion conditioning: single path ---
  from .openfold3 import _reorder_features_1d as _of3_rf1
  # 831, not openfold3's 833: this graph's single_cond_initial_norm
  # spans the AF3-width block, so the two classes AF3 lacks are dropped.
  _rf1 = lambda a, **kw: _of3_rf1(a, pad_unk_dna=False, **kw)
  _set(params, f'{scope}/single_cond_initial_norm', 'scale', _rf1(_get(sd, f'{dc}.layernorm_s.weight')))
  _set(params, f'{scope}/single_cond_initial_projection', 'weights', _rf1(_t(_get(sd, f'{dc}.linear_no_bias_s.weight'))))
  _set(params, f'{scope}/noise_embedding_initial_norm', 'scale', _get(sd, f'{dc}.layernorm_n.weight'))
  _set(params, f'{scope}/noise_embedding_initial_projection', 'weights', _t(_get(sd, f'{dc}.linear_no_bias_n.weight')))
  _set(params, scope, 'fourier_embedding_weight', _get(sd, f'{dc}.fourier_embedding.w').astype(np.float32))
  _set(params, scope, 'fourier_embedding_bias', _get(sd, f'{dc}.fourier_embedding.b').astype(np.float32))
  _p_flat_swiglu(sd, params, scope, f'{dc}.transition_s1', 'single_transition_0')
  _p_flat_swiglu(sd, params, scope, f'{dc}.transition_s2', 'single_transition_1')

  # --- diffusion atom encoder (ref embedders + trunk cond + atom transformer) ---
  ae = f'{dm}.atom_attention_encoder'
  _p_ref_embedders(sd, params, scope, ae, 'diffusion_')
  _set(params, f'{scope}/diffusion_embed_trunk_single_cond', 'weights', _t(_get(sd, f'{ae}.linear_no_bias_s.weight')))
  _set(params, f'{scope}/diffusion_lnorm_trunk_single_cond', 'scale', _get(sd, f'{ae}.layernorm_s.weight'))
  _set(params, f'{scope}/diffusion_embed_trunk_pair_cond', 'weights', _t(_get(sd, f'{ae}.linear_no_bias_z.weight')))
  _set(params, f'{scope}/diffusion_lnorm_trunk_pair_cond', 'scale', _get(sd, f'{ae}.layernorm_z.weight'))
  _set(params, f'{scope}/diffusion_atom_positions_to_features', 'weights', _t(_get(sd, f'{ae}.linear_no_bias_r.weight')))
  _p_atom_transformer(sd, params, f'{scope}/diffusion_atom_transformer_encoder',
                      f'{ae}.atom_transformer.diffusion_transformer',
                      'diffusion_atom_transformer_encoder', n_atom, atom_H, atom_D)

  # --- diffusion atom decoder ---
  ad = f'{dm}.atom_attention_decoder'
  _set(params, f'{scope}/diffusion_project_token_features_for_broadcast', 'weights',
       _t(_get(sd, f'{ad}.linear_no_bias_a.weight')))
  _set(params, f'{scope}/diffusion_atom_features_layer_norm', 'scale', _get(sd, f'{ad}.layernorm_q.weight'))
  _set(params, f'{scope}/diffusion_atom_features_to_position_update', 'weights',
       _t(_get(sd, f'{ad}.linear_no_bias_out.weight')))
  _p_atom_transformer(sd, params, f'{scope}/diffusion_atom_transformer_decoder',
                      f'{ad}.atom_transformer.diffusion_transformer',
                      'diffusion_atom_transformer_decoder', n_atom, atom_H, atom_D)

  # --- token transformer (24 blocks nested (n_super, n_token//n_super)) ---
  tr_base = f'{dm}.diffusion_transformer'
  blocks = [_p_diff_block(sd, i, tr_base, 'transformer', diff_H, diff_D, cross=False) for i in range(n_token)]
  ss = n_token // n_super
  tr_inner = f'{scope}/transformer/__layer_stack_no_per_layer/__layer_stack_no_per_layer'
  stacked = {}
  for k in blocks[0]:
    flat = np.stack([b[k] for b in blocks], axis=0)
    stacked[k] = flat.reshape((n_super, ss) + flat.shape[1:])
  _populate_scope(params, tr_inner, stacked)
  _set(params, f'{scope}/single_cond_embedding_norm', 'scale', _get(sd, f'{dm}.layernorm_s.weight'))
  _set(params, f'{scope}/single_cond_embedding_projection', 'weights', _t(_get(sd, f'{dm}.linear_no_bias_s.weight')))
  _set(params, f'{scope}/output_norm', 'scale', _get(sd, f'{dm}.layernorm_a.weight'))


def map_evoformer_conditioning(sd, params, *, n_atom=3, atom_H=4, atom_D=32):
  """Input-embedder atom encoder -> evoformer_conditioning_* scopes (no trunk cond).

  Same ref-feature embedders + cross-attn atom transformer as the diffusion atom
  encoder, but the input embedder's atom encoder has no noisy-position/trunk cond
  (it computes s_inputs on the first pass).
  """
  scope = 'diffuser'
  enc = 'input_embedder.atom_attention_encoder'
  _p_ref_embedders(sd, params, scope, enc, 'evoformer_conditioning_')
  _p_atom_transformer(sd, params, f'{scope}/evoformer_conditioning_atom_transformer_encoder',
                      f'{enc}.atom_transformer.diffusion_transformer',
                      'evoformer_conditioning_atom_transformer_encoder', n_atom, atom_H, atom_D)
