"""Shared PyTorch -> AF3-haiku weight-conversion primitives.

The three AF3-lineage families we port (IntelliFold-v2, OpenFold3, OpenDDE) all
target the SAME vendored AF3 haiku graph, so the reshape/transpose math for each
module is identical across them (verified: the schema tags id/T/reshape/T_reshape
correspond exactly to the helpers below). They differ only in (a) the torch leaf
names of each submodule, and (b) whether the gated a/b projections are pre-fused.

So each family is a `Dialect` (leaf names + fusion mode) plus a map that calls
these primitives; see ifv2.py / of3.py / opendde.py. The blob writer/reader live
here too -- one place encodes the .bin record format (from alphafold3 params.py).
"""

from __future__ import annotations

import dataclasses
import io
import os

import numpy as np
import zstandard

from alphafold3.model.params import encode_record, read_records

# ---------------------------------------------------------------------------
# blob io
# ---------------------------------------------------------------------------

def write_params_blob(output_dir, filename, params, *, add_meta=True,
                      level=10, dtype=np.float32):
  """Write a nested {scope: {name: array}} tree to output_dir/filename (.bin.zst).

  Records are scope/name-sorted for a stable byte layout. add_meta prepends the
  64-byte __meta__/__identifier__ record AlphaFold3's own blob carries; a run
  stamps it into every structure it writes as the provenance of the weights, so
  we fill it with the blob's file stem (the model name) rather than the zeros
  the first OF3 conversion used. Returns the path.
  """
  output_dir = os.path.expanduser(str(output_dir))
  os.makedirs(output_dir, exist_ok=True)
  out_path = os.path.join(output_dir, filename)
  with zstandard.ZstdCompressor(level=level).stream_writer(open(out_path, 'wb')) as comp:
    if add_meta:
      ident = np.zeros(64, dtype=np.uint8)
      name = os.path.basename(filename).split('.')[0].encode()[:64]
      ident[:len(name)] = np.frombuffer(name, dtype=np.uint8)
      comp.write(encode_record('__meta__', '__identifier__', ident))
    for scope in sorted(params):
      for name in sorted(params[scope]):
        comp.write(encode_record(scope, name, np.asarray(params[scope][name], dtype=dtype)))
  return out_path


def write_records_blob(out_path, records, *, level=10, identifier=None):
  """Write (scope, name, array) records verbatim -- no dtype coercion.

  write_params_blob casts everything to float32, which is right for a converted
  AF3 model but cannot express a QUANTISED blob, and cannot write ESM-C at all:
  the record header packs the payload length as a signed 32-bit int and the
  tower's fused fc1 stack is one 5.27 GiB tensor. Quantising to int8 during
  conversion brings every record under the limit and never materialises the
  float32 blob that could not be written in the first place.
  """
  from alphafold3.model.params import encode_record
  out_path = os.path.expanduser(str(out_path))
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  with zstandard.ZstdCompressor(level=level).stream_writer(open(out_path, 'wb')) as comp:
    if identifier is not None:
      ident = np.zeros(64, dtype=np.uint8)
      name = str(identifier).encode()[:64]
      ident[:len(name)] = np.frombuffer(name, dtype=np.uint8)
      comp.write(encode_record('__meta__', '__identifier__', ident))
    for scope, name, arr in records:
      arr = np.asarray(arr)
      if arr.nbytes >= 2 ** 31:
        raise ValueError(
            '%s/%s is %.2f GiB; a record header packs its length as a SIGNED '
            '32-bit int, so 2 GiB is the ceiling. Split the array (see '
            'converters.esmc.split_stacked_blocks) or store it smaller.'
            % (scope, name, arr.nbytes / 2 ** 30))
      comp.write(encode_record(scope, name, arr))
  return out_path


def tree_to_records(params):
  """{scope: {name: array}} -> sorted (scope, name, array) records."""
  return [(scope, name, params[scope][name])
          for scope in sorted(params) for name in sorted(params[scope])]


def fold_element_index_shift(w):
  """Fold a 0-indexed-element convention into the embedding rows.

  OpenFold3 (and the families derived from it) featurise elements as
  `GetAtomicNumber() - 1`; AlphaFold 3 uses `GetAtomicNum()`, 1-indexed. The
  graph used to reconcile that by shifting the INPUT, `max(0, element - 1)`,
  which meant mutating the batch inside the forward pass and keeping a list of
  which models need it -- a list that is silent when wrong (boltz2 and rf3 are
  1-indexed and must NOT shift).

  It is a row gather on the weight instead: one_hot(max(0, e-1)) @ W is
  one_hot(e) @ W[max(0, arange(n) - 1)], exactly, for every e. Verified to
  max|d| = 0.
  """
  rows = np.maximum(0, np.arange(w.shape[0]) - 1)
  return w[rows]


def read_blob(path):
  """Read a .bin / .bin.zst back to a list of (scope, name, ndarray) records."""
  path = os.path.expanduser(str(path))
  with open(path, 'rb') as fh:
    data = zstandard.ZstdDecompressor().stream_reader(fh).read() if path.endswith('.zst') else fh.read()
  return list(read_records(io.BytesIO(data)))


# ---------------------------------------------------------------------------
# tensor helpers  (the four schema tags, as functions)
# ---------------------------------------------------------------------------

def _arr(w):
  if hasattr(w, 'detach'):
    return w.detach().float().numpy()
  return np.asarray(w, dtype=np.float32)


def t(w):
  """Linear transpose: torch (out, in) -> haiku (in, out)."""
  return _arr(w).T


def qk_grid(w, H, D):
  """Trunk GridSelfAttention Q/K: (H*D, in) -> (H, D, in)  (schema 'reshape')."""
  return _arr(w).reshape(H, D, -1)


def v_std(w, H, D):
  """V (all attn) / standard Q/K: (H*D, in) -> (in, H, D)  (schema 'T_reshape')."""
  return np.ascontiguousarray(_arr(w).T).reshape(-1, H, D)


qk_std = v_std  # standard (non-transposed) Q/K uses the same (in, H, D) layout


def gating_grid(w):
  """Trunk GridSelfAttention gating: keep (H*D, in)  (schema 'id')."""
  return _arr(w)


def _pfx(prefix, name):
  """Join a torch prefix with a leaf name; a bare name when prefix is empty."""
  return f'{prefix}.{name}' if prefix else name


def ln(sd, prefix):
  """LayerNorm weight/bias -> {scale, offset}."""
  return {'scale': _arr(sd[_pfx(prefix, 'weight')]), 'offset': _arr(sd[_pfx(prefix, 'bias')])}


def interleave(a, b):
  """GLU a/b -> fused (in, 2*hidden), a=even b=odd  (tri-mul projection/gate)."""
  a, b = _arr(a), _arr(b)
  return np.stack([a.T, b.T], axis=-1).reshape(a.shape[1], -1)


def block_concat(a, b):
  """SwiGLU a/b -> fused (in, 2*hidden) block [a|b]  (transition1)."""
  return np.concatenate([_arr(a).T, _arr(b).T], axis=-1)


# ---------------------------------------------------------------------------
# Dialect: per-family leaf names + fusion modes. Defaults describe IntelliFold-v2.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Dialect:
  # transition
  tr_ln: str = 'layer_norm'
  tr_mode: str = 'prefused'            # 'prefused' | 'swiglu' | 'block'
  tr_lin: str = 'linear'               # prefused transition1
  tr_lin_o: str = 'linear_o'           # prefused transition2
  tr_a: str = 'swiglu.linear_a'        # swiglu/block transition1 halves
  tr_b: str = 'swiglu.linear_b'
  tr_out: str = 'linear_out'           # swiglu/block transition2
  # triangle multiplication
  tm_fused: bool = True
  tm_ln_in: str = 'layer_norm_in'
  tm_ln_out: str = 'layer_norm_out'
  tm_g: str = 'linear_g'
  tm_z: str = 'linear_z'
  tm_ab_p: str = 'linear_ab_p'
  tm_ab_g: str = 'linear_ab_g'
  tm_a_p: str = 'linear_a_p'
  tm_b_p: str = 'linear_b_p'
  tm_a_g: str = 'linear_a_g'
  tm_b_g: str = 'linear_b_g'
  # grid (triangle) attention
  ga_ln: str = 'layer_norm'
  ga_bias: str = 'linear'
  ga_mha: str = 'mha'                    # sub-module holding q/k/v/g/o (empty for flat, e.g. RF3)
  ga_q: str = 'linear_q'                 # q/k/v/gate/out leaf names under ga_mha
  ga_k: str = 'linear_k'
  ga_v: str = 'linear_v'
  ga_g: str = 'linear_g'
  ga_o: str = 'linear_o'
  # single attention
  sa_ln_a: str = 'layer_norm'
  sa_ln_z: str = 'layer_norm_z'
  sa_z: str = 'linear_z'
  sa_mha: str = 'mha'
  sa_q: str = 'linear_q'
  sa_k: str = 'linear_k'
  sa_v: str = 'linear_v'
  sa_g: str = 'linear_g'
  sa_o: str = 'linear_o'
  # msa pair-weighted averaging
  msa_ln_m: str = 'layer_norm_m'
  msa_ln_z: str = 'layer_norm_z'
  msa_z: str = 'linear_z'
  msa_v: str = 'mha.linear_v'
  msa_g: str = 'mha.linear_g'
  msa_o: str = 'mha.linear_o'
  # outer product mean
  opm_ln: str = 'layer_norm'
  opm_l: str = 'linear_1'
  opm_r: str = 'linear_2'
  opm_out_direct: bool = True          # if2 stores output_w/output_b directly
  opm_out: str = 'linear_out'
  # adaptive layer norm (diffusion / atom transformer)
  ada_mod: str = 'adaptive_layer_norm'
  ada_lns: str = 'layer_norm_s'
  ada_gamma: str = 'linear_s_gamma'
  ada_beta: str = 'linear_s_beta'
  # diffusion attention block
  d_mha: str = 'mha'
  d_ada_out: str = 'linear_s'          # attention_pair_bias.<this> -> adaptive_zero_cond
  d_ct: str = 'single_transition'      # conditioned transition submodule
  d_ct_mode: str = 'prefused'          # ffw transition fusion (like tr_mode)
  d_ct_a: str = 'linear_a'             # prefused ffw transition1
  d_ct_b: str = 'linear_b'             # prefused ffw transition2
  d_ct_swa: str = 'swiglu.linear_a'    # swiglu ffw halves
  d_ct_swb: str = 'swiglu.linear_b'
  d_ct_out: str = 'linear_out'         # swiglu ffw transition2
  d_ct_ada_out: str = 'linear_s'       # conditioned_transition.<this> -> ffw_adaptive_zero_cond


DIALECT_INTELLIFOLD2 = Dialect()


# ---------------------------------------------------------------------------
# module primitives  (return flat {'sub/param': arr} under the block scope)
# ---------------------------------------------------------------------------

def transition(sd, prefix, d: Dialect):
  """(pair|single|msa) transition -> TransitionBlock params."""
  g = lambda leaf: sd[_pfx(prefix, leaf)]
  out = {'input_layer_norm/scale': _arr(g(f'{d.tr_ln}.weight')),
         'input_layer_norm/offset': _arr(g(f'{d.tr_ln}.bias'))}
  if d.tr_mode == 'prefused':
    out['transition1/weights'] = t(g(f'{d.tr_lin}.weight'))
    out['transition2/weights'] = t(g(f'{d.tr_lin_o}.weight'))
  elif d.tr_mode == 'fused':
    # one Linear(d, 2*hidden) split by torch .split(hidden) -> a=[:h], b=[h:],
    # consumed as silu(a)*b.  Same halves as 'block', already concatenated.
    out['transition1/weights'] = t(g(f'{d.tr_lin}.weight'))
    out['transition2/weights'] = t(g(f'{d.tr_out}.weight'))
  else:  # swiglu / block: two halves concatenated
    out['transition1/weights'] = block_concat(g(f'{d.tr_a}.weight'), g(f'{d.tr_b}.weight'))
    out['transition2/weights'] = t(g(f'{d.tr_out}.weight'))
  return out


def triangle_mul(sd, prefix, d: Dialect, outgoing=True):
  """One triangle multiplication."""
  g = lambda leaf: sd[_pfx(prefix, leaf)]
  out = {'left_norm_input/scale': _arr(g(f'{d.tm_ln_in}.weight')),
         'left_norm_input/offset': _arr(g(f'{d.tm_ln_in}.bias')),
         'center_norm/scale': _arr(g(f'{d.tm_ln_out}.weight')),
         'center_norm/offset': _arr(g(f'{d.tm_ln_out}.bias')),
         'gating_linear/weights': t(g(f'{d.tm_g}.weight')),
         'output_projection/weights': t(g(f'{d.tm_z}.weight'))}
  if d.tm_fused == 'bundle':
    # ESMFold2: a single proj_bundle Linear(c, 4*h).  The forward splits it
    #   signal, gate_logits = bundle.split(2*h)      -> p = rows[:2h], g = rows[2h:]
    # then chunks each half into (left, right).  Our haiku reads the projection
    # axis as an interleave, so split to a/b and re-fuse.
    w = _arr(g(f'{d.tm_ab_p}.weight'))               # (4*h, c)
    q = w.shape[0] // 4
    a_p, b_p, a_g, b_g = w[:q], w[q:2*q], w[2*q:3*q], w[3*q:]
    if not outgoing:                                 # incoming swaps the a/b roles
      a_p, b_p = b_p, a_p
      a_g, b_g = b_g, a_g
    out['projection/weights'] = interleave(a_p, b_p)
    out['gate/weights'] = interleave(a_g, b_g)
  elif d.tm_fused == 'block':
    # one fused Linear(dim, 2*dim) per gate, split by torch.chunk into block halves
    # a=[:dim], b=[dim:] (Boltz); our haiku reads the axis as an interleave, so split
    # then re-fuse with interleave. Incoming swaps the a/b roles.
    def split_interleave(leaf):
      w = _arr(g(f'{leaf}.weight'))          # (2*dim, in)
      half = w.shape[0] // 2
      a, b = w[:half], w[half:]
      if not outgoing:
        a, b = b, a
      return interleave(a, b)
    out['projection/weights'] = split_interleave(d.tm_ab_p)
    out['gate/weights'] = split_interleave(d.tm_ab_g)
  elif d.tm_fused:
    out['projection/weights'] = t(g(f'{d.tm_ab_p}.weight'))
    out['gate/weights'] = t(g(f'{d.tm_ab_g}.weight'))
  else:
    ap, bp = f'{d.tm_a_p}.weight', f'{d.tm_b_p}.weight'
    ag, bg = f'{d.tm_a_g}.weight', f'{d.tm_b_g}.weight'
    if not outgoing:                       # incoming swaps the a/b roles
      ap, bp = bp, ap
      ag, bg = bg, ag
    out['projection/weights'] = interleave(g(ap), g(bp))
    out['gate/weights'] = interleave(g(ag), g(bg))
  return out


def grid_attention(sd, prefix, d: Dialect, H, D):
  """One triangle/grid self-attention (GridSelfAttention).

  q/k/v/gate/out leaf names come from the dialect (ga_q..ga_o) under the ga_mha
  sub-module (empty ga_mha => flat, e.g. RF3's to_q/to_k/to_v/to_g/to_out)."""
  g = lambda leaf: sd[_pfx(prefix, leaf)]
  m = f'{d.ga_mha}.' if d.ga_mha else ''   # 'mha.' for AF3-lineage, '' for flat (RF3)
  out = {
      'act_norm/scale': _arr(g(f'{d.ga_ln}.weight')),
      'act_norm/offset': _arr(g(f'{d.ga_ln}.bias')),
      'pair_bias_projection/weights': t(g(f'{d.ga_bias}.weight')),
      'q_projection/weights': qk_grid(g(f'{m}{d.ga_q}.weight'), H, D),
      'k_projection/weights': qk_grid(g(f'{m}{d.ga_k}.weight'), H, D),
      'v_projection/weights': v_std(g(f'{m}{d.ga_v}.weight'), H, D),
      'gating_query/weights': gating_grid(g(f'{m}{d.ga_g}.weight')),
      'output_projection/weights': t(g(f'{m}{d.ga_o}.weight')),
  }
  # RF3 tri-attn gate/output carry biases (AF3 GridSelfAttention is bias-free); emit
  # them when present so the rf3 use_bias branch is fed. Other families lack these keys.
  gb = _pfx(prefix, f'{m}{d.ga_g}.bias')
  if gb in sd:
    out['gating_query/bias'] = _arr(sd[gb])
  ob = _pfx(prefix, f'{m}{d.ga_o}.bias')
  if ob in sd:
    out['output_projection/bias'] = _arr(sd[ob])
  return out


def single_attention(sd, prefix, d: Dialect, H, D):
  """AttentionPairBias single-attn -> PairFormerIteration single_* params."""
  g = lambda leaf: sd[_pfx(prefix, leaf)]
  mha = f'{d.sa_mha}.' if d.sa_mha else ''   # 'mha.' for AF3-lineage, '' for flat (RF3)
  out = {
      'single_pair_logits_norm/scale': _arr(g(f'{d.sa_ln_z}.weight')),
      'single_pair_logits_norm/offset': _arr(g(f'{d.sa_ln_z}.bias')),
      'single_pair_logits_projection/weights': t(g(f'{d.sa_z}.weight')),
      'single_attention_layer_norm/scale': _arr(g(f'{d.sa_ln_a}.weight')),
      'single_attention_layer_norm/offset': _arr(g(f'{d.sa_ln_a}.bias')),
      'single_attention_q_projection/weights': qk_std(g(f'{mha}{d.sa_q}.weight'), H, D),
      'single_attention_k_projection/weights': qk_std(g(f'{mha}{d.sa_k}.weight'), H, D),
      'single_attention_v_projection/weights': v_std(g(f'{mha}{d.sa_v}.weight'), H, D),
      'single_attention_gating_query/weights': t(g(f'{mha}{d.sa_g}.weight')),
      'single_attention_transition2/weights': t(g(f'{mha}{d.sa_o}.weight')),
  }
  qb = _pfx(prefix, f'{mha}{d.sa_q}.bias')
  if qb in sd:
    out['single_attention_q_projection/bias'] = _arr(sd[qb]).reshape(H, D)
  return out


def msa_attention(sd, prefix, d: Dialect, H, D):
  """MSAPairWeightedAveraging -> MSAAttention params."""
  g = lambda leaf: sd[_pfx(prefix, leaf)]
  return {
      'act_norm/scale': _arr(g(f'{d.msa_ln_m}.weight')),
      'act_norm/offset': _arr(g(f'{d.msa_ln_m}.bias')),
      'pair_norm/scale': _arr(g(f'{d.msa_ln_z}.weight')),
      'pair_norm/offset': _arr(g(f'{d.msa_ln_z}.bias')),
      'pair_logits/weights': t(g(f'{d.msa_z}.weight')),
      'v_projection/weights': v_std(g(f'{d.msa_v}.weight'), H, D),
      'gating_query/weights': t(g(f'{d.msa_g}.weight')),
      'output_projection/weights': t(g(f'{d.msa_o}.weight')),
  }


def outer_product_mean(sd, prefix, d: Dialect, c_hidden=None, c_z=None,
                       lr_bias=False):
  """OuterProductMean -> params (output_w/output_b live at the module scope).

  lr_bias carries the left/right projection BIASES, which AF3's bias-free OPM has no
  slot for. Only RF3 has them (and its are trained, not left at their zero init).
  """
  g = lambda leaf: sd[_pfx(prefix, leaf)]
  out = {
      'layer_norm_input/scale': _arr(g(f'{d.opm_ln}.weight')),
      'layer_norm_input/offset': _arr(g(f'{d.opm_ln}.bias')),
      'left_projection/weights': t(g(f'{d.opm_l}.weight')),
      'right_projection/weights': t(g(f'{d.opm_r}.weight')),
  }
  if lr_bias:
    out['left_projection/bias'] = _arr(g(f'{d.opm_l}.bias'))
    out['right_projection/bias'] = _arr(g(f'{d.opm_r}.bias'))
  if d.opm_out_direct:
    out['::output_w'] = _arr(g('output_w'))
    out['::output_b'] = _arr(g('output_b'))
  else:
    w = _arr(g(f'{d.opm_out}.weight'))       # (c_z, c_hidden**2)
    cz = w.shape[0]
    out['::output_w'] = w.T.reshape(c_hidden, c_hidden, cz)
    out['::output_b'] = _arr(g(f'{d.opm_out}.bias'))
  return out


def pair_block(sd, prefix, d: Dialect, pair_H, pair_D):
  """The pair stack shared by pairformer and template blocks (no single attn)."""
  out = {}
  for tag, name, outgoing in [('triangle_multiplication_outgoing', 'tri_mul_out', True),
                              ('triangle_multiplication_incoming', 'tri_mul_in', False)]:
    for k, v in triangle_mul(sd, f'{prefix}.{name}', d, outgoing=outgoing).items():
      out[f'{tag}/{k}'] = v
  for tag, name in [('pair_attention1', 'tri_att_start'), ('pair_attention2', 'tri_att_end')]:
    for k, v in grid_attention(sd, f'{prefix}.{name}', d, pair_H, pair_D).items():
      out[f'{tag}/{k}'] = v
  for k, v in transition(sd, f'{prefix}.pair_transition', d).items():
    out[f'pair_transition/{k}'] = v
  return out


def pairformer_block(sd, block_prefix, pair_prefix, single_prefix, d: Dialect,
                     pair_H, pair_D, single_H, single_D):
  """A full PairformerBlock: pair stack + single attention + single transition.

  block_prefix: torch prefix of the block; pair_prefix/single_prefix are the sub-
  module names for the pair stack and the single attention (they differ per family,
  e.g. 'pair_stack' + 'attention_pair_bias').
  """
  out = pair_block(sd, f'{block_prefix}.{pair_prefix}', d, pair_H, pair_D)
  out.update(single_attention(sd, f'{block_prefix}.{single_prefix}', d, single_H, single_D))
  for k, v in transition(sd, f'{block_prefix}.single_transition', d).items():
    out[f'single_transition/{k}'] = v
  return out


# ---------------------------------------------------------------------------
# diffusion / atom transformer primitives
# ---------------------------------------------------------------------------

def adaln(sd, prefix, d: Dialect):
  """Adaptive LayerNorm -> single_cond_layer_norm / single_cond_scale / single_cond_bias."""
  return {
      'single_cond_layer_norm/scale': _arr(sd[f'{prefix}.{d.ada_lns}.weight']),
      'single_cond_scale/weights': t(sd[f'{prefix}.{d.ada_gamma}.weight']),
      'single_cond_scale/bias': _arr(sd[f'{prefix}.{d.ada_gamma}.bias']),
      'single_cond_bias/weights': t(sd[f'{prefix}.{d.ada_beta}.weight']),
  }


def _cond_transition(sd, prefix, d: Dialect):
  """Conditioned (adaLN + gated) FFW of a diffusion block -> ffw_* params."""
  out = {}
  for k, v in adaln(sd, f'{prefix}.{d.ada_mod}', d).items():
    out[f'ffw_{k}'] = v
  if d.d_ct_mode == 'prefused':
    out['ffw_transition1/weights'] = t(sd[f'{prefix}.{d.d_ct_a}.weight'])
    out['ffw_transition2/weights'] = t(sd[f'{prefix}.{d.d_ct_b}.weight'])
  else:
    out['ffw_transition1/weights'] = block_concat(sd[f'{prefix}.{d.d_ct_swa}.weight'],
                                                  sd[f'{prefix}.{d.d_ct_swb}.weight'])
    out['ffw_transition2/weights'] = t(sd[f'{prefix}.{d.d_ct_out}.weight'])
  out['ffw_adaptive_zero_cond/weights'] = t(sd[f'{prefix}.{d.d_ct_ada_out}.weight'])
  out['ffw_adaptive_zero_cond/bias'] = _arr(sd[f'{prefix}.{d.d_ct_ada_out}.bias'])
  return out


def diff_attn_block(sd, block_prefix, d: Dialect, H, D, cross=False):
  """One diffusion/atom transformer block. Returns flat params under the block scope.

  Emits the per-block bare 'pair_input_layer_norm'/'pair_logits_projection' (self
  blocks only) plus the name-prefixed q/k/v/adaLN/ffw entries. The caller wraps the
  q/k/v/... under the stack's name prefix; pair_input/pair_logits are handled by the
  caller for the whole stack (they are not per-block in the haiku layout).
  """
  pa = f'{block_prefix}.attention_pair_bias'
  pt = f'{block_prefix}.{d.d_ct}'
  out = {}
  if cross:
    for tag in ('q', 'k'):
      for k, v in adaln(sd, f'{pa}.{d.ada_mod}_{tag}', d).items():
        out[f'{tag}{k}'] = v
  else:
    out.update(adaln(sd, f'{pa}.{d.ada_mod}', d))
  mha = f'{pa}.{d.d_mha}'
  out['q_projection/weights'] = qk_std(sd[f'{mha}.linear_q.weight'], H, D)
  out['q_projection/bias'] = _arr(sd[f'{mha}.linear_q.bias']).reshape(H, D)
  out['k_projection/weights'] = qk_std(sd[f'{mha}.linear_k.weight'], H, D)
  out['v_projection/weights'] = v_std(sd[f'{mha}.linear_v.weight'], H, D)
  out['gating_query/weights'] = t(sd[f'{mha}.linear_g.weight'])
  out['transition2/weights'] = t(sd[f'{mha}.linear_o.weight'])
  out['adaptive_zero_cond/weights'] = t(sd[f'{pa}.{d.d_ada_out}.weight'])
  out['adaptive_zero_cond/bias'] = _arr(sd[f'{pa}.{d.d_ada_out}.bias'])
  out.update(_cond_transition(sd, pt, d))
  return out


# ---------------------------------------------------------------------------
# stacking
# ---------------------------------------------------------------------------

def stack_blocks(block_fn, n_blocks):
  """Stack per-block dicts along a leading block axis (layer_stack collapse).

  Block 0 must be a full block (it is the key/shape reference). A later block that
  omits a submodule -- e.g. IF2's "dead" final MSA block, whose per-block builder
  drops the absent keys via `if _has(...)` -- gets that slot zero-filled, matching
  the OF3/OpenDDE convention (the layer_stack param spans every block).
  """
  blocks = [block_fn(i) for i in range(n_blocks)]
  ref = blocks[0]
  return {k: np.stack([b.get(k, np.zeros_like(ref[k])) for b in blocks], axis=0)
          for k in ref}


def stack_super(block_fn, n_blocks, n_super):
  """Stack into a nested (n_super, inner, ...) layout (diffusion main transformer)."""
  inner = n_blocks // n_super
  blocks = [block_fn(i) for i in range(n_blocks)]
  out = {}
  for k in blocks[0]:
    flat = np.stack([b[k] for b in blocks], axis=0)
    out[k] = flat.reshape((n_super, inner) + flat.shape[1:])
  return out


def nest(prefix, local):
  """Re-key a module dict under `prefix`. 'sub::name' (module-scope param) keeps its
  '::'; 'sub/param' and bare 'param' are joined with '/'."""
  out = {}
  for key, arr in local.items():
    if key.startswith('::'):          # a param at the module's own scope
      out[f'{prefix}{key}'] = arr
    else:
      out[f'{prefix}/{key}'] = arr
  return out


def populate(params, scope, local):
  """Merge a re-keyed dict into params under `scope`.

  'a/b::param' -> params[scope/a/b][param] (a param at that sub-scope);
  'a/b/param'  -> params[scope/a/b][param]; 'param' -> params[scope][param].
  """
  for key, arr in local.items():
    if '::' in key:
      sub, name = key.split('::')
      full = f'{scope}/{sub}' if sub else scope
    elif '/' in key:
      sub, name = key.rsplit('/', 1)
      full = f'{scope}/{sub}'
    else:
      full, name = scope, key
    params.setdefault(full, {})[name] = arr
