'''convert AF2 monomer haiku params to run on the multimer graph -- at load time

The unification runs everything on one graph (multimer). Monomer weights are
reshaped/split/padded onto that graph in memory; nothing is written to disk (the
params directory stays the canonical DeepMind .npz files). The full map, with
shapes, is in MERGE_NOTES.md -- every step is a value/shape transform with no
free parameters.

Scope: trunk + structure module, no-template path (the monomer use cases:
hallucination, fixbb). The template embedder is architecturally different
between monomer and multimer and is NOT converted; a converted monomer runs with
templates masked off. pTM head is identical shape (passthrough for a ptm source;
a non-ptm source needs it zero-filled -- handled where the graph's params are
completed, not here).
'''
from __future__ import annotations

import numpy as np

_A = 'alphafold/alphafold_iteration/'
_SM = _A + 'structure_module/fold_iteration/'
_IPA = _SM + 'invariant_point_attention/'
_EVO = _A + 'evoformer/'

# IPA head geometry (fixed for AF2): 12 heads, 16 scalar q/k/v, 4 qk points
# (x3 = 12), 8 v points (x3 = 24).
_H, _SC, _PQK, _PV = 12, 16, 12, 24

# fused monomer modules replaced by the split multimer ones
_FUSED = {
    _IPA + 'q_scalar', _IPA + 'kv_scalar',
    _IPA + 'q_point_local', _IPA + 'kv_point_local',
    _SM + 'affine_update', _EVO + 'pair_activiations',
    _EVO + 'left_single', _EVO + 'right_single', _EVO + 'preprocess_1d',
    _A + 'masked_msa_head/logits',
}


def _np(x):
  return np.asarray(x)


def convert_monomer_params(p: dict) -> dict:
  '''monomer param dict -> multimer-graph-shaped param dict (in memory)'''
  out = {}
  for k, v in p.items():
    if k in _FUSED or 'template' in k:
      continue                    # handled below / template embedder excluded
    out[k] = v

  # ---- IPA scalar. Monomer reshapes the flat projection to (head, scalar)
  # C-order, so q_scalar -> (D,12,16) directly. kv_scalar is (head, 32) split
  # at 16 PER HEAD (num_scalar_qk) into k,v -- NOT a flat split at 192, which
  # would take whole heads. So reshape to (D,12,32) then split the last axis.
  P_QK, P_V = 4, 8   # num_point_qk, num_point_v (x3 coords = 12, 24)

  def _scalar(mod, out_names):
    w, b = _np(mod['weights']), _np(mod['bias'])
    D = w.shape[0]
    per = w.shape[1] // _H
    w = w.reshape(D, _H, per)
    b = b.reshape(_H, per)
    off = 0
    for name, width in out_names:
      out[_IPA + name] = {'weights': w[:, :, off:off + width],
                          'bias': b[:, off:off + width]}
      off += width

  _scalar(p[_IPA + 'q_scalar'], [('q_scalar_projection', _SC)])
  _scalar(p[_IPA + 'kv_scalar'],
          [('k_scalar_projection', _SC), ('v_scalar_projection', _SC)])

  # ---- IPA point. Monomer packs points COORD-major: the flat projection is
  # split into 3 coord groups (x,y,z), each (head, npoint); multimer packs them
  # HEAD-major as (head, [x..., y..., z...]). So reshape (D, 3, head, npoint),
  # transpose coord<->head to (D, head, 3, npoint), then flatten -> (D, head,
  # 3*npoint). kv_point splits per head into qk (4) then v (8) points.
  def _point(mod, out_names):
    w, b = _np(mod['weights']), _np(mod['bias'])
    D = w.shape[0]
    npts = w.shape[1] // (3 * _H)
    w = w.reshape(D, 3, _H, npts).transpose(0, 2, 1, 3)   # (D, head, 3, npts)
    b = b.reshape(3, _H, npts).transpose(1, 0, 2)         # (head, 3, npts)
    off = 0
    for name, width in out_names:
      w_i = w[:, :, :, off:off + width].reshape(D, _H, 3 * width)
      b_i = b[:, :, off:off + width].reshape(_H, 3 * width)
      out[_IPA + name] = {'weights': w_i, 'bias': b_i}
      off += width

  _point(p[_IPA + 'q_point_local'],
         [('q_point_projection/point_projection', P_QK)])
  _point(p[_IPA + 'kv_point_local'],
         [('k_point_projection/point_projection', P_QK),
          ('v_point_projection/point_projection', P_V)])

  # ---- affine_update (D,6) -> quat_rigid/rigid (D,6), rename only
  au = p[_SM + 'affine_update']
  out[_SM + 'quat_rigid/rigid'] = {'weights': _np(au['weights']),
                                   'bias': _np(au['bias'])}

  # ---- rel-pos: pair_activiations (65,128) -> ~_relative_encoding/
  #      position_activations (73,128), zero-pad the 8 chain-feature columns
  pa = p[_EVO + 'pair_activiations']
  w = _np(pa['weights'])
  target = 73
  wp = np.concatenate(
      [w, np.zeros((target - w.shape[0], w.shape[1]), w.dtype)], axis=0)
  out[_EVO + '~_relative_encoding/position_activations'] = {
      'weights': wp, 'bias': _np(pa['bias'])}

  # ---- alphabet reduction: drop the LEADING row, not the trailing one.
  # The embedding wrapper pads target_feat/avg_target differently in the two
  # graphs: monomer pads [1,1] (leading AND trailing zero -> 22 wide, real
  # restypes at indices 1..20); multimer pads [0,1] (trailing only -> 21 wide,
  # real restypes at 0..19). So monomer's real-restype weight rows are 1..20 and
  # must map onto multimer's rows 0..19: drop monomer row 0 (the leading pad).
  # Dropping the trailing row instead misaligns every restype by one -- a ~1.0
  # error in the pair init, which is how this was caught.
  for name in ('left_single', 'right_single', 'preprocess_1d'):
    src = p[_EVO + name]
    d = {'weights': _np(src['weights'])[1:]}
    if 'bias' in src:
      d['bias'] = _np(src['bias'])
    out[_EVO + name] = d
  mm = _A + 'masked_msa_head/logits'
  out[mm] = {'weights': _np(p[mm]['weights'])[:, :22],
             'bias': _np(p[mm]['bias'])[:22]}

  return out
