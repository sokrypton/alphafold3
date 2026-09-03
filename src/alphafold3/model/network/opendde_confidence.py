"""OpenDDE confidence head (pLDDT / PAE / PDE / experimentally-resolved).

OpenDDE's confidence head is its own parametrisation, distinct from AF3's: it
initialises the pair rep from s_inputs (row+col), adds a distance embedding of the
predicted structure (binned + raw), runs a 4-block pairformer, and reads out four
heads -- PAE/PDE as linear projections of the pair rep, and pLDDT/resolved as a
per-token-atom einsum against learned weight tensors [24, c_s, bins]. Runs on the
structural-token set (same tokens as the diffusion). Gated by the opendde caller;
nothing here touches AF3/OF3/IF2, which keep confidence_head.ConfidenceHead.

Ref: opendde/model/modules/confidence.py (forward + memory_efficient_forward).
"""

from ..components import haiku_modules as hm
from . import modules as pairformer_modules
from . import diffusion_transformer
import haiku as hk
import jax
import jax.numpy as jnp

# distance_bin_start/end/step from opendde config (3.25, 52.0, 1.25 -> 39 bins).
_BIN_START, _BIN_END, _BIN_STEP = 3.25, 52.0, 1.25


def _dist_onehot(dist):
  """OpenDDE one_hot(dist, lower_bins, upper_bins): (dist>=lo)&(dist<hi) over 39 bins."""
  lower = jnp.arange(_BIN_START, _BIN_END, _BIN_STEP)          # (39,)
  upper = jnp.concatenate([lower[1:], jnp.asarray([1e6], lower.dtype)])
  d = dist[..., None]
  return ((d >= lower) & (d < upper)).astype(jnp.float32)


class OpenDDEConfidenceHead(hk.Module):
  """Structural-token confidence head. c_s/c_z=384, c_s_inputs=447 (our order),
  24 atoms/token, PAE/PDE=64 bins, pLDDT=50, resolved=2."""

  def __init__(self, c_s, c_z, c_s_inputs, global_config, n_blocks=4,
               max_atoms_per_token=24, b_pae=64, b_pde=64, b_plddt=50, b_resolved=2,
               name='confidence_head'):
    super().__init__(name=name)
    self.c_s, self.c_z, self.c_s_inputs = c_s, c_z, c_s_inputs
    self.gc = global_config
    self.n_blocks = n_blocks
    self.max_atoms_per_token = max_atoms_per_token
    self.b_pae, self.b_pde = b_pae, b_pde
    self.b_plddt, self.b_resolved = b_plddt, b_resolved

  def __call__(self, s_inputs, s_trunk, z_trunk, x_pred_rep_coords,
               atom_to_token_idx, atom_to_tokatom_idx, seq_mask, extra_pair_bias=None):
    c_z = self.c_z
    # s_trunk clamped + LN'd; z initialised from s_inputs (row from s2, col from s1).
    s_trunk = hm.LayerNorm(name='input_strunk_ln')(
        jnp.clip(s_trunk, -512.0, 512.0))
    s1 = hm.Linear(c_z, use_bias=False, name='linear_no_bias_s1')(s_inputs)
    s2 = hm.Linear(c_z, use_bias=False, name='linear_no_bias_s2')(s_inputs)
    z = z_trunk + s1[None, :, :] + s2[:, None, :]

    # distance embedding of the predicted structure (binned one-hot + raw scalar).
    dist = jnp.sqrt(jnp.maximum(
        1e-10, jnp.sum((x_pred_rep_coords[:, None, :] - x_pred_rep_coords[None, :, :]) ** 2,
                       axis=-1)))
    z = z + hm.Linear(c_z, use_bias=False, name='linear_no_bias_d')(_dist_onehot(dist))
    z = z + hm.Linear(c_z, use_bias=False, name='linear_no_bias_d_wo_onehot')(
        dist[..., None])

    # 4-block pairformer (single n_heads=16, transitions x4, hidden_scale_up tri-att).
    ref_cfg = pairformer_modules.PairFormerIteration.Config(
        num_layer=1,
        pair_attention=pairformer_modules.GridSelfAttention.Config(num_head=c_z // 32),
        single_attention=diffusion_transformer.SelfAttentionConfig(num_head=16),
        pair_transition=pairformer_modules.TransitionBlock.Config(num_intermediate_factor=4),
        single_transition=pairformer_modules.TransitionBlock.Config(num_intermediate_factor=4))
    ref_cfg.shard_transition_blocks = False
    seq_mask = seq_mask.astype(jnp.float32)
    pair_mask = seq_mask[:, None] * seq_mask[None, :]

    def blk(carry):
      zz, ss = carry
      return pairformer_modules.PairFormerIteration(
          ref_cfg, self.gc, with_single=True, name='trunk_pairformer')(
              zz, pair_mask, single_act=ss, seq_mask=seq_mask,
              extra_pair_bias=extra_pair_bias)

    z, s_single = hk.experimental.layer_stack(
        self.n_blocks, name='pairformer_stack')(blk)((z, s_trunk))

    # PAE from z; PDE from the symmetrised z.
    pae = hm.Linear(self.b_pae, use_bias=False, name='linear_no_bias_pae')(
        hm.LayerNorm(name='pae_ln')(z))
    pde = hm.Linear(self.b_pde, use_bias=False, name='linear_no_bias_pde')(
        hm.LayerNorm(name='pde_ln')(z + jnp.swapaxes(z, -2, -3)))

    # pLDDT / resolved: per-atom einsum of the atom's token single rep against a
    # per-token-atom weight tensor selected by the atom's dense-slot index.
    a = s_single[atom_to_token_idx]                             # (n_atom, c_s)
    plddt_w = hk.get_parameter(
        'plddt_weight', [self.max_atoms_per_token, self.c_s, self.b_plddt],
        jnp.float32, init=hk.initializers.Constant(0.0))
    resolved_w = hk.get_parameter(
        'resolved_weight', [self.max_atoms_per_token, self.c_s, self.b_resolved],
        jnp.float32, init=hk.initializers.Constant(0.0))
    plddt = jnp.einsum('nc,ncb->nb', hm.LayerNorm(name='plddt_ln')(a),
                       plddt_w[atom_to_tokatom_idx])
    resolved = jnp.einsum('nc,ncb->nb', hm.LayerNorm(name='resolved_ln')(a),
                          resolved_w[atom_to_tokatom_idx])
    # REDUCE the logits to scores, under the same key names the shared
    # ConfidenceHead uses. Returning raw logits here (which this head did) is a
    # silent trap: every consumer -- get_ranking_scores, any design loss, the
    # confidence parity screen -- reads `predicted_lddt` as a 0-100 number, and
    # for opendde alone it was getting a (n_atom, 50) logit tensor. Mean over
    # that reads -0.1 on a 1.071 A fold, i.e. outside pLDDT's range entirely.
    #
    # The module gate in tests/test_convert_opendde.py could not see it: it
    # compares this head's output against native's plddt_weight einsum, which is
    # ALSO logits, so logits-vs-logits passed while the end-to-end value was
    # unusable. The logits stay available under *_logits for exactly that gate.
    #
    # Bin parameters are native OpenDDE's (config/model_base.py
    # confidence_configs) and its reduction is
    # sample_confidence.logits_to_score: centres at min + (i + 0.5) * width,
    # softmax-weighted, and pLDDT scaled by 100 (summary_confidence["plddt"] =
    # atom_plddt.mean() * 100).
    #   plddt  50 bins over [0, 1]    -> x100
    #   pae    64 bins over [0, 32]
    #   pde    64 bins over [0, 32]
    def _score(logits, min_bin, max_bin, n_bins):
      width = (max_bin - min_bin) / n_bins
      centers = min_bin + width * (jnp.arange(n_bins, dtype=jnp.float32) + 0.5)
      return jnp.sum(jax.nn.softmax(logits, axis=-1) * centers, axis=-1)

    predicted_lddt = _score(plddt, 0.0, 1.0, self.b_plddt) * 100.0
    full_pae = _score(pae, 0.0, 32.0, self.b_pae)
    full_pde = _score(pde, 0.0, 32.0, self.b_pde)
    # average_pde is masked and normalised the way the shared head does it, so
    # ranking is comparable across models. pair_mask is the one built above.
    full_pde = full_pde * pair_mask
    average_pde = (jnp.sum(full_pde, axis=[-2, -1])
                   / jnp.maximum(jnp.sum(pair_mask, axis=[-2, -1]), 1.0))
    return {
        'predicted_lddt': predicted_lddt,
        'full_pae': full_pae,
        'full_pde': full_pde,
        'average_pde': average_pde,
        'predicted_experimentally_resolved': jax.nn.softmax(resolved,
                                                            axis=-1)[..., 1],
        # the raw logits, for the module-level gate against native's own einsums
        'predicted_lddt_logits': plddt,
        'predicted_aligned_error_logits': pae,
        'predicted_distance_error_logits': pde,
        'experimentally_resolved_logits': resolved,
    }
