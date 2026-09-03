"""OpenDDE structural-token expansion.

OpenDDE is not vanilla AF3: between the residue-level pairformer trunk and the
diffusion module it expands each residue into ~2 "structural tokens" (a backbone
token + a sidechain token; glycine stays a single token), and runs the diffusion
and confidence heads on that expanded token set. This module ports OpenDDE's
`StructuralTokenExpander` (opendde/model/modules/structural_tokens.py) to haiku.

The expander gathers the residue-level single/pair representations onto the
structural tokens (each token copies its parent residue's rep), adds learned
role embeddings, a per-role-pair linear projection of the pair rep, learned pair
init biases from structural relationships, and a learned scalar attention bias.
It is gated by global_config.opendde; nothing here runs for AF3/OF3/IF2.

The role ids follow OpenDDE's STRUCTURAL_TOKEN_ROLES:
    atom=0 protein_bb=1 protein_sc=2 dna_bb=3 dna_base=4 rna_bb=5 rna_base=6
"""

from . import featurization  # noqa: F401  (kept for parity with sibling modules)
from ..components import haiku_modules as hm
import haiku as hk
import jax
import jax.numpy as jnp


N_ROLES = 7          # max(STRUCTURAL_TOKEN_ROLES.values()) + 1
_BACKBONE_ROLES = (1, 3, 5)   # protein_bb, dna_bb, rna_bb
_SIDECHAIN_ROLE = 2           # protein_sc
_BASE_ROLES = (4, 6)          # dna_base, rna_base


def _embed(table_name, n, c, idx):
  """Learned embedding lookup: a (n, c) table indexed by integer idx."""
  table = hk.get_parameter(
      table_name, shape=[n, c], dtype=jnp.float32,
      init=hk.initializers.Constant(0.0))
  return table[idx]


class StructuralTokenExpander(hk.Module):
  """Residue-level (s_inputs, s, z) -> structural-token-level, OpenDDE v1.

  pair_projection_mode='full': one LinearNoBias(c_z, c_z) per (role_i, role_j)
  pair, i.e. N_ROLES**2 = 49 matrices, selected per token pair.
  """

  def __init__(self, c_s, c_z, c_s_inputs, global_config, name='structural_token_expander'):
    super().__init__(name=name)
    self.c_s = c_s
    self.c_z = c_z
    self.c_s_inputs = c_s_inputs
    self.global_config = global_config

  @hk.transparent
  def _single(self, s_inputs_res, s_res, parent, role):
    # s_inputs_struct = s_inputs_res[parent] + role_emb(role)
    s_inputs_struct = s_inputs_res[parent] + _embed(
        'single_input_role_embedding', N_ROLES, self.c_s_inputs, role)
    # s_struct = s_parent + split_mlp(s_parent) + role_emb(role)
    s_parent = s_res[parent]
    h = hm.LayerNorm(name='single_split_norm')(s_parent)
    h = hm.Linear(2 * self.c_s, initializer='relu', name='single_split_1')(h)
    h = jax.nn.silu(h)
    h = hm.Linear(self.c_s, initializer=self.global_config.final_init,
                  name='single_split_2')(h)
    s_struct = s_parent + h + _embed('single_role_embedding', N_ROLES, self.c_s, role)
    return s_inputs_struct, s_struct

  def _pair_features(self, batch_struct, parent, role):
    """Boolean/int structural pair features, from parent/role + adjacency."""
    residue_index = batch_struct['residue_index'][parent]
    asym_id = batch_struct['asym_id'][parent]
    prev_parent = batch_struct['prev_parent_residue_idx']
    next_parent = batch_struct['next_parent_residue_idx']

    is_bb = (role == _BACKBONE_ROLES[0]) | (role == _BACKBONE_ROLES[1]) | (
        role == _BACKBONE_ROLES[2])
    is_sc = role == _SIDECHAIN_ROLE
    is_base = (role == _BASE_ROLES[0]) | (role == _BASE_ROLES[1])

    same_parent = parent[:, None] == parent[None, :]
    same_chain = asym_id[:, None] == asym_id[None, :]
    twin = same_parent & (
        (is_bb[:, None] & (is_sc[None, :] | is_base[None, :]))
        | (is_bb[None, :] & (is_sc[:, None] | is_base[:, None])))
    prev_bb = is_bb[:, None] & is_bb[None, :] & same_chain & (
        prev_parent[:, None] == parent[None, :])
    next_bb = is_bb[:, None] & is_bb[None, :] & same_chain & (
        next_parent[:, None] == parent[None, :])

    # role_pair_type: default 7; bb-bb=0 bb-sc=1 sc-bb=2 sc-sc=3 bb-base=4
    # base-bb=5 base-base=6
    rpt = jnp.full(same_parent.shape, 7, dtype=jnp.int32)
    rpt = jnp.where(is_bb[:, None] & is_bb[None, :], 0, rpt)
    rpt = jnp.where(is_bb[:, None] & is_sc[None, :], 1, rpt)
    rpt = jnp.where(is_sc[:, None] & is_bb[None, :], 2, rpt)
    rpt = jnp.where(is_sc[:, None] & is_sc[None, :], 3, rpt)
    rpt = jnp.where(is_bb[:, None] & is_base[None, :], 4, rpt)
    rpt = jnp.where(is_base[:, None] & is_bb[None, :], 5, rpt)
    rpt = jnp.where(is_base[:, None] & is_base[None, :], 6, rpt)
    del residue_index
    return {'same_parent': same_parent, 'twin': twin, 'prev_bb': prev_bb,
            'next_bb': next_bb, 'role_pair_type': rpt}

  def _pair(self, z_res, parent, role, pf):
    z_parent = z_res[parent][:, parent]                      # (S, S, c_z)

    # Per-(role_i, role_j) linear projection: 49 matrices selected by role pair.
    # delta[i,j] = z[i,j] @ W[role_pair(i,j)]. Computed as a rolled fori_loop that
    # masks z to the k-th role pair BEFORE projecting and accumulates -- identical to
    # the select form (disjoint masks; a zeroed row projects to zero) but keeps only ONE
    # (S,S,c_z) projection live at a time. The naive unrolled `where(role==k, z@W[k], .)`
    # let XLA fuse all 49 projections into one select_n, blowing up peak memory ~49x
    # (the opendde structural-token OOM at S~512).
    W = hk.get_parameter(
        'pair_block_proj', shape=[N_ROLES * N_ROLES, self.c_z, self.c_z],
        dtype=jnp.float32, init=hk.initializers.Constant(0.0))
    role_pair_idx = role[:, None] * N_ROLES + role[None, :]  # (S, S)

    def _accum(k, delta):
      zk = jnp.where((role_pair_idx == k)[..., None], z_parent, 0.0)
      return delta + jnp.einsum('ijc,cd->ijd', zk, W[k])

    delta = jax.lax.fori_loop(
        0, N_ROLES * N_ROLES, _accum, jnp.zeros_like(z_parent))
    z_struct = z_parent + delta

    # Learned pair init bias from structural relationships.
    z_struct = z_struct + _embed('same_parent_embedding', 2, self.c_z, pf['same_parent'].astype(jnp.int32))
    z_struct = z_struct + _embed('same_residue_twin_embedding', 2, self.c_z, pf['twin'].astype(jnp.int32))
    z_struct = z_struct + _embed('prev_bb_chain_embedding', 2, self.c_z, pf['prev_bb'].astype(jnp.int32))
    z_struct = z_struct + _embed('next_bb_chain_embedding', 2, self.c_z, pf['next_bb'].astype(jnp.int32))
    z_struct = z_struct + _embed('role_pair_type_embedding', 8, self.c_z, pf['role_pair_type'])
    return z_struct

  def _attn_bias(self, pf):
    def scalar(nm):
      return hk.get_parameter(nm, shape=[], dtype=jnp.float32,
                              init=hk.initializers.Constant(0.0))
    role_tab = hk.get_parameter('attn_bias_role_pair_type', shape=[8],
                                dtype=jnp.float32, init=hk.initializers.Constant(0.0))
    return (scalar('attn_bias_same_parent') * pf['same_parent'].astype(jnp.float32)
            + scalar('attn_bias_same_residue_twin') * pf['twin'].astype(jnp.float32)
            + scalar('attn_bias_prev_bb_chain') * pf['prev_bb'].astype(jnp.float32)
            + scalar('attn_bias_next_bb_chain') * pf['next_bb'].astype(jnp.float32)
            + role_tab[pf['role_pair_type']])

  def __call__(self, batch_struct, s_inputs_res, s_res, z_res):
    """Expand residue-level reps to structural tokens.

    Args:
      batch_struct: dict with per-structural-token int arrays 'parent_residue_idx',
        'subtoken_role_id', 'prev_parent_residue_idx', 'next_parent_residue_idx',
        and per-residue 'residue_index', 'asym_id' (indexed by parent).
      s_inputs_res: (n_res, c_s_inputs)
      s_res: (n_res, c_s)
      z_res: (n_res, n_res, c_z)

    Returns:
      (s_inputs_struct, s_struct, z_struct, structural_pair_attn_bias)
    """
    parent = batch_struct['parent_residue_idx'].astype(jnp.int32)
    role = batch_struct['subtoken_role_id'].astype(jnp.int32)
    s_inputs_struct, s_struct = self._single(s_inputs_res, s_res, parent, role)
    pf = self._pair_features(batch_struct, parent, role)
    z_struct = self._pair(z_res, parent, role, pf)
    attn_bias = self._attn_bias(pf)
    return s_inputs_struct, s_struct, z_struct, attn_bias
