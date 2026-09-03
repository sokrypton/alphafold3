# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Modules for embedding templates."""

from alphafold3.common import base_config
from alphafold3.constants import residue_names
from alphafold3.jax import geometry
from alphafold3.model import features
from alphafold3.model import model_config
from alphafold3.model import protein_data_processing
from alphafold3.model.components import haiku_modules as hm
from . import modules
from alphafold3.model.scoring import scoring
import haiku as hk
import jax
import jax.numpy as jnp


class DistogramFeaturesConfig(base_config.BaseConfig):
  # The left edge of the first bin.
  min_bin: float = 3.25
  # The left edge of the final bin. The final bin catches everything larger than
  # `max_bin`.
  max_bin: float = 50.75
  # The number of bins in the distogram.
  num_bins: int = 39


def dgram_from_positions(positions, config: DistogramFeaturesConfig):
  """Compute distogram from amino acid positions.

  Args:
    positions: (num_res, 3) Position coordinates.
    config: Distogram bin configuration.

  Returns:
    Distogram with the specified number of bins.
  """
  lower_breaks = jnp.linspace(config.min_bin, config.max_bin, config.num_bins)
  lower_breaks = jnp.square(lower_breaks)
  upper_breaks = jnp.concatenate(
      [lower_breaks[1:], jnp.array([1e8], dtype=jnp.float32)], axis=-1
  )
  dist2 = jnp.sum(
      jnp.square(
          jnp.expand_dims(positions, axis=-2)
          - jnp.expand_dims(positions, axis=-3)
      ),
      axis=-1,
      keepdims=True,
  )

  dgram = (dist2 > lower_breaks).astype(jnp.float32) * (
      dist2 < upper_breaks
  ).astype(jnp.float32)
  return dgram


def make_backbone_rigid(
    positions: geometry.Vec3Array,
    mask: jnp.ndarray,
    group_indices: jnp.ndarray,
) -> tuple[geometry.Rigid3Array, jnp.ndarray]:
  """Make backbone Rigid3Array and mask.

  Args:
    positions: (num_res, num_atoms) of atom positions as Vec3Array.
    mask: (num_res, num_atoms) for atom mask.
    group_indices: (num_res, num_group, 3) for atom indices forming groups.

  Returns:
    tuple of backbone Rigid3Array and mask (num_res,).
  """
  backbone_indices = group_indices[:, 0]

  # main backbone frames differ in sidechain frame convention.
  # for sidechain it's (C, CA, N), for backbone it's (N, CA, C)
  # Hence using c, b, a, each of shape (num_res,).
  c, b, a = [backbone_indices[..., i] for i in range(3)]

  slice_index = jax.vmap(lambda x, i: x[i])
  rigid_mask = (
      slice_index(mask, a) * slice_index(mask, b) * slice_index(mask, c)
  ).astype(jnp.float32)

  frame_positions = []
  for indices in [a, b, c]:
    frame_positions.append(
        jax.tree.map(lambda x, idx=indices: slice_index(x, idx), positions)
    )

  rotation = geometry.Rot3Array.from_two_vectors(
      frame_positions[2] - frame_positions[1],
      frame_positions[0] - frame_positions[1],
  )
  rigid = geometry.Rigid3Array(rotation, frame_positions[1])  # pyrefly: ignore[bad-argument-count]

  return rigid, rigid_mask


class TemplateEmbedding(hk.Module):
  """Embed a set of templates."""

  class Config(base_config.BaseConfig):
    num_channels: int = 64
    template_stack: modules.PairFormerIteration.Config = base_config.autocreate(
        num_layer=2,
        pair_transition=base_config.autocreate(num_intermediate_factor=2),
    )
    dgram_features: DistogramFeaturesConfig = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='template_embedding',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(
      self,
      query_embedding: jnp.ndarray,
      templates: features.Templates,
      padding_mask_2d: jnp.ndarray,
      multichain_mask_2d: jnp.ndarray,
      key: jnp.ndarray,
      use_dropout=False,
  ) -> jnp.ndarray:
    """Generate an embedding for a set of templates.

    Args:
      query_embedding: [num_res, num_res, num_channel] a query tensor that will
        be used to attend over the templates to remove the num_templates
        dimension.
      templates: A 'Templates' object.
      padding_mask_2d: [num_res, num_res] Pair mask for attention operations.
      multichain_mask_2d: [num_res, num_res] Pair mask for multichain.
      key: random key generator.

    Returns:
      An embedding of size [num_res, num_res, num_channels]
    """
    c = self.config
    num_residues = query_embedding.shape[0]
    num_templates = templates.aatype.shape[0]
    query_num_channels = query_embedding.shape[2]
    num_atoms = 24
    assert query_embedding.shape == (
        num_residues,
        num_residues,
        query_num_channels,
    )
    assert templates.aatype.shape == (num_templates, num_residues)
    assert templates.atom_positions.shape == (
        num_templates,
        num_residues,
        num_atoms,
        3,
    )
    assert templates.atom_mask.shape == (num_templates, num_residues, num_atoms)
    assert padding_mask_2d.shape == (num_residues, num_residues)

    num_templates = templates.aatype.shape[0]
    num_res, _, query_num_channels = query_embedding.shape

    # Embed each template separately.
    template_embedder = SingleTemplateEmbedding(self.config, self.global_config)

    subkeys = jnp.array(jax.random.split(key, num_templates))

    def scan_fn(carry, x):
      templates, key = x
      embedding = template_embedder(
          query_embedding,
          templates,
          padding_mask_2d,
          multichain_mask_2d,
          key,
          use_dropout,
      )
      return carry + embedding, None

    scan_init = jnp.zeros(
        (num_res, num_res, c.num_channels), dtype=query_embedding.dtype
    )
    summed_template_embeddings, _ = hk.scan(
        scan_fn, scan_init, (templates, subkeys)
    )

    # AF3 divides by the number of template SLOTS, and the pipeline always pads
    # to max_templates=4 (features.py _pad_to), so one real template is summed
    # over 4 slots and divided by 4. chai divides by clamp_min(n_templates, 1)
    # -- the count of templates it actually has; the clamp only makes sense for
    # a real count, since a padded one is never 0. With the per-template mask
    # applied above the empty slots contribute exactly zero, so the two differ
    # by a pure scale: 1 real template of 4 slots reaches the trunk at a QUARTER
    # of chai's amplitude. Invisible on the 4-real-template oracle capture,
    # which is why the trunk gate could not see it.
    denom = 1e-7 + num_templates
    if self.global_config.model == 'chai1':
      present = (templates.atom_mask.reshape(num_templates, -1).sum(-1) > 0)
      denom = jnp.maximum(present.astype(query_embedding.dtype).sum(), 1.0)
    embedding = summed_template_embeddings / denom
    embedding = jax.nn.relu(embedding)
    embedding = hm.Linear(
        query_num_channels, initializer='relu', name='output_linear'
    )(embedding)

    assert embedding.shape == (num_residues, num_residues, query_num_channels)
    return embedding


class SingleTemplateEmbedding(hk.Module):
  """Embed a single template."""

  def __init__(
      self,
      config: TemplateEmbedding.Config,
      global_config: model_config.GlobalConfig,
      name='single_template_embedding',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(
      self,
      query_embedding: jnp.ndarray,
      templates: features.Templates,
      padding_mask_2d: jnp.ndarray,
      multichain_mask_2d: jnp.ndarray,
      key: jnp.ndarray,
      use_dropout=False,
  ) -> jnp.ndarray:
    """Build the single template embedding graph.

    Args:
      query_embedding: (num_res, num_res, num_channels) - embedding of the query
        sequence/msa.
      templates: 'Templates' object containing single Template.
      padding_mask_2d: Padding mask (Note: this doesn't care if a template
        exists, unlike the template_pseudo_beta_mask).
      multichain_mask_2d: A mask indicating intra-chain residue pairs, used to
        mask out between chain distances/features when templates are for single
        chains.
      key: Random key generator.

    Returns:
      A template embedding (num_res, num_res, num_channels).
    """
    gc = self.global_config
    c = self.config
    assert padding_mask_2d.dtype == query_embedding.dtype
    dtype = query_embedding.dtype
    num_channels = self.config.num_channels

    def construct_input(
        query_embedding, templates: features.Templates, multichain_mask_2d
    ):

      # Compute distogram feature for the template.
      aatype = templates.aatype
      dense_atom_mask = templates.atom_mask

      dense_atom_positions = templates.atom_positions
      dense_atom_positions *= dense_atom_mask[..., None]

      pseudo_beta_positions, pseudo_beta_mask = scoring.pseudo_beta_fn(
          templates.aatype, dense_atom_positions, dense_atom_mask
      )
      pseudo_beta_mask_2d = (
          pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
      )
      pseudo_beta_mask_2d *= multichain_mask_2d
      if self.global_config.model == 'chai1':
        # chai's TemplateDistogramGenerator is NOT AF3's dgram_from_positions.
        # Two differences, both in chai_lab source
        # (data/features/generators/templates.py) and both visible in its
        # captured feature stream:
        #   * 38 distance classes from searchsorted over
        #     linspace(3.25, 50.75, 38)[1:] -- 37 thresholds. AF3 uses 39 evenly
        #     spaced bins over the same range, so the two disagree by about one
        #     bin through the middle of the range and more at the top.
        #   * class 38 is a MASK class (can_mask=True -> mask_value=num_classes),
        #     assigned to pairs the template does not cover and, with
        #     allow_inter_chain=False, to every cross-chain pair. AF3 instead
        #     multiplies the distogram by pseudo_beta_mask_2d, i.e. feeds an
        #     ALL-ZERO row there, so our column 38 was never once activated.
        # Confirmed in ~/chai_1stp_tmpl: at pairs whose template mask is 0 the
        # native classes are 38 (3872 of 4369), and the padding region is 0.
        d = jnp.sqrt(jnp.sum(jnp.square(pseudo_beta_positions[:, None]
                                        - pseudo_beta_positions[None, :]), -1)
                     + 1e-10)
        edges = jnp.linspace(3.25, 50.75, 38)[1:]
        cls = jnp.sum(d[..., None] > edges, -1)
        cls = jnp.where(pseudo_beta_mask_2d > 0, cls, 38)
        dgram = jax.nn.one_hot(cls, 39, dtype=dtype)
      else:
        dgram = dgram_from_positions(
            pseudo_beta_positions, self.config.dgram_features
        )
        dgram *= pseudo_beta_mask_2d[..., None]
      dgram = dgram.astype(dtype)
      pseudo_beta_mask_2d = pseudo_beta_mask_2d.astype(dtype)
      to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]

      if self.global_config.model == 'chai1':
        # chai marks template residues the template does not COVER as the gap
        # restype, not as the query's own residue: its captured
        # TemplateResType is class 31 = '-' in
        # residue_types_with_nucleotides_order at exactly the uncovered
        # positions. CHAI1_RESTYPE_PERM maps that onto our index 21, also '-'.
        # AF3 passes templates.aatype straight through, which tells the
        # embedder a residue identity for positions that have no structure.
        covered = dense_atom_mask.sum(-1) > 0
        aatype = jnp.where(covered, aatype, 21)
      aatype = jax.nn.one_hot(
          aatype,
          residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
          axis=-1,
          dtype=dtype,
      )
      to_concat.append((aatype[None, :, :], 1))
      to_concat.append((aatype[:, None, :], 1))

      # Compute a feature representing the normalized vector between each
      # backbone affine - i.e. in each residues local frame, what direction are
      # each of the other residues.

      template_group_indices = jnp.take(
          protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX,
          templates.aatype,
          axis=0,
      )
      rigid, backbone_mask = make_backbone_rigid(
          geometry.Vec3Array.from_array(dense_atom_positions),
          dense_atom_mask,
          template_group_indices.astype(jnp.int32),
      )
      points = rigid.translation
      rigid_vec = rigid[:, None].inverse().apply_to_point(points)  # pyrefly: ignore[bad-index]
      unit_vector = rigid_vec.normalized()
      unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

      unit_vector = [x.astype(dtype) for x in unit_vector]
      backbone_mask = backbone_mask.astype(dtype)

      backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
      backbone_mask_2d *= multichain_mask_2d
      unit_vector = [x * backbone_mask_2d for x in unit_vector]

      # Note that the backbone_mask takes into account C, CA and N (unlike
      # pseudo beta mask which just needs CB) so we add both masks as features.
      to_concat.extend([(x, 0) for x in unit_vector])
      to_concat.append((backbone_mask_2d, 0))

      query_embedding = hm.LayerNorm(name='query_embedding_norm')(
          query_embedding
      )
      # Allow the template embedder to see the query embedding.  Note this
      # contains the position relative feature, so this is how the network knows
      # which residues are next to each other.
      to_concat.append((query_embedding, 1))

      # chai's template aggregation multiplies each template's LayerNorm'd
      # embedding by that template's pair mask before summing, which AF3 does
      # not -- and it is not a no-op, because LayerNorm has a bias, so pairs the
      # template does not cover are NONZERO after it. Read off chai's trunk
      # graph (sum(LN(v) * template_mask) / clamp_min(n_templates, 1)) and
      # confirmed by measurement: applying it moves the 1STP 4-template trunk
      # from s 0.999792 / z 0.999562 to s 0.999851 / z 0.999654 -- small, but
      # the same direction on BOTH tracks (tools/oracles/chai1/
      # cmp_trunk_templates.py, TMASK). chai's own mask is
      # outer(covered_i, covered_j) & same_asym, which is what
      # pseudo_beta_mask_2d already is here.
      self._chai1_mask_2d = pseudo_beta_mask_2d

      act = 0

      for i, (x, n_input_dims) in enumerate(to_concat):
        act += hm.Linear(
            num_channels,
            num_input_dims=n_input_dims,
            initializer='relu',
            name=f'template_pair_embedding_{i}',
        )(x)
      return act

    act = construct_input(query_embedding, templates, multichain_mask_2d)

    if c.template_stack.num_layer:

      def template_iteration_fn(x):
        return modules.PairFormerIteration(
            c.template_stack, gc, name='template_embedding_iteration'
        )(act=x, pair_mask=padding_mask_2d, use_dropout=use_dropout)

      template_stack = hk.experimental.layer_stack(c.template_stack.num_layer)(
          template_iteration_fn
      )
      act = template_stack(act)

    act = hm.LayerNorm(name='output_layer_norm')(act)
    if self.global_config.model == 'chai1':
      act = act * self._chai1_mask_2d[..., None].astype(act.dtype)
    return act


class Boltz2TemplateEmbedding(hk.Module):
  """Boltz-2 template module (gated by global_config.boltz2), ported and validated
  bit-close (corr 1.0) against Boltz's TemplateModule (trunkv2.py).

  Differs from AF3's TemplateEmbedding in both featurisation and forward:
    * features (109-d): distogram(38, CB-CB, linspace(3.25,50.75,37)) + cb_mask(1) +
      unit_vector(3, per-residue backbone frame) + frame_mask(1), * asym_mask, then
      res_type_i(33) + res_type_j(33) one-hots.
    * frame = Boltz compute_frame(N,CA,C): e1=(C-CA)^, e2=orth(N-CA)^, e3=e1xe2, t=CA.
      unit_vector[i,j] = R_j^T @ (ca_i - t_j) (exact matmul).
    * res_type token = aatype + 2 (AF3->Boltz vocab), token 0 where not covered.
    * forward: v = z_proj(z_norm(z)) + a_proj(a_tij); v = v + pairformer(v) x2;
      v = v_norm(v); aggregate over templates (mean of present); u = u_proj(relu(u)).
  N/CA/C come from RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX group 0, whose atom order is
  [C, CA, N] (not [N, CA, C]); CB via pseudo_beta_fn.
  """

  def __init__(self, config, global_config, name='template_embedding'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def _features(self, aatype, atom_positions, atom_mask, asym_mask_2d):
    """Build the 109-d a_tij for one template. Shapes: aatype (N,), positions
    (N,24,3), atom_mask (N,24), asym_mask_2d (N,N). Returns (N,N,109)."""
    N = aatype.shape[0]
    covered = (atom_mask.sum(-1) > 0)                                   # (N,)
    # CB (pseudo-beta) + mask
    cb, cb_mask = scoring.pseudo_beta_fn(aatype, atom_positions, atom_mask, use_jax=True)
    # backbone N/CA/C via group 0 (order [C, CA, N])
    grp = jnp.take(protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX,
                   aatype, axis=0).astype(jnp.int32)                    # (N, G, 3)
    bb = grp[:, 0]                                                      # (N,3) = [C,CA,N]
    take = jax.vmap(lambda x, i: x[i])
    c_at = take(atom_positions, bb[:, 0]); ca = take(atom_positions, bb[:, 1])
    n_at = take(atom_positions, bb[:, 2])
    frame_mask = (take(atom_mask, bb[:, 0]) * take(atom_mask, bb[:, 1])
                  * take(atom_mask, bb[:, 2])).astype(jnp.float32)      # (N,)
    # Boltz compute_frame: e1=(C-CA)^, e2=orth(N-CA)^, e3=e1xe2; rot cols [e1,e2,e3]; t=CA
    v1 = c_at - ca; v2 = n_at - ca
    e1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-10)
    u2 = v2 - e1 * jnp.sum(e1 * v2, axis=-1, keepdims=True)
    e2 = u2 / (jnp.linalg.norm(u2, axis=-1, keepdims=True) + 1e-10)
    e3 = jnp.cross(e1, e2)
    rot = jnp.stack([e1, e2, e3], axis=-1)                             # (N,3,3), columns e1,e2,e3
    t = ca                                                             # (N,3)
    # distogram (38) over CB-CB
    cb_d = jnp.sqrt(jnp.sum((cb[:, None] - cb[None]) ** 2, -1) + 1e-10)
    bnd = jnp.linspace(3.25, 50.75, 37)
    disto = jax.nn.one_hot((cb_d[..., None] > bnd).sum(-1), 38)         # (N,N,38)
    # unit_vector[i,j] = R_j^T @ (ca_i - t_j)
    diff = ca[:, None, :] - t[None, :, :]                              # (N,N,3): ca_i - t_j
    uvec = jnp.einsum('jdk,ijd->ijk', rot, diff)                       # R_j^T @ diff (rot[:,d,k]=e_k[d])
    # Boltz normalizes with torch.norm(vector, dim=-1) where vector is (...,3,1); dim=-1 is
    # the SIZE-1 axis, so it is abs() per component -> the feature is the element-wise SIGN
    # of R_j^T @ (ca_i - t_j), not a unit 3-vector. Replicate that (trained weights depend on it).
    uvec = jnp.sign(uvec)                                             # (N,N,3), components in {-1,0,1}
    b_cb = (cb_mask[:, None] * cb_mask[None])[..., None]
    b_fr = (frame_mask[:, None] * frame_mask[None])[..., None]
    a_geo = jnp.concatenate([disto, b_cb, uvec, b_fr], -1) * asym_mask_2d[..., None]
    tok = jnp.where(covered, aatype + 2, 0)
    rt = jax.nn.one_hot(tok, 33)                                       # (N,33)
    rt_i = jnp.broadcast_to(rt[:, None, :], (N, N, 33))
    rt_j = jnp.broadcast_to(rt[None, :, :], (N, N, 33))
    return jnp.concatenate([a_geo, rt_i, rt_j], -1)                    # (N,N,109)

  def __call__(self, query_embedding, templates, padding_mask_2d,
               multichain_mask_2d, key, use_dropout=False):
    c = self.config
    gc = self.global_config
    dtype = query_embedding.dtype
    z = query_embedding
    N = z.shape[0]
    aatype = templates.aatype
    atom_positions = templates.atom_positions.astype(jnp.float32)
    atom_mask = templates.atom_mask
    T = aatype.shape[0]

    z_part = hm.Linear(c.num_channels, use_bias=False, name='z_proj')(
        hm.LayerNorm(name='z_norm', use_fast_variance=False)(z))       # (N,N,64)
    pair_mask = padding_mask_2d

    def per_template(aa, pos, am):
      a_tij = self._features(aa, pos, am, multichain_mask_2d).astype(dtype)
      v = z_part + hm.Linear(c.num_channels, use_bias=False, name='a_proj')(a_tij)

      def block(x):
        return modules.PairFormerIteration(
            c.template_stack, gc, name='tmpl_pairformer')(
                act=x, pair_mask=pair_mask, use_dropout=use_dropout)
      pf = hk.experimental.layer_stack(c.template_stack.num_layer)(block)(v)
      v = v + pf
      return hm.LayerNorm(name='v_norm', use_fast_variance=False)(v)   # (N,N,64)

    v_all = hk.vmap(per_template, in_axes=0, out_axes=0,
                    split_rng=False)(aatype, atom_positions, atom_mask)  # (T,N,N,64)
    present = (atom_mask.reshape(T, -1).sum(-1) > 0).astype(v_all.dtype)  # (T,)
    num_t = jnp.clip(present.sum(), 1.0, None)
    u = jnp.einsum('t,tijc->ijc', present, v_all) / num_t             # (N,N,64)
    out = hm.Linear(z.shape[-1], use_bias=False, name='u_proj')(jax.nn.relu(u))
    return out.astype(dtype)


class Protenix2TemplateEmbedding(Boltz2TemplateEmbedding):
  """Protenix-v2 template module: identical fused forward to Boltz2TemplateEmbedding
  (z_proj/a_proj/tmpl_pairformer/v_norm/u_proj -- same converter scopes) but a
  108-d feature builder matching Protenix's featurizer (template_utils.py), VALIDATED
  exact vs native geometry:
    * dgram(39): squared-range one-hot over linspace(3.25,50.75,39)^2 of CB-CB dist^2,
      masked by pb_mask_i*pb_mask_j.
    * unit_vector(3): frame at residue i (e1=(C-CA)^, e2=orth(N-CA)^, e3=e1xe2),
      uv[i,j] = normalize(R_i^T @ (ca_j - ca_i)) -- a REAL unit vector (NOT Boltz's
      sign quirk), masked by frame2d * asym.
    * restype_i/j(32): one_hot(aatype, 32) in OF3/Protenix class order (AF3->OF3 remap;
      same remap the 1.58A fold validated for s_inputs restype).
    * order: [dgram(39), pb_mask(1), restype_i(32), restype_j(32), uvec(3), bb_mask(1)].
  N/CA/C via RESTYPE_RIGIDGROUP group 0 (order [C,CA,N]); CB via pseudo_beta_fn.
  """

  # AF3 aatype index -> OF3/Protenix 32-class index (matches of3._AF3_TO_OF3_AATYPE).
  _AF3_TO_OF3 = (tuple(range(21)) + (31,) + (21, 22, 23, 24) + (26, 27, 28, 29) + (25,))

  def _features(self, aatype, atom_positions, atom_mask, asym_mask_2d):
    N = aatype.shape[0]
    cb, cb_mask = scoring.pseudo_beta_fn(aatype, atom_positions, atom_mask, use_jax=True)
    grp = jnp.take(protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX,
                   aatype, axis=0).astype(jnp.int32)
    bb = grp[:, 0]                                                    # [C, CA, N]
    take = jax.vmap(lambda x, i: x[i])
    c_at = take(atom_positions, bb[:, 0]); ca = take(atom_positions, bb[:, 1])
    n_at = take(atom_positions, bb[:, 2])
    frame_mask = (take(atom_mask, bb[:, 0]) * take(atom_mask, bb[:, 1])
                  * take(atom_mask, bb[:, 2])).astype(jnp.float32)    # (N,)
    eps = 1e-6
    v1 = c_at - ca; v2 = n_at - ca
    e1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + eps)
    u2 = v2 - e1 * jnp.sum(e1 * v2, axis=-1, keepdims=True)
    e2 = u2 / (jnp.linalg.norm(u2, axis=-1, keepdims=True) + eps)
    e3 = jnp.cross(e1, e2)
    rot = jnp.stack([e1, e2, e3], axis=-1)                           # (N,3,3), cols [e1,e2,e3]
    diff = ca[None, :, :] - ca[:, None, :]                           # (N,N,3): ca_j - ca_i
    uvec = jnp.einsum('ilk,ijl->ijk', rot, diff)                     # R_i^T @ (ca_j - ca_i)
    uvec = uvec / (jnp.linalg.norm(uvec, axis=-1, keepdims=True) + eps)
    cb_d2 = jnp.sum((cb[:, None] - cb[None]) ** 2, -1)[..., None]
    lower = jnp.square(jnp.linspace(3.25, 50.75, 39))
    upper = jnp.concatenate([lower[1:], jnp.asarray([1e8], lower.dtype)])
    disto = ((cb_d2 > lower) & (cb_d2 < upper)).astype(jnp.float32)   # (N,N,39)
    pb2d = cb_mask[:, None] * cb_mask[None]                           # pb_mask_i * pb_mask_j
    fr2d = frame_mask[:, None] * frame_mask[None]
    # featurizer masks dgram by pb2d; the FORWARD then masks it by multichain too
    # (protenix pairformer.py: dgram = dgram * multichain_mask * pair_mask).
    # Missing the multichain half is invisible on a monomer -- multichain_mask is
    # all ones there -- and actively harmful on a complex: a distogram one-hot is
    # nonzero for EVERY pair, because a distance always lands in some bin, so
    # unmasked cross-chain entries are not zeros but confident FABRICATED
    # inter-chain distances from a template that carries no inter-chain
    # information. Measured: with them, a template made the 146+74 heterodimer
    # WORSE (interface 31.76 -> 46.76 A) while the same template rescues four
    # other ports to ~1-2 A, and the same embedder rescues a monomer
    # (streptavidin 6.47 -> 3.37).
    disto = disto * (pb2d * asym_mask_2d)[..., None]
    uvec = uvec * (fr2d * asym_mask_2d)[..., None]                    # featurizer(fr2d) * forward(asym)
    pb_ch = (pb2d * asym_mask_2d)[..., None]
    bb_ch = (fr2d * asym_mask_2d)[..., None]
    remap = jnp.asarray(self._AF3_TO_OF3, jnp.int32)
    rt = jax.nn.one_hot(remap[aatype], 32)                           # (N,32)
    rt_i = jnp.broadcast_to(rt[:, None, :], (N, N, 32))
    rt_j = jnp.broadcast_to(rt[None, :, :], (N, N, 32))
    return jnp.concatenate([disto, pb_ch, rt_i, rt_j, uvec, bb_ch], -1)  # (N,N,108)


class RoseTTAFold3TemplateEmbedding(Boltz2TemplateEmbedding):
  """RoseTTAFold3 template module. RF3's "template" is NOT a coordinate-geometry
  embedder (no unit_vector / frame / restype like Boltz2/Protenix) -- it is a
  DISTANCE-DISTRIBUTION CONDITIONING module (src/rf3/.../pairformer_layers.py +
  ground_truth_template.py). It rides the same fused forward as Boltz2 (z_proj /
  a_proj / tmpl_pairformer x2 / v_norm / u_proj -- identical converter scopes;
  emb_templ -> a_proj is 64x66) but with a 66-d feature:

    a_tij = cat[ distogram_condition(64), has_condition(1), joint_noise_level(1) ]
            * has_condition

    * distogram_condition[i,j,64]: one-hot of the pairwise TOKEN-CENTER (CA for
      protein) distance, bucketized into 64 bins via RF3 DEFAULT_DISTOGRAM_BINS =
      concat[arange(1.0,4.0,0.1) (0.1A res, 1-4A), arange(4.0,20.5,0.5) (0.5A res,
      4-20A)] = 63 boundaries; NaN / out-of-range -> last bin. CLOSE range (1-20A).
    * has_condition[i,j]: 1 where the template covers the pair (CA present on both,
      intra-chain); masks the whole feature.
    * joint_noise_level: af3_noise_scale_to_noise_level(sqrt(ns_i^2+ns_j^2)); for an
      EXACT template noise_scale=0 -> use eps (masked by has_condition anyway). A clean
      template fold feeds noise_scale=0. af3_noise_scale_to_noise_level(t)=(log(t/16)+1.2)/1.5.

  GOOD FOR BINDERS: distogram-conditioning = feed the target's internal distances as a
  flexible hint (define the target without pinning exact coordinates).
  """

  # RF3 DEFAULT_DISTOGRAM_BINS: 63 boundaries -> 64 bins.
  @staticmethod
  def _rosettafold3_dist_boundaries():
    b1 = jnp.arange(1.0, 4.0, 0.1)        # 1.0 .. 3.9  (30)
    b2 = jnp.arange(4.0, 20.5, 0.5)       # 4.0 .. 20.0 (33)
    return jnp.concatenate([b1, b2])      # (63,)

  @staticmethod
  def _af3_noise_scale_to_noise_level(ns):
    return (jnp.log(ns / 16.0) + 1.2) / 1.5

  def _features(self, aatype, atom_positions, atom_mask, asym_mask_2d):
    N = aatype.shape[0]
    # token center = CA (protein), via RESTYPE_RIGIDGROUP group 0 order [C, CA, N]
    grp = jnp.take(protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX,
                   aatype, axis=0).astype(jnp.int32)
    bb = grp[:, 0]                                                    # [C, CA, N]
    take = jax.vmap(lambda x, i: x[i])
    ca = take(atom_positions, bb[:, 1])                               # (N,3)
    ca_mask = take(atom_mask, bb[:, 1]).astype(jnp.float32)          # (N,)
    # pairwise CA-CA distance
    d = jnp.sqrt(jnp.sum((ca[:, None] - ca[None]) ** 2, -1) + 1e-10)  # (N,N)
    d = jnp.nan_to_num(d, nan=1e9)                                    # NaN -> last bin
    bnd = self._rosettafold3_dist_boundaries()                                # (63,)
    idx = (d[..., None] > bnd).sum(-1)                               # (N,N) in 0..63
    disto = jax.nn.one_hot(idx, 64)                                  # (N,N,64)
    # has_condition: covered pairs, intra-chain
    has_cond = (ca_mask[:, None] * ca_mask[None]) * asym_mask_2d      # (N,N)
    # joint_noise_level: exact template -> noise_scale ~ 0 (use eps); scalar broadcast
    nl = self._af3_noise_scale_to_noise_level(jnp.asarray(1e-4, disto.dtype))
    nl_ch = jnp.broadcast_to(nl, (N, N))                            # (N,N)
    feat = jnp.concatenate(
        [disto, has_cond[..., None], nl_ch[..., None]], -1)          # (N,N,66)
    return feat * has_cond[..., None]

  def __call__(self, query_embedding, templates, padding_mask_2d,
               multichain_mask_2d, key, use_dropout=False):
    """RF3's template forward -- NOT Boltz's, despite sharing every weight scope.

    RF3 (RF3TemplateEmbedder.forward) has no per-template loop and no template gating:
    it ALWAYS runs one pass, and the pass is a function of Z as well as of the template
    feature. With no template the 66-d feature is zeroed by has_condition, but
      v = emb_pair(norm_pair_before_pairformer(Z))        <- nonzero, Z-dependent
      v = pairformer(v)                                   <- no outer residual (Boltz has one)
      u = norm_after_pairformer(v)                        <- LayerNorm: nonzero even from small v
      out = agg_emb(relu(u))
    still contributes to Z on EVERY recycle. Boltz's forward instead averages over the
    PRESENT templates, so with zero templates it returns exactly 0 -- dropping a real term
    from the RF3 trunk and leaving the pair representation (and the distogram) wrong.

    RF3 conditions on ONE distogram_condition, so multiple templates are collapsed to the
    mean of the present ones (identical to the single-template case, zeros when none).
    """
    c = self.config
    gc = self.global_config
    dtype = query_embedding.dtype
    z = query_embedding
    aatype = templates.aatype
    atom_positions = templates.atom_positions.astype(jnp.float32)
    atom_mask = templates.atom_mask
    T = aatype.shape[0]

    feats = hk.vmap(
        lambda aa, pos, am: self._features(aa, pos, am, multichain_mask_2d),
        in_axes=0, out_axes=0, split_rng=False)(aatype, atom_positions, atom_mask)
    present = (atom_mask.reshape(T, -1).sum(-1) > 0).astype(feats.dtype)   # (T,)
    a_tij = (jnp.einsum('t,tijc->ijc', present, feats)
             / jnp.clip(present.sum(), 1.0, None)).astype(dtype)

    v = hm.Linear(c.num_channels, use_bias=False, name='z_proj')(
        hm.LayerNorm(name='z_norm', use_fast_variance=False)(z))
    v = v + hm.Linear(c.num_channels, use_bias=False, name='a_proj')(a_tij)

    def block(x):
      return modules.PairFormerIteration(
          c.template_stack, gc, name='tmpl_pairformer')(
              act=x, pair_mask=padding_mask_2d, use_dropout=use_dropout)

    # PairFormerIteration returns the UPDATED act (its residuals are internal), so this is
    # RF3's `for block in self.pairformer: _, v_II = block(None, v_II)` -- no `v + stack(v)`.
    v = hk.experimental.layer_stack(c.template_stack.num_layer)(block)(v)
    u = hm.LayerNorm(name='v_norm', use_fast_variance=False)(v)
    out = hm.Linear(z.shape[-1], use_bias=False, name='u_proj')(jax.nn.relu(u))
    return out.astype(dtype)
