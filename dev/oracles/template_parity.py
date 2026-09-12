"""Gates the TEMPLATE EMBEDDER -- 9 models carry one and nothing had gated any.

Templates demonstrably WORK end to end (boltz2 folds 5CAJ to 0.72 A with one,
rosettafold3 to 1.56 A), but that is evidence from folds. No vendor-vs-ours
comparison of the module existed, which also bounds an earlier claim: zeroing
our template contribution on 6MRR changed nothing, and that shows the path is
inert when NO template is supplied, not that the embedder is right when one is.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:/home/ubuntu/protenix \
    python dev/oracles/template_parity.py protenix2

THE SEAM. Native protenix's `TemplateEmbedder` takes the template features
PRECOMPUTED in a dict (`template_distogram` 39, `template_backbone_frame_mask`
1, `template_unit_vector` 3, `template_pseudo_beta_mask` 1, and restype_i/j 32
each = 108 channels); ours derives all of them inside
`SingleTemplateEmbedding.construct_input` from a `Templates` object. So this
harness derives them with OUR OWN library functions (`scoring.pseudo_beta_fn`,
`dgram_from_positions`, `make_backbone_rigid`) and hands them to native, which
means the two sides run on identical features and any gap is in the projections
or the pairformer stack -- not in the feature construction. Gating the feature
construction itself needs native's featuriser on a templated input and is a
separate job; it is NOT covered here, and this file should not be read as
covering it.

The template is the target's own structure (a self-template). For a module gate
the features only need to be realistic and non-degenerate; making it a
biologically interesting template would test nothing extra here.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confidence_parity import _cmp                         # noqa: E402


def _self_template(cif, chain):
  """-> (sequence, folding_input.Template) using a structure as its own template."""
  import modality_check as M
  from alphafold3.common import folding_input

  case = dict(cif=cif, kind='ptm', chain=chain, ptms=[])
  ref_seq, _ = M.reference(case)
  seq = ''.join(c for _, c in ref_seq)
  # query index -> template index, identity: the template IS the query here.
  q2t = {i: i for i in range(len(seq))}
  return seq, folding_input.Template(mmcif=open(cif).read(),
                                     query_to_template_map=q2t)


def our_features(model, cfg, single, asym_mask_2d):  # noqa: C901
  """The 108 template channels, from the MODEL'S OWN feature builder.

  Not a re-derivation: this runs `Protenix2TemplateEmbedding._features`, the
  same code the graph runs, so a difference found downstream cannot be an
  artefact of the harness rebuilding the features differently. The documented
  column order is
  [dgram(39), pb_mask(1), restype_i(32), restype_j(32), uvec(3), bb_mask(1)],
  and it is split back out here into the names native takes.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model.network import template_modules as T

  # of3 (and openbind0) ride AF3's TemplateEmbedding, which has no `_features`
  # -- it builds them inline in SingleTemplateEmbedding.construct_input -- so
  # this branch returns before the shared path below touches that method.
  if model in ('openfold3', 'openbind0', 'intellifold2', 'opendde'):
    dgram, pb, bb, uv = _af3_template_features(cfg, single, asym_mask_2d)
    return dict(aatype=np.asarray(single.aatype), dgram=dgram, pb_mask=pb,
                bb_mask=bb, uvec=uv)

  cls = {'rosettafold3': T.RoseTTAFold3TemplateEmbedding,
         'boltz2': T.Boltz2TemplateEmbedding,
         'openfold3': T.TemplateEmbedding, 'openbind0': T.TemplateEmbedding,
         'intellifold2': T.TemplateEmbedding,
         'opendde': T.TemplateEmbedding}.get(
             model, T.Protenix2TemplateEmbedding)

  def fwd():
    mod = cls(cfg, None)
    return mod._features(jnp.asarray(single.aatype),
                         jnp.asarray(single.atom_positions),
                         jnp.asarray(single.atom_mask),
                         jnp.asarray(asym_mask_2d))

  f = hk.transform(fwd)
  a = np.asarray(f.apply(f.init(jax.random.PRNGKey(0)), jax.random.PRNGKey(0)))
  if model == 'boltz2':
    # boltz builds its own 109 channels, so hand it the RAW geometry instead.
    import jax
    import jax.numpy as jnp
    from alphafold3.model import protein_data_processing as pdp
    from alphafold3.model.scoring import scoring

    def geo():
      cb, cb_mask = scoring.pseudo_beta_fn(
          jnp.asarray(single.aatype), jnp.asarray(single.atom_positions),
          jnp.asarray(single.atom_mask), use_jax=True)
      return cb, cb_mask

    g = hk.transform(geo)
    cb, cb_mask = g.apply(g.init(jax.random.PRNGKey(0)), jax.random.PRNGKey(0))
    grp = np.asarray(pdp.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX)[
        np.asarray(single.aatype)].astype(int)
    return dict(aatype=np.asarray(single.aatype),
                atom_positions=np.asarray(single.atom_positions),
                atom_mask=np.asarray(single.atom_mask),
                group0=grp[:, 0], cb=np.asarray(cb),
                cb_mask=np.asarray(cb_mask, np.float32))
  if a.shape[-1] == 66:
    # rf3: [distogram_condition(64), has_condition(1), joint_noise_level(1)],
    # already masked by has_condition. Native rebuilds the noise channel itself
    # from a PER-TOKEN scale as f(sqrt(ns_i^2 + ns_j^2)), where ours uses a
    # scalar eps as the JOINT level directly -- so ns = eps/sqrt(2) is what
    # makes native's joint equal ours. Getting that wrong shifts one of 66
    # channels and would read as a port difference.
    eps = 1e-4
    return dict(distogram_condition=a[..., :64],
                has_distogram_condition=a[..., 64],
                noise_scale=np.full(a.shape[0], eps / np.sqrt(2.0), np.float32))
  assert a.shape[-1] == 108, 'expected 108 or 66 template channels, got %d' % a.shape[-1]
  o = 0
  out = {}
  for name, w in (('template_distogram', 39), ('template_pseudo_beta_mask', 1),
                  ('restype_i', 32), ('restype_j', 32),
                  ('template_unit_vector', 3),
                  ('template_backbone_frame_mask', 1)):
    out[name] = a[..., o:o + w] if w > 1 else a[..., o]
    o += w
  assert o == 108
  return out


def native_protenix(model, feats, z, pair_mask, hidden_scale_up):
  import torch

  from diffusion_parity import _stub_layer_norm
  _stub_layer_norm()
  from protenix.model.modules.pairformer import TemplateEmbedder

  from denoise_parity import _PROTENIX_CKPT
  ckpt = os.path.expanduser('~/protenix_weights/' + _PROTENIX_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.template_embedder.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))
  c_z = sub['layernorm_z.weight'].shape[0]
  c = sub['linear_no_bias_z.weight'].shape[0]
  bpre = 'pairformer_stack.blocks.'
  n_blocks = 1 + max(int(k[len(bpre):].split('.')[0]) for k in sub
                     if k.startswith(bpre))
  print('  checkpoint: c_z %d, c %d, %d blocks, hidden_scale_up %s'
        % (c_z, c, n_blocks, hidden_scale_up))
  net = TemplateEmbedder(n_blocks=n_blocks, c=c, c_z=c_z,
                         hidden_scale_up=hidden_scale_up)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
  # UNBATCHED z (native reads `num_residues = z.shape[0]` and indexes asym_id
  # as [:, None]), and ONE leading TEMPLATE dim on each feature -- native loops
  # `for template_id in range(template_aatype.shape[0])` and indexes each
  # feature with it. It also builds restype_i/j itself by one-hotting
  # template_aatype, so those two are not passed.
  ifd = {
      'asym_id': torch.zeros(z.shape[0], dtype=torch.long),
      'template_aatype': torch.tensor(
          np.asarray(feats['restype_i']).max(axis=0).argmax(-1),
          dtype=torch.long)[None],
      'template_distogram': t(feats['template_distogram'])[None],
      'template_backbone_frame_mask':
          t(feats['template_backbone_frame_mask'])[None],
      'template_unit_vector': t(feats['template_unit_vector'])[None],
      'template_pseudo_beta_mask':
          t(feats['template_pseudo_beta_mask'])[None],
  }
  with torch.no_grad():
    out = net(ifd, t(z), pair_mask=t(pair_mask),
              triangle_attention='torch', triangle_multiplicative='torch')
  return np.asarray(out).reshape(z.shape[0], z.shape[1], -1)


def native_rf3(model, feats, z, pair_mask, hidden_scale_up):
  """-> the template embedding from RoseTTAFold3's own RF3TemplateEmbedder.

  rf3's template is distance CONDITIONING, not a structural template in the
  AF3 sense: 66 channels = [distogram_condition(64), has_condition(1),
  joint_noise_level(1)], all of them i/j-SYMMETRIC. So the restype column-order
  bug that bit protenix cannot apply here -- there are no restype blocks. What
  this gate is actually here to settle is the OUTER RESIDUAL: rf3 runs
  `for block in self.pairformer: _, v_II = block(None, v_II)` with no outer
  term, where our shared forward inherited boltz2's `v = v + stack(v)`.
  """
  import torch

  from rf3.model.layers.pairformer_layers import RF3TemplateEmbedder

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  raw = raw.get('state_dict', raw.get('model', raw))
  pre = 'shadow.recycler.template_embedder.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  c, c_z = sub['emb_pair.weight'].shape
  raw_dim = sub['emb_templ.weight'].shape[1]
  n_block = 1 + max(int(k.split('.')[1]) for k in sub
                    if k.startswith('pairformer.'))
  print('  checkpoint: c %d, c_z %d, raw_template_dim %d, %d blocks'
        % (c, c_z, raw_dim, n_block))
  net = RF3TemplateEmbedder(n_block=n_block, raw_template_dim=raw_dim,
                            c_z=c_z, c=c, p_drop=0.0)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  for m in net.modules():
    if getattr(m, 'force_bfloat16', False):
      m.force_bfloat16 = False
  net.eval()
  t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
  # rf3 takes the RAW conditioning inputs and builds the 66 channels itself,
  # so ours are split back into its three names.
  f = {
      'distogram_condition': t(feats['distogram_condition']),
      'has_distogram_condition': t(feats['has_distogram_condition']),
      'distogram_condition_noise_scale': t(feats['noise_scale']),
  }
  with torch.no_grad():
    out = net(f, t(z))
  return np.asarray(out).reshape(z.shape[0], z.shape[1], -1)


def native_boltz2(model, feats, z, pair_mask, hidden_scale_up):
  """-> the template embedding from Boltz-2's own TemplateModule.

  A STRONGER gate than the protenix one. boltz takes the RAW geometry --
  `template_frame_rot`, `template_frame_t`, `template_ca`, `template_cb` and
  their masks -- and builds its 109 channels itself, so this compares our
  feature DERIVATION as well as the forward. (The protenix adapter feeds native
  our own 108-d features, so it tests only the forward.) The frames come from
  boltz's own `compute_frame`, not a re-derivation here.

  boltz2 is also the one model that legitimately has the OUTER RESIDUAL around
  the template pairformer -- `v = v + self.pairformer(v, ...)`, which is why it
  is the sole member of `model_config.TEMPLATE_STACK_OUTER_RESIDUAL`. Nothing
  had ever confirmed that reading; this gate does.
  """
  import torch

  from boltz.model.modules.trunkv2 import TemplateModule

  def compute_frame(n, ca, c):
    """Transcribed VERBATIM from boltz/data/tokenize/boltz2.py::compute_frame.

    Not imported: that module pulls `boltz.data.types`, which needs mashumaro,
    which is not in this venv and must not be installed into it. The model
    module itself imports fine. Kept literal, including the 1e-10 guards and the
    column_stack order, because a transcribed frame that is subtly wrong mirrors
    the rotation and is invisible in a distogram.
    """
    v1 = c - ca
    v2 = n - ca
    e1 = v1 / (np.linalg.norm(v1) + 1e-10)
    u2 = v2 - e1 * np.dot(e1.T, v2)
    e2 = u2 / (np.linalg.norm(u2) + 1e-10)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3]), ca

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd)
  pre = 'template_module.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  c, token_z = sub['z_proj.weight'].shape
  raw_dim = sub['a_proj.weight'].shape[1]
  bl = [k for k in sub if k.startswith('pairformer.layers.')]
  n_blocks = 1 + max(int(k.split('.')[2]) for k in bl)
  print('  checkpoint: template_dim %d, token_z %d, a_proj %d-d, %d blocks'
        % (c, token_z, raw_dim, n_blocks))
  net = TemplateModule(token_z=token_z, template_dim=c,
                       template_blocks=n_blocks, dropout=0.0)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()

  aa = np.asarray(feats['aatype'])
  pos = np.asarray(feats['atom_positions'])
  msk = np.asarray(feats['atom_mask'])
  N = aa.shape[0]
  # RESTYPE_RIGIDGROUP group 0 is ordered [C, CA, N] -- not [N, CA, C]. Getting
  # that backwards mirrors the frame and is invisible in a distogram.
  grp = np.asarray(feats['group0'])
  cvec, ca, nvec = pos[np.arange(N), grp[:, 0]], pos[np.arange(N), grp[:, 1]], \
      pos[np.arange(N), grp[:, 2]]
  ca_mask = msk[np.arange(N), grp[:, 1]].astype(np.float32)
  rot = np.zeros((N, 3, 3), np.float32)
  for i in range(N):
    r, _ = compute_frame(nvec[i], ca[i], cvec[i])
    rot[i] = r
  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  # boltz's restype vocabulary is AF3's + 2 (see Boltz2TemplateEmbedding).
  restype = np.eye(raw_dim - 76, dtype=np.float32)[np.clip(aa + 2, 0, None)]
  fd = {
      'asym_id': torch.zeros(1, N, dtype=torch.long),
      'template_restype': t(restype)[None][None],
      'template_frame_rot': t(rot)[None][None],
      'template_frame_t': t(ca)[None][None],
      'template_mask_frame': t(ca_mask)[None][None],
      'template_cb': t(feats['cb'])[None][None],
      'template_ca': t(ca)[None][None],
      'template_mask_cb': t(feats['cb_mask'])[None][None],
      # DERIVED, not assumed. Boltz masks its per-template output with
      # `feats["template_mask"].any(dim=2)` before averaging, so an all-ones mask
      # tells it an empty template is PRESENT -- which made the EMPTY gate read
      # rms 2.18 for native against our 0, and that was the harness, not a port
      # difference. With a real template this is all-ones anyway.
      'template_mask': t((msk.sum(-1) > 0).astype(np.float32))[None][None],
  }
  with torch.no_grad():
    out = net(t(z)[None], fd, t(pair_mask)[None])
  return np.asarray(out).reshape(z.shape[0], z.shape[1], -1)


def _af3_template_features(cfg, single, asym_mask_2d):
  """The per-feature pieces AF3's own SingleTemplateEmbedding builds.

  Derived with OUR library functions (`scoring.pseudo_beta_fn`,
  `dgram_from_positions`, `make_backbone_rigid`) -- the same calls the module
  makes -- and handed to of3, which takes them precomputed. of3 wants the
  1-D per-token masks and builds its own pair masks, where our module builds the
  pair masks internally; that is fine as long as the 1-D masks agree, which is
  what this feeds.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.jax import geometry
  from alphafold3.model import protein_data_processing as pdp
  from alphafold3.model.network import template_modules as T
  from alphafold3.model.scoring import scoring

  def fwd():
    aatype = jnp.asarray(single.aatype)
    pos = jnp.asarray(single.atom_positions) * jnp.asarray(
        single.atom_mask)[..., None]
    msk = jnp.asarray(single.atom_mask)
    pb, pb_mask = scoring.pseudo_beta_fn(aatype, pos, msk)
    dgram = T.dgram_from_positions(pb, cfg.dgram_features)
    grp = jnp.take(pdp.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX, aatype, axis=0)
    rigid, bb_mask = T.make_backbone_rigid(
        geometry.Vec3Array.from_array(pos), msk, grp.astype(jnp.int32))
    uv = rigid[:, None].inverse().apply_to_point(rigid.translation).normalized()
    return dgram, pb_mask, bb_mask, jnp.stack([uv.x, uv.y, uv.z], -1)

  f = hk.transform(fwd)
  dgram, pb_mask, bb_mask, uv = f.apply(f.init(jax.random.PRNGKey(0)),
                                        jax.random.PRNGKey(0))
  return (np.asarray(dgram, np.float32), np.asarray(pb_mask, np.float32),
          np.asarray(bb_mask, np.float32), np.asarray(uv, np.float32))


def native_of3(model, feats, z, pair_mask, hidden_scale_up):
  """-> the template embedding from OpenFold3's own TemplateEmbedderAllAtom.

  of3 is on AF3's template design (one Linear PER FEATURE, not one fused
  a_proj), which is where protenix's restype bug would have lived if the
  converter had mapped positionally: of3's `aatype_linear_1` takes the
  **i**-varying block and `aatype_linear_2` the j-varying, while AF3's
  `to_concat` puts the **j**-varying block first. `converters/openfold3.py`
  already crosses them -- `[(3, 'aatype_linear_1'), (2, 'aatype_linear_2')]` --
  and this gate is what confirms it.
  """
  import copy

  import torch

  from openfold3.core.model.latent.template_module import (
      TemplateEmbedderAllAtom)
  from openfold3.projects.of3_all_atom.config.model_config import model_config

  from denoise_parity import _OF3_CKPT
  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'template_embedder.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)

  def _find(cfg):
    """The template subtree, located rather than assumed.

    of3 calls it `architecture/template`, not `template_embedder` (which is the
    CHECKPOINT prefix); searching for the latter finds nothing.
    """
    if hasattr(cfg, 'keys'):
      if 'template_pair_stack' in cfg and 'template_pair_embedder' in cfg:
        return cfg
      for k in cfg:
        got = _find(cfg[k]) if hasattr(cfg[k], 'keys') else None
        if got is not None:
          return got
    return None

  tcfg = _find(copy.deepcopy(model_config))
  if tcfg is None:
    raise SystemExit('no template subtree in of3 model_config')
  net = TemplateEmbedderAllAtom(tcfg)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()

  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  # of3's restype is its own 32-class order, the same permutation the converter
  # and every other of3 gate use.
  from converters.openfold3 import _AF3_TO_OF3_AATYPE as remap
  aa = np.asarray(feats['aatype'])
  restype = np.eye(32, dtype=np.float32)[np.asarray(remap)[aa]]
  batch = {
      'asym_id': torch.zeros(z.shape[0], dtype=torch.long)[None],
      'template_restype': t(restype)[None][None],
      'template_distogram': t(feats['dgram'])[None][None],
      'template_pseudo_beta_mask': t(feats['pb_mask'])[None][None],
      'template_backbone_frame_mask': t(feats['bb_mask'])[None][None],
      'template_unit_vector': t(feats['uvec'])[None][None],
  }
  with torch.no_grad():
    out = net(batch, t(z)[None], t(pair_mask)[None])
  return np.asarray(out).reshape(z.shape[0], z.shape[1], -1)


def native_if2(model, feats, z, pair_mask, hidden_scale_up):
  """-> the template embedding from IntelliFold-2's own TemplateEmbedder.

  Also on AF3's per-feature-Linear design, and it names the two restype
  projections explicitly: `linear_aatype_col` takes `unsqueeze(-3)` (the
  **j**-varying block) and `linear_aatype_row` `unsqueeze(-2)` (i-varying),
  which is the same asymmetry protenix got wrong. `converters/intellifold2.py`
  maps col->slot 2 and row->slot 3, matching AF3's `to_concat` order; this
  measures that.

  if2 takes the 1-D per-token masks and squares them itself, like of3, and its
  restype block is **31** classes where of3/protenix carry 32.
  """
  import torch

  from intellifold.openfold.model.embedders import TemplateEmbedder

  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  cands = sorted({k.rsplit('.', 1)[0] for k in sd
                  if k.endswith('linear_aatype_col.weight')})
  if not cands:
    raise SystemExit('no linear_aatype_col in the checkpoint')
  pre = cands[0][:-len('linear_aatype_col')]
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  c_t, c_z = sub['linear_z.weight'].shape
  n_bins = sub['linear_d.weight'].shape[1]
  n_aa = sub['linear_aatype_col.weight'].shape[1]
  bl = [k for k in sub if k.startswith('pairformer_stack.')]
  n_blocks = 1 + max(int(k.split('.')[1]) for k in bl) if bl else 2
  # if2 is the widened "full_fat" tree, so the template stack is NOT AF3's
  # 64/16/4: read every width off the checkpoint. tri_att `linear.weight` is
  # (heads, c_t) and `mha.linear_q` is (heads*c_hidden, c_t); tri_mul's
  # `linear_ab_p` is (2*c_hidden_mul, c_t).
  heads = sub['pairformer_stack.0.tri_att_start.linear.weight'].shape[0]
  c_att = sub['pairformer_stack.0.tri_att_start.mha.linear_q.weight'
              ].shape[0] // heads
  c_mul = sub['pairformer_stack.0.tri_mul_out.linear_ab_p.weight'].shape[0] // 2
  print('  prefix %r: c_t %d, c_z %d, %d bins, %d aatype classes, %d blocks, '
        '%d heads, c_hidden att %d / mul %d'
        % (pre, c_t, c_z, n_bins, n_aa, n_blocks, heads, c_att, c_mul))
  # Construction args are if2's own (config.py `template_embedder`): note c_a
  # is the CONCATENATED feature width 39+1+3+1+31+31 = 106, not the restype
  # class count, and its restype blocks are 31 classes where of3/protenix use
  # 32. The widths still come off the checkpoint, so a release that changed one
  # fails in load_state_dict rather than comparing quietly.
  net = TemplateEmbedder(c_z=c_z, c_t=c_t, c_a=n_bins + 1 + 3 + 1 + 2 * n_aa,
                         no_blocks=n_blocks, c_hidden_mul=c_mul,
                         c_hidden_pair_att=c_att, no_bins=n_bins,
                         no_heads_pair=heads, transition_n=2, pair_dropout=0.0,
                         inf=1e9, eps=1e-6)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  aa = np.asarray(feats['aatype'])
  restype = np.eye(n_aa, dtype=np.float32)[np.clip(aa, 0, n_aa - 1)]
  batch = {
      'asym_id': torch.zeros(z.shape[0], dtype=torch.long)[None],
      'template_aatype': t(restype)[None][None],
      'template_distogram': t(feats['dgram'])[None][None],
      'template_pseudo_beta_mask': t(feats['pb_mask'])[None][None],
      'template_backbone_frame_mask': t(feats['bb_mask'])[None][None],
      'template_unit_vector': t(feats['uvec'])[None][None],
  }
  with torch.no_grad():
    out = net(batch, t(z)[None], t(pair_mask)[None], chunk_size=None)
  out = out[0] if isinstance(out, (tuple, list)) else out
  return np.asarray(out).reshape(z.shape[0], z.shape[1], -1)


def native_opendde(model, feats, z, pair_mask, hidden_scale_up):
  """-> the template embedding from OpenDDE's own TemplateEmbedder.

  opendde is protenix-lineage, so its template embedder takes ONE fused 108-d
  `linear_no_bias_a` -- while OUR side runs AF3's per-feature TemplateEmbedding.
  `converters/opendde.py` therefore SPLITS that fused weight into AF3's nine
  slots, and the split encodes the very convention the protenix fix established
  today: cols 40:72 are the **j**-varying restype block (-> AF3 slot 2) and
  72:104 the i-varying (-> slot 3). Notably the opendde converter had this right
  while `Protenix2TemplateEmbedding` had it backwards, so the two disagreed and
  the converter was correct.

  The fused feature vector is assembled here in opendde's own order, with the
  restype one-hots in opendde's 32-class vocabulary (RESTYPE_PERM maps ours ->
  theirs; our 31 classes are a subset of their 32).
  """
  import torch

  from opendde.model.modules.pairformer import TemplateEmbedder

  from converters.opendde import RESTYPE_PERM

  ckpt = os.path.expanduser('~/opendde_weights/opendde.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'module.template_embedder.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  c_z = sub['layernorm_z.weight'].shape[0]
  c = sub['linear_no_bias_z.weight'].shape[0]
  n_feat = sub['linear_no_bias_a.weight'].shape[1]
  bl = [k for k in sub if k.startswith('pairformer_stack.blocks.')]
  n_blocks = 1 + max(int(k.split('.')[2]) for k in bl) if bl else 2
  print('  checkpoint: c_z %d, c %d, %d fused features, %d blocks'
        % (c_z, c, n_feat, n_blocks))
  net = TemplateEmbedder(n_blocks=n_blocks, c=c, c_z=c_z,
                         hidden_scale_up=c_z > 128)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()

  # opendde's forward takes the SAME named keys protenix's does and builds the
  # concat itself, so nothing is assembled here beyond squaring the 1-D masks
  # and putting the restype indices into opendde's own 32-class vocabulary
  # (RESTYPE_PERM maps ours -> theirs).
  aa = np.asarray(feats['aatype'])
  perm = np.asarray(RESTYPE_PERM)
  aat = perm[np.clip(aa, 0, len(perm) - 1)]
  pb2 = feats['pb_mask'][:, None] * feats['pb_mask'][None, :]
  bb2 = feats['bb_mask'][:, None] * feats['bb_mask'][None, :]
  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  ifd = {
      'asym_id': torch.zeros(z.shape[0], dtype=torch.long),
      'template_aatype': t(aat, torch.long)[None],
      'template_distogram': t(feats['dgram'])[None],
      'template_pseudo_beta_mask': t(pb2)[None],
      'template_backbone_frame_mask': t(bb2)[None],
      'template_unit_vector': t(feats['uvec'])[None],
  }
  with torch.no_grad():
    out = net(ifd, t(z), pair_mask=t(pair_mask),
              triangle_attention='torch', triangle_multiplicative='torch')
  return np.asarray(out).reshape(z.shape[0], z.shape[1], -1)


NATIVES = {}
try:
  from denoise_parity import _PROTENIX_CKPT
  NATIVES.update({m: native_protenix for m in _PROTENIX_CKPT})
  NATIVES['rosettafold3'] = native_rf3
  NATIVES['boltz2'] = native_boltz2
  from denoise_parity import _OF3_CKPT as _OF3
  NATIVES.update({m: native_of3 for m in _OF3})
  NATIVES['intellifold2'] = native_if2
  NATIVES['opendde'] = native_opendde
except ImportError:
  pass


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--cif', default=os.path.expanduser('~/5K9P.cif'),
                  help='a SINGLE-chain mmCIF; folding_input.Template rejects '
                       'more than one (5CAJ has 2, which is what caught this)')
  ap.add_argument('--chain', default='A')
  ap.add_argument('--model_dir', default=None)
  args = ap.parse_args(argv)
  sys.argv = sys.argv[:1]

  if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
    raise SystemExit('set JAX_DEFAULT_MATMUL_PRECISION=highest')
  # OURS_ONLY=1 runs our side alone, with no native comparison. It exists for
  # chai1, the one model whose template embedder cannot be invoked from the
  # shipped artifacts, and it is only useful with ZERO=<scope>/<leaf> -- an
  # output with nothing to compare against is not a gate.
  if args.model not in NATIVES and not os.environ.get('OURS_ONLY'):
    raise SystemExit('no native adapter for %r (OURS_ONLY=1 with '
                     'ZERO=<scope>/<leaf> sizes one parameter instead)'
                     % args.model)

  import fold_check
  from alphafold3.model import feat_batch

  seq, tmpl = _self_template(args.cif, args.chain)
  print('%s template embedder: %d residues, self-template from %s'
        % (args.model, len(seq), os.path.basename(args.cif)))
  batch, cfg, model_dir = fold_check._fold_setup(
      args.model, seq, args.model_dir, templates=[tmpl])
  fb = feat_batch.Batch.from_data_dict(batch)
  templates = fb.templates
  n_tmpl = np.asarray(templates.aatype).shape[0]
  n_tok = np.asarray(fb.token_features.mask).shape[0]
  covered = int((np.asarray(templates.atom_mask).sum(-1) > 0).sum())
  print('  %d template(s), %d tokens, %d covered residues'
        % (n_tmpl, n_tok, covered))
  # EMPTY=1: the NO-TEMPLATE case, which is the one every fold in this repo
  # actually runs and the one nothing gated. It is not an all-zero comparison:
  # protenix, opendde and intellifold2 all keep a Z-DEPENDENT half
  # (`linear_z(layer_norm_z(z))`, summed over the padded slots and divided by
  # the SLOT count), and rf3 runs its single pass unconditionally, so the term
  # is live with no template at all. Dropping it cost protenix1 9 A on
  # ubiquitin. What differs per vendor is the empty slot's RESTYPE -- protenix
  # fills the first slot with GAP, opendde fills all four, intellifold2
  # deliberately uses 0 -- and the featuriser knob `empty_template_gap` is what
  # this checks.
  empty = bool(os.environ.get('EMPTY'))
  if empty:
    # Take the TRUE empty slots -- re-featurise with no template at all, so the
    # slot carries whatever the vendor's convention puts there (protenix and
    # opendde: the GAP restype; intellifold2, rf3 and boltz2: zeros). Zeroing a
    # self-template's coordinates instead would leave the QUERY's restypes in
    # slot 0 and test nothing about the convention.
    e_batch, _, _ = fold_check._fold_setup(args.model, seq, args.model_dir)
    templates = feat_batch.Batch.from_data_dict(e_batch).templates
    n_tmpl = np.asarray(templates.aatype).shape[0]
    covered = 0
    print('  EMPTY: no template supplied; %d slots, aatype per slot %s, any '
          'atom mask %s'
          % (n_tmpl, [np.unique(np.asarray(templates.aatype)[t]).tolist()
                      for t in range(n_tmpl)],
             bool(np.asarray(templates.atom_mask).any())))
  if not covered and not empty:
    raise SystemExit('the template covers NOTHING -- the gate would compare '
                     'two all-zero paths and pass meaninglessly')

  c_z = cfg.evoformer.pair_channel
  rng = np.random.default_rng(0)
  z = (rng.normal(size=(n_tok, n_tok, c_z)) * 0.5).astype(np.float32)
  pair_mask = np.ones((n_tok, n_tok), np.float32)
  multichain = np.ones((n_tok, n_tok), np.float32)

  # ONE template on BOTH sides. The batch pads to 4 template slots; our module
  # aggregates over all of them while native loops over exactly the ones it is
  # given, so feeding native 1 and letting ours aggregate 4 is not a comparison
  # (it read corr 0.9985 that way -- the empty slots, not a port difference).
  # .copy() is load-bearing: construct_input does `dense_atom_positions *=
  # dense_atom_mask[..., None]` in place, which numpy refuses on a read-only
  # view ("output array is read-only"). Harmless under jax tracing, fatal here.
  # ONE slot on both sides, in EMPTY mode too: the native adapters take a single
  # template's features, and feeding ours four while native sees one compares
  # 4/4 against 1/1 -- which is a comparison of the DIVISOR, not of the term. The
  # divisor is checked separately (EMPTY_SLOTS below), where it is exact by
  # construction whenever the slots are identical.
  keep = 1
  one = type(templates)(
      aatype=np.array(templates.aatype)[:keep].copy(),
      atom_positions=np.array(templates.atom_positions)[:keep].copy(),
      atom_mask=np.array(templates.atom_mask)[:keep].copy())
  single = type(templates)(
      aatype=np.asarray(templates.aatype)[0],
      atom_positions=np.asarray(templates.atom_positions)[0],
      atom_mask=np.asarray(templates.atom_mask)[0])
  feats = our_features(args.model, cfg.evoformer.template, single, multichain)
  print('  our features: ' + ', '.join(
      '%s %s' % (k.replace('template_', ''), v.shape)
      for k, v in sorted(feats.items())))

  # hidden_scale_up is derived from the TRUNK's c_z, but the template stack
  # runs at c=64 -- so this is a guess, and TMPL_HSU=0/1 forces it either way.
  # (load_state_dict would raise on a size mismatch, so a wrong value here is
  # not silent -- but a value that loads is not automatically the right one.)
  hsu = c_z > 128
  if os.environ.get('TMPL_HSU') is not None:
    hsu = bool(int(os.environ['TMPL_HSU']))
  if args.model not in NATIVES:
    ours(args.model, cfg, model_dir, one, z, pair_mask, multichain)
    return 0
  ref = NATIVES[args.model](args.model, feats, z, pair_mask, hsu)
  got = ours(args.model, cfg, model_dir, one, z, pair_mask, multichain)
  if empty:
    # THE DIVISOR. Our module sums over every padded slot and divides by their
    # COUNT, which is what protenix, opendde and intellifold2 all do
    # (`u / (n_templ + eps)`). With identical empty slots that makes the
    # aggregate equal to the single-slot result -- so this comparison IS the
    # divisor check, and it fails loudly if either side divides by the number of
    # PRESENT templates (which would be a division by zero, clipped to 1) or by
    # anything else.
    allslots = type(templates)(
        aatype=np.array(templates.aatype).copy(),
        atom_positions=np.array(templates.atom_positions).copy(),
        atom_mask=np.array(templates.atom_mask).copy())
    agg = ours(args.model, cfg, model_dir, allslots, z, pair_mask, multichain)
    ident = all(np.array_equal(np.asarray(templates.aatype)[0],
                               np.asarray(templates.aatype)[t])
                for t in range(np.asarray(templates.aatype).shape[0]))
    _cmp('divisor: %d slots vs 1%s' % (np.asarray(templates.aatype).shape[0],
                                       '' if ident else ' (slots DIFFER)'),
         np.asarray(agg), np.asarray(got))
    if np.abs(np.asarray(got)).max() < 1e-6 and np.abs(ref).max() < 1e-6:
      print('  BOTH ZERO -- this vendor MASKS its empty template slots before '
            'averaging, so the term really is absent with no template. That is '
            'the convention, not a vacuous pass: protenix, opendde, '
            'intellifold2 and rf3 all keep a Z-dependent half here.')
  print('  shapes: ours %s native %s' % (got.shape, ref.shape))
  _cmp('template_embed', got, ref)
  return 0


def ours(model, cfg, model_dir, templates, z, pair_mask, multichain):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import template_modules

  cfg.global_config.bfloat16 = 'none'
  full = afp.get_model_haiku_params(model_dir=model_dir)

  def fwd():
    # The model's OWN dispatch: protenix does not use AF3's TemplateEmbedding
    # (that was the first mistake here -- it builds AF3-named scopes the blob
    # does not have, 50 unmapped). evoformer.py picks a per-family class.
    cls = {'rosettafold3': template_modules.RoseTTAFold3TemplateEmbedding,
           'boltz2': template_modules.Boltz2TemplateEmbedding,
           'openfold3': template_modules.TemplateEmbedding,
           'openbind0': template_modules.TemplateEmbedding,
           'intellifold2': template_modules.TemplateEmbedding,
           'opendde': template_modules.TemplateEmbedding,
           # chai1's blob uses AF3's own scope names
           # (template_embedding/single_template_embedding + output_linear), so
           # it is AF3's TemplateEmbedding, not protenix's fused one. There is no
           # NATIVE side for chai1 (TorchScript, no callable submodule forward),
           # but `ours` alone is still useful -- see BIAS_EFFECT below.
           'chai1': template_modules.TemplateEmbedding}.get(
               model, template_modules.Protenix2TemplateEmbedding)
    return cls(cfg.evoformer.template, cfg.global_config)(
            query_embedding=jnp.asarray(z), templates=templates,
            padding_mask_2d=jnp.asarray(pair_mask),
            multichain_mask_2d=jnp.asarray(multichain),
            key=jax.random.PRNGKey(0))

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      src = None
      for cand in ('diffuser/evoformer/' + sc, 'diffuser/' + sc, sc):
        if cand in full:
          src = full[cand]
          break
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = np.asarray(v, np.float32)
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:3]))
  assert not unmapped, 'our template embedder is partly at init'
  out = np.asarray(f.apply(params, jax.random.PRNGKey(0)))

  # ZERO=<scope>/<leaf> reruns with one parameter zeroed and reports what it was
  # worth. This is how a dropped weight gets SIZED when no native module can be
  # called: a fold cannot do it (chai1's 5K9P templated mean moved 1.779 vs
  # 1.811 with the bias on/off, and its no-template 6MRR moved 0.017 in the same
  # comparison -- i.e. both inside the process-to-process band), but the module
  # output is deterministic and the difference is exactly the term's worth.
  zero = os.environ.get('ZERO')
  if zero:
    sc, leaf = zero.rsplit('/', 1)
    if sc not in params or leaf not in params[sc]:
      raise SystemExit('no such parameter %r (have e.g. %s)'
                       % (zero, sorted(params.get(sc, {}))[:4]))
    keep = params[sc][leaf]
    params[sc][leaf] = np.zeros_like(keep)
    alt = np.asarray(f.apply(params, jax.random.PRNGKey(0)))
    d = np.abs(out - alt)
    print('  ZERO %s: |param| max %.4f -> output max|d| %.5f  '
          'rms(out) %.4f  relative %.4f  corr %.6f'
          % (zero, np.abs(keep).max(), d.max(),
             float(np.sqrt((out ** 2).mean())),
             d.max() / max(float(np.sqrt((out ** 2).mean())), 1e-9),
             float(np.corrcoef(out.ravel(), alt.ravel())[0, 1])))
  return out


if __name__ == '__main__':
  raise SystemExit(main())
