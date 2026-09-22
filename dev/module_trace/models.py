"""Model builders: name -> (config, params, forward) ready to trace.

Every ported model loads the same way: build the config for its name, init the
haiku graph, and overlay the converted checkpoint on top of the init (the
init+merge path the fold tests use -- an unported head stays at its init value
instead of blowing up, which keeps a partially-ported model traceable).
"""
import os

import numpy as np

CHECKPOINTS = {
    'alphafold3':   '~/af3_official_weights',        # already a haiku blob
    'openfold3':    '~/af3_converted_weights',       # already a haiku blob
    'intellifold2': '~/model_v2',                    # already a haiku blob
    'opendde':      '~/opendde_converted_weights',   # already a haiku blob
    'boltz2':       '~/boltz2_weights/boltz2_conf.ckpt',
    'protenix2':    '~/protenix_weights/protenix-v2.pt',
    'rosettafold3': '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt',
    'chai1':        '~/chai1_weights',               # five TorchScript archives
}

# models whose converted weights are produced in-process from a torch checkpoint
# (rather than read from a .bin.zst blob written by ensure_weights):
#   module, assembler, checkpoint loader (None -> plain torch.load + state_dict)
_FROM_TORCH = {
    'boltz2':       ('boltz2', 'map_boltz2_to_af3', None),
    'protenix2':    ('protenix2', 'map_protenix2_to_af3', 'load_protenix_checkpoint'),
    'rosettafold3': ('rosettafold3', 'map_rosettafold3_to_af3', 'load_rf3_checkpoint'),
    # chai1 used to be read from a pre-converted blob, which was written before the
    # template embedder, the MSA input projection and the profile embedding were
    # ported -- so every trace and every parity row built through models.build ran
    # those 54 params at RANDOM INIT while the fold scripts (which call
    # map_chai1_to_af3 directly) had them. Converting in-process cannot drift from
    # the converter that way.
    'chai1':        ('chai1', 'map_chai1_to_af3', 'load_chai1'),
}


# featurise_spec kwargs a model REQUIRES. OpenDDE diffuses on structural tokens,
# so a plain residue batch runs it on the wrong input and folds to nothing (~106 A)
# rather than failing loudly -- exactly the silent-mismatch class this harness is
# meant to catch, so it is declared here once.
FEATURISE = {
    # boltz2 keeps a modified residue as ONE token holding all its atoms
    # (data/tokenize/boltz2.py: standard -> per residue, NONPOLYMER -> per atom,
    # else -> one token, all atoms). AF3 atomises instead, and handing boltz2 ten
    # single-atom tokens where it wants one ten-atom token inflates the residue
    # ~2.4x. Inert when nothing is modified, and ligands atomise either way.
    'boltz2': dict(modified_as_one_token=True),
    'opendde': dict(opendde=True, struct_num_tokens=160, padded_keys=True),
    'protenix2': dict(padded_keys=True),
    'rosettafold3': dict(chirals=True, atomized_element_names=True,
                         restype_alignment=True,
                         atomized_unknown_restype=True,
                         atomized_backbone_bonds=True),
}


def featurise_kwargs(model):
  return dict(FEATURISE.get(model, {}))


def available(model):
  p = os.path.expanduser(CHECKPOINTS[model])
  if model in _FROM_TORCH:
    return os.path.isfile(p), ('missing ' + p if not os.path.isfile(p) else '')
  import glob
  ok = bool(glob.glob(os.path.join(p, '*.bin.zst')))
  return ok, ('no *.bin.zst in ' + p if not ok else '')


def _load_torch_params(model, path):
  import importlib
  mod_name, map_name, loader = _FROM_TORCH[model]
  mod = importlib.import_module('colabdesign2.af3.converters.' + mod_name)
  if loader is not None:
    sd = getattr(mod, loader)(path)
  else:
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    sd = {k: v.numpy() for k, v in sd.items() if hasattr(v, 'numpy')}
  return getattr(mod, map_name)(sd)


# ------------------------------------------------------------------ shape cache
# jax.eval_shape derives the parameter tree in ~4.7s without executing anything.
# Caching that to disk takes it to ~0.05s. The cache is a SECOND SOURCE OF TRUTH for
# the graph's shapes, so the only safe version is one that cannot go stale silently:
# the key covers the config AND the bytes of every module that builds the graph, so
# any edit to the network -- or any config flag that changes a width -- is a miss.
_SHAPE_CACHE_VERSION = 2   # v2: the key now covers which batch features exist


def _graph_source_digest():
  """sha256 over every source file that can change the parameter tree."""
  import hashlib, glob
  h = hashlib.sha256()
  roots = [os.path.join(os.path.dirname(__file__), '..', '..', 'colabdesign2', 'af3',
                        'alphafold3', 'model')]
  files = []
  for r in roots:
    files += glob.glob(os.path.join(r, '*.py'))
    files += glob.glob(os.path.join(r, 'network', '*.py'))
  for f in sorted(files):
    with open(f, 'rb') as fh:
      h.update(fh.read())
  return h.hexdigest()


_SRC_DIGEST = None


def _batch_signature(fbatch):
  '''Which OPTIONAL batch features are present -- the tree depends on it.

  🔴 The parameter tree is not a function of (model, config) alone. A graph
  branches on whether an optional feature is in the batch at all: chai1 builds
  `chai1_esm_embedding` only when esm_embeddings is not None, rf3 builds its
  chiral module only with chirals, and so on. Keying the cache without this
  meant the FIRST batch to populate a key decided the tree for every later one:
  rna_parity and ligand_parity run chai1 at num_msa=1 with no ESM, so the
  cached table had 308 scopes and no esm entry, and a later ESM run at the same
  num_msa merged its converted weights against that tree, silently DROPPED the
  esm scope, and died at apply with

      Unable to retrieve parameter 'weights' for module 'diffuser/chai1_esm_embedding'

  Presence, not shape: shapes would fragment the cache per input length for no
  benefit, since the widths come from the config.
  '''
  import numpy as _np
  out = []
  for k in sorted(fbatch):
    v = fbatch[k]
    out.append((k, v is None or (hasattr(v, 'size') and _np.size(v) == 0)))
  return repr(out)


# Config fields that are LOOP COUNTS, not shapes. The parameter tree cannot
# depend on them -- recycling reuses one set of weights, and the sampler runs one
# denoiser N times -- so hashing them only makes the cache miss.
#
# It did exactly that. `repr(cfg)` put them in the key, the tables were generated
# at diffusion_steps=1 / num_recycles=0, and every REAL run (200 steps, 10
# recycles) therefore missed and silently re-derived. Measured before the fix:
#
#   (steps=1,   recycles=0)  hit      <- the generator's own config, and the
#   (steps=200, recycles=0)  miss        only one the old test checked
#   (steps=1,   recycles=10) miss
#   (steps=200, recycles=10) miss     <- every real fold
#
# Normalising is only safe because it is CHECKED: test_shape_table_ignores_loop
# _counts derives the tree at both extremes and asserts the leaves are identical,
# so if either field ever becomes shape-relevant the test fails rather than the
# cache quietly serving a wrong tree. Under-specifying this key is worse than
# over-specifying: a miss costs 4.3 s, a wrong hit costs correctness.
_LOOP_COUNT_FIELDS = (
    ('num_recycles',),
    ('heads', 'diffusion', 'eval', 'steps'),
    ('heads', 'diffusion', 'eval', 'num_samples'),
)


def _shape_relevant_repr(cfg) -> str:
  """repr(cfg) with the loop counts pinned, so only shape-affecting fields key it."""
  import copy
  c = copy.deepcopy(cfg)
  for path in _LOOP_COUNT_FIELDS:
    node = c
    for part in path[:-1]:
      node = getattr(node, part, None)
      if node is None:
        break
    if node is not None and hasattr(node, path[-1]):
      setattr(node, path[-1], 0)
  return repr(c)


def _shape_cache_signature(model, cfg, fbatch=None):
  """Identifies (model, config, graph source, which batch features exist).

  Any change makes every cache a miss.

  Must be STABLE ACROSS PROCESSES -- the checked-in tables exist so a fresh Colab VM
  never re-derives, and a signature containing anything per-run (an object id, a
  path, a timestamp) would silently miss every time and look merely slow.
  """
  import hashlib
  global _SRC_DIGEST
  if _SRC_DIGEST is None:
    _SRC_DIGEST = _graph_source_digest()
  h = hashlib.sha256()
  h.update(f'v{_SHAPE_CACHE_VERSION}|{model}|'.encode())
  h.update(_shape_relevant_repr(cfg).encode())
  h.update(_SRC_DIGEST.encode())
  if fbatch is not None:
    h.update(_batch_signature(fbatch).encode())
  return h.hexdigest()


def _shape_cache_path(model, cfg, fbatch=None):
  d = os.path.expanduser(os.environ.get('COLABDESIGN2_CACHE_DIR',
                                        '~/.cache/colabdesign2'))
  d = os.path.join(d, 'param_shapes')
  return os.path.join(d, f'{model}-{_shape_cache_signature(model, cfg, fbatch)[:20]}.json')


def _repo_table(model, cfg, fbatch=None):
  """The checked-in table from gen_param_shapes, if it matches this exact graph.

  Generated, never hand-edited. It is what survives a fresh machine: the user
  cache under ~/.cache goes with the VM, so without a table in the repo the first
  build of each process re-derives the tree (measured 12.6 s with derivation vs
  8.3 s with the table, one fresh process each -- about 4.3 s).

  WHO ACTUALLY BENEFITS, since this used to be described too broadly: everything
  that goes through `build()` -- tests/, tools/oracles/, module_trace. NOT the
  af3-any-model notebook and NOT colabdesign2/ itself, neither of which imports
  this module; run_alphafold.py loads its parameters straight from the blob and
  has no derivation path at all, so its <model>.shapes.json is a gap-reporting
  guard rather than a speed-up.

  The signature check means a table left behind by an older graph is a miss, not
  a wrong parameter tree.

  Lives in `colabdesign2/af3/param_shapes/`, i.e. INSIDE the installed package --
  `tool.setuptools.packages.find` only includes `colabdesign2*`, so a table under
  tools/ ships with the git clone but not with a pip install, which is precisely the
  Colab case it exists for.
  """
  import json
  from colabdesign2 import af3 as _af3_pkg
  path = os.path.join(os.path.dirname(os.path.abspath(_af3_pkg.__file__)),
                      'param_shapes', model + '.json')
  if not os.path.exists(path):
    return None
  try:
    with open(path) as fh:
      t = json.load(fh)
  except Exception:
    return None
  want = _shape_cache_signature(model, cfg, fbatch)
  # A file holds one ENTRY PER CASE. The parameter tree depends on which optional
  # features the batch carries (see _batch_signature), so one table generated from
  # a ligand case cannot serve a protein-only run -- and the miss is silent, costing
  # a re-derivation nobody sees. Several entries make the common cases all hit.
  for entry in t.get('entries', [t]):
    if entry.get('signature') == want:
      return entry['shapes']
  return None


def _param_shapes(model, cfg, init_fn, key, fbatch, use_cache=True):
  """-> {scope: {name: ShapeDtypeStruct}}, from disk when the key matches."""
  import json
  import jax
  if use_cache:
    t = _repo_table(model, cfg, fbatch)
    if t is not None:
      return {sc: {nm: jax.ShapeDtypeStruct(tuple(shape), np.dtype(dt))
                   for nm, (dt, shape) in v.items()} for sc, v in t.items()}
  path = _shape_cache_path(model, cfg, fbatch)
  if use_cache and os.path.exists(path):
    try:
      with open(path) as fh:
        raw = json.load(fh)
      return {sc: {nm: jax.ShapeDtypeStruct(tuple(shape), np.dtype(dt))
                   for nm, (dt, shape) in v.items()} for sc, v in raw.items()}
    except Exception:
      pass                              # unreadable/corrupt -> just re-derive
  shapes = jax.eval_shape(init_fn, key, fbatch)
  if use_cache:
    try:
      os.makedirs(os.path.dirname(path), exist_ok=True)
      tmp = path + '.tmp'
      with open(tmp, 'w') as fh:
        json.dump({sc: {nm: [str(np.dtype(r.dtype)), list(r.shape)]
                        for nm, r in v.items()} for sc, v in shapes.items()}, fh)
      os.replace(tmp, path)             # atomic: a killed run cannot leave a partial
    except Exception:
      pass
  return shapes


def build(model, batch, num_recycles=0, diffusion_steps=200, num_msa=1,
          flash_attention='xla', seed=0, shape_cache=True, fp32=False):
  """-> (cfg, params, fwd, missing).

  ALWAYS init + merge, for blob-loaded models too: a converter that has not
  ported a head (opendde's confidence head, boltz2's confidence/affinity) leaves
  those scopes absent, and apply would die on the first missing parameter instead
  of tracing. `missing` names every scope left at its init value, so a trace can
  never quietly present init noise as a ported result.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp
  from colabdesign2.af3.alphafold3.model.components import utils
  from colabdesign2.af3.runner import _af3, make_config, resolve_weights

  _hk, _feat, af3_model, _mc, params_io = _af3()
  # the registry owns the arch flags that go with this checkpoint (full_fat for
  # intellifold2, trained_fourier, the default dir) -- do not re-derive them here
  _model, full_fat, trained_fourier, reg_dir = resolve_weights(model)
  cfg = make_config(num_recycles=num_recycles, model=model, num_msa=num_msa,
                    diffusion_steps=diffusion_steps, full_fat=full_fat,
                    trained_fourier=trained_fourier,
                    flash_attention=flash_attention)
  if fp32:
    # BEFORE _param_shapes: eval_shape reads global_config to decide the param
    # dtypes, so flipping this afterwards changes nothing that matters.
    cfg.global_config.bfloat16 = 'none'

  def fwd(b, soft_seq=None, design_mask=None, structure=True, use_dropout=False):
    return af3_model.Model(cfg)(b, soft_seq=soft_seq, design_mask=design_mask,
                                structure=structure, use_dropout=use_dropout)

  path = os.path.expanduser(CHECKPOINTS.get(model) or reg_dir)
  if model in _FROM_TORCH:
    conv = _load_torch_params(model, path)
  else:
    import pathlib
    loaded = params_io.get_model_haiku_params(model_dir=pathlib.Path(path))
    conv = {s: dict(v) for s, v in loaded.items()}

  # Deriving the parameter TREE is the expensive part of a cold run -- not
  # compilation. Measured on a 16-atom ligand: eager init 201s, jax.jit(init) 66s,
  # jax.eval_shape 4.7s; the XLA compile of apply is 3.5s-70s depending only on how
  # many copies of the diffusion body hk.scan's `unroll` hands to XLA. Unjitted haiku
  # dispatches every op of a 48-block graph individually just to learn each weight's
  # shape, which is why init dominates.
  #
  # But we do not need init to RUN: for a fully-converted model every value is
  # overwritten by the checkpoint, and all init is asked for is the tree's shapes
  # and dtypes. jax.eval_shape gives exactly that abstractly, in 4.7s, and it is
  # config-aware for free -- it re-derives from the same graph, so it cannot drift
  # from the model the way a cached/precomputed tree would.
  #
  # (The FILTERED batch matters either way: the raw batch has opaque entries
  # -- "empty_output_struc" is not an abstract array -- which is why this looked
  # un-jittable. remove_invalidly_typed_feats drops exactly those, 64 -> 59 keys,
  # and is what runner.predict already applies before its own jit.)
  init_fn = hk.transform(fwd).init
  key = jax.random.PRNGKey(seed)
  fbatch = utils.remove_invalidly_typed_feats(batch)
  shapes = _param_shapes(model, cfg, init_fn, key, fbatch, use_cache=shape_cache)

  params, missing = {}, []
  for scope in shapes:
    params[scope] = {}
    for name, ref in shapes[scope].items():
      got = conv.get(scope, {}).get(name)
      if got is not None and np.asarray(got).shape == tuple(ref.shape):
        params[scope][name] = jnp.asarray(np.asarray(got, np.float32)).astype(ref.dtype)
      else:
        params[scope][name] = None          # needs a real initialiser value
        missing.append(f'{scope}/{name}')

  if missing:
    # An UNPORTED head (currently only boltz2's confidence head, 66 params) has no
    # converted weights, so it runs on haiku's init values -- which means a real init
    # here, ~35s. That cost is deliberate and NOT cached: the values are random noise
    # feeding a head whose outputs are meaningless, and caching them would only make
    # an unported head cheaper to ignore. `missing` is the signal that a converter is
    # incomplete, and it should stay slightly painful. The fix is to port the head.
    real = jax.jit(init_fn)(key, fbatch)
    for spec in missing:
      scope, name = spec.rsplit('/', 1)
      params[scope][name] = real[scope][name]

  return cfg, params, fwd, missing
