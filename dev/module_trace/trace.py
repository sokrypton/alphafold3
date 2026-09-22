"""Capture: run one (model, case) and record every haiku module's output.

HOW: hk.intercept_methods wraps every module call during apply, so the trace is
the module tree itself -- nothing is instrumented by hand and nothing goes stale
when the graph changes.

WHAT IS CAPTURED: everything. The trunk recycles run in a fori_loop and the
pairformer/diffusion stacks in scans, so at trace time every activation is a
tracer and cannot be read. Instead each output leaf is handed to a
jax.debug.callback, which fires with CONCRETE values at run time -- inside loops
and scans included. A module that runs 48 times (one per pairformer block) or
twice (two recycles) therefore records 48 or 2 entries under the same key, in
execution order, which is the per-layer granularity you want when hunting the
first divergence. The array is fingerprinted inside the callback and dropped, so
tracing a 48-block trunk costs kilobytes, not gigabytes.

FINGERPRINT: shape, dtype, mean/std/min/max, count of non-finite, and a
fixed-seed random projection of the flattened array. The projection is what makes
"identical" mean identical: two runs that differ by a permutation, a transpose or
a single element land on different projections while mean/std can easily agree.
"""
import json
import os
import time

import numpy as np

FINGERPRINT_DIM = 8
_PROJ_SEED = 20260829
_PROJ_MAX = 1 << 18            # elements of a flattened array the projection sees
_PROJ = None                   # built once: regenerating it per call dominated the run


def _projection(n):
  global _PROJ
  if _PROJ is None:
    rng = np.random.default_rng(_PROJ_SEED)
    _PROJ = rng.standard_normal((_PROJ_MAX, FINGERPRINT_DIM))
  return _PROJ[:n]


def fingerprint(x):
  """summary of one array that is cheap to store and hard to alias."""
  a = np.asarray(x)
  out = dict(shape=list(a.shape), dtype=str(a.dtype), size=int(a.size))
  if a.size == 0:
    return out
  f = a.ravel()
  if f.dtype.kind not in 'fc':
    f = f.astype(np.float32)
  finite = np.isfinite(f)
  n_bad = int(f.size - int(finite.sum()))
  out['n_nonfinite'] = n_bad
  g = f if n_bad == 0 else np.where(finite, f, np.float32(0))
  out.update(mean=float(g.mean(dtype=np.float64)), std=float(g.std(dtype=np.float64)),
             min=float(g.min()), max=float(g.max()),
             absmax=float(np.abs(g).max()),
             frac_zero=float((g == 0).mean(dtype=np.float64)))
  n = min(g.size, _PROJ_MAX)
  out['proj'] = (g[:n].astype(np.float64) @ _projection(n)).tolist()
  return out


def _leaves(obj, prefix=''):
  """yield (name, array) for every array leaf in a nested output.

  Arrays are matched FIRST: a jax Array has a __dict__, so the generic
  object-walk below would swallow it and yield nothing.
  """
  import jax
  if isinstance(obj, (np.ndarray, jax.Array)):
    yield prefix, obj
  elif isinstance(obj, dict):
    for k, v in obj.items():
      yield from _leaves(v, f'{prefix}.{k}' if prefix else str(k))
  elif isinstance(obj, (list, tuple)):
    for i, v in enumerate(obj):
      yield from _leaves(v, f'{prefix}[{i}]')
  elif hasattr(obj, '__dict__') and not isinstance(obj, (str, bytes)):
    for k, v in vars(obj).items():
      if not k.startswith('_'):
        yield from _leaves(v, f'{prefix}.{k}' if prefix else str(k))


def _is_array(x):
  import jax
  return isinstance(x, (np.ndarray, jax.Array))


def batch_report(batch, uses):
  """what the model was actually given: per-key stats + a check that every
  declared optional channel is really populated."""
  rep, live = {}, {}
  for k, v in batch.items():
    a = np.asarray(v) if not isinstance(v, dict) else None
    if a is None or a.dtype == object:
      rep[k] = dict(kind='opaque')
      continue
    nz = float((a != 0).mean()) if a.size else 0.0
    rep[k] = dict(shape=list(a.shape), dtype=str(a.dtype), frac_nonzero=nz,
                  n_unique=int(min(np.unique(a).size, 1000)))
  def arr(k, default=None):
    v = batch.get(k)
    return np.asarray(v) if v is not None else default

  msa = arr('msa', np.zeros((0, 0)))
  live['msa'] = int(np.asarray(
      batch.get('msa_mask', np.ones(msa.shape))).any(-1).sum()) > 1
  # AF3's own featurised batches carry template_atom_mask but no template_mask, so
  # keying on the latter alone silently reports "no templates" for a batch with four
  # of them. Accept either.
  tm = arr('template_mask')
  if tm is None:
    tm = arr('template_atom_mask')
  live['template'] = bool(tm.any()) if tm is not None else False
  il = arr('is_ligand')
  live['ligand'] = bool(il.any()) if il is not None else False
  aid = arr('asym_id', np.zeros(1))
  live['multimer'] = int(np.unique(aid).size) > 1
  for nuc in ('rna', 'dna'):
    v = arr('is_' + nuc)
    live[nuc] = bool(v.any()) if v is not None else False

  # An ion is a ligand CHAIN with a single atom. Atom count alone cannot say it:
  # AF3 tokenises every ligand per-atom, so all ligand tokens have one atom.
  rm = arr('ref_mask')
  ions = 0
  if il is not None and rm is not None and il.any():
    for a in np.unique(aid[il.astype(bool)]):
      if rm[aid == a].sum() == 1:
        ions += 1
  live['ion'] = ions > 0

  # A covalent polymer<->ligand link (glycosylation, a covalent inhibitor). Distinct
  # from the ligand_ligand bonds, which are just the internal bonds of any atomised
  # component and so are present for essentially every ligand/NA case.
  pl = arr('token_atoms_to_polymer_ligand_bonds:gather_mask')
  live['bond'] = bool(pl.any()) if pl is not None else False
  ll = arr('tokens_to_ligand_ligand_bonds:gather_mask')
  live['atomized'] = bool(ll.any()) if ll is not None else False

  # Not separable once featurised, so do not pretend to check them: a SMILES ligand
  # produces the same features as the same molecule from the CCD; a glycan is a
  # covalently bonded ligand; and a modified residue keeps its parent aatype (UNK
  # count is 0 across all of AF3's PTM examples), showing up only as atomisation.
  for opaque in ('smiles', 'glycan', 'ptm'):
    live[opaque] = None
  # None = channel is real but not decidable from the batch; only False is a failure.
  unmet = [u for u in uses if live.get(u) is False]
  return rep, live, unmet


def run(model, case_name, out_root, label=None, num_recycles=0,
        diffusion_steps=200, seed=1, save_arrays=False, max_array_mb=8.0,
        max_calls_per_site=8):
  """trace one (model, case) and write it to out_root/<model>/<case>/<label>/."""
  import haiku as hk
  import jax

  from . import cases, models

  case = cases.build(case_name, **models.featurise_kwargs(model))
  batch = case['batch']
  cfg, params, fwd, missing = models.build(
      model, batch, num_recycles=num_recycles, diffusion_steps=diffusion_steps,
      num_msa=case.get('num_msa') or 1)

  # key -> list of fingerprints, one per RUNTIME call (48 for a per-block module
  # inside the pairformer scan, one per recycle for the trunk, ...).
  store, counts, order = {}, {}, []

  calls = {}

  def record(key, value):
    n = calls.get(key, 0) + 1
    calls[key] = n
    if key not in store:
      store[key] = []
      order.append(key)
    # a repetitive loop (the 200-step sampler) is fully COUNTED but only the
    # first `max_calls_per_site` iterations are fingerprinted
    if n <= max_calls_per_site:
      store[key].append(fingerprint(value))

  def interceptor(next_f, args, kwargs, context):
    out = next_f(*args, **kwargs)
    base = f'{context.module.module_name}/{context.method_name}'
    i = counts.get(base, 0)
    counts[base] = i + 1
    site = base if i == 0 else f'{base}#{i}'      # distinct CALL SITES in the graph
    for name, v in _leaves(out):
      if not _is_array(v):
        continue
      key = f'{site}:{name}' if name else site
      jax.debug.callback(lambda x, k=key: record(k, x), v)
    return out

  # same prep AF3Runner.predict does: drop object-dtype feats and make every
  # leaf a jax array, else shuffle_msa indexes a numpy array with a tracer.
  from colabdesign2.af3.alphafold3.model.components import utils
  live_batch = jax.tree_util.tree_map(
      jax.numpy.asarray, utils.remove_invalidly_typed_feats(batch))

  t0 = time.time()
  f = hk.transform(fwd)
  with hk.intercept_methods(interceptor):
    out = f.apply(params, jax.random.PRNGKey(seed), live_batch)
  jax.block_until_ready(jax.tree_util.tree_leaves(out))
  wall = time.time() - t0

  # one entry per call site: the fingerprint list in execution order, plus how
  # many times it ran (a changed call count is itself a graph difference).
  modules = {k: dict(calls=calls[k], fp=store[k]) for k in order}
  outputs = {k: dict(calls=1, fp=[fingerprint(np.asarray(v))])
             for k, v in _leaves(out) if _is_array(v)}

  brep, live, unmet = batch_report(batch, case['uses'])
  metrics = _metrics(out, case)

  label = label or time.strftime('%Y%m%d-%H%M%S')
  d = os.path.join(os.path.expanduser(out_root), model, case_name, label)
  os.makedirs(d, exist_ok=True)
  manifest = dict(
      model=model, case=case_name, label=label,
      created=time.strftime('%Y-%m-%dT%H:%M:%S'), wall_seconds=round(wall, 1),
      num_recycles=num_recycles, diffusion_steps=diffusion_steps, seed=seed,
      max_calls_per_site=max_calls_per_site,
      n_call_sites=len(modules),
      n_module_calls=int(sum(m['calls'] for m in modules.values())),
      n_fingerprints=int(sum(len(m['fp']) for m in modules.values())),
      call_order=order,
      params_left_at_init=missing[:50], n_params_left_at_init=len(missing),
      inputs=brep, channels_live=live, channels_declared=list(case['uses']),
      channels_unmet=unmet, metrics=metrics,
      modules=modules, outputs=outputs)
  with open(os.path.join(d, 'manifest.json'), 'w') as fh:
    json.dump(manifest, fh, indent=1, sort_keys=True)

  if save_arrays:
    keep = {k: v for k, v in store if v.nbytes <= max_array_mb * 1e6}
    np.savez_compressed(os.path.join(d, 'modules.npz'), **keep)
  return d, manifest


def _metrics(out, case):
  """end-to-end numbers worth carrying alongside the per-module trace."""
  m = {}
  try:
    x = np.asarray(out['diffusion_samples']['atom_positions'])
    while x.ndim > 4:
      x = x[0]
    x = x[0] if x.ndim == 4 else x
    ca = x[:, 1, :]
    d = np.sqrt(((ca[1:] - ca[:-1]) ** 2).sum(-1))
    m['ca_ca_mean'] = float(d.mean())
    nat = case.get('native_ca')
    if nat is not None and len(nat) <= len(ca):
      m['ca_rmsd'] = _kabsch(ca[:len(nat)], np.asarray(nat))
  except Exception as e:                        # structure path off / no samples
    m['structure_error'] = f'{type(e).__name__}: {e}'
  for key, path in (('plddt', ('predicted_lddt', 'predicted_lddt')),
                    ('plddt2', ('plddt',)), ('pae', ('predicted_aligned_error',))):
    node = out
    for p in path:
      node = node.get(p) if isinstance(node, dict) else None
      if node is None:
        break
    if node is not None and not isinstance(node, dict):
      m[key] = float(np.asarray(node).mean())
  return m


def _kabsch(P, Q):
  P = P - P.mean(0); Q = Q - Q.mean(0)
  V, _, W = np.linalg.svd(P.T @ Q)
  d = np.sign(np.linalg.det(V @ W))
  U = V @ np.diag([1, 1, d]) @ W
  return float(np.sqrt(((P @ U - Q) ** 2).sum(-1).mean()))
