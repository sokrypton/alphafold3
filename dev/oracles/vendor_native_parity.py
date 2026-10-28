"""A ported model against its STOCK vendor, whole model, every input modality.

  ~/venv/bin/python dev/oracles/vendor_native_parity.py MODEL [CASE ...] [--recycles N]

Both sides run their SHIPPED entry points on the same input. The vendor side is
a per-family runner (VENDORS below) executed in the vendor's own environment:
its own featuriser, its own model loader, its own top-level forward. Ours: our
featurisation with the model's spec, `Model.__call__`. The only things either
side is handed are RANDOM DRAWS -- the vendor's reference-conformer draw
(`ref_pos`), the samples the confidence head scores, and one (noised
coordinates, sigma) the denoiser runs on -- so no constructor argument, flag or
feature is chosen by this file. That is the point: the hand-built adapters this
replaces as the gate of record each re-derived the vendor's setup and could get
it wrong silently (boltz2's fix_sym_check, 2026-10-18).

RUNNER CONTRACT (see boltz2_upstream_runner.py, protenix_upstream_runner.py):
  featurise MODEL CASE.json DIR      -> DIR/layout.npz: atom_name, atom_token,
                                        atom_pad, token_pad, feat.ref_pos,
                                        feat.ref_space_uid
  forward MODEL DIR IN.npz OUT.npz   <- positions, positions_noisy, noise_level,
                                        recycling_steps, precision (all in the
                                        vendor's flat atom order)
                                     -> s_inputs, s, z, distogram_logits,
                                        denoised, pae, pde, plddt

Graded, per case: s_inputs, trunk single/pair after recycling, distogram
logits, one denoise step, pLDDT / PAE / PDE. Atoms are aligned by (token, atom
name), so an atom one side has and the other does not is REPORTED, never
silently dropped. Prints `__VERDICT PASS|FAIL`.
"""
import os as _os, sys as _sys  # dev/af3_dev_paths.py: vendor trees, venvs and weights
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from af3_dev_paths import dev_path  # noqa: E402
import json
import os
import subprocess
import sys
import tempfile

# Precision is not the model: XLA's default GPU matmul is TF32-class, which
# alone reads the boltz2 denoise step at 2e-2 where 'highest' reads 3e-5.
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import af3_native_parity as P   # noqa: E402  (cases, compare, grade)

_H = dev_path('~')
# model -> (interpreter, PYTHONPATH for the vendor's tree or None, runner)
VENDORS = {
    'boltz2': (_H + '/venv_boltz/bin/python', None, 'boltz2_upstream_runner.py'),
    'protenix2': (_H + '/venv_protenix/bin/python', _H + '/protenix',
                  'protenix_upstream_runner.py'),
    'protenix1': (_H + '/venv_protenix/bin/python', _H + '/protenix',
                  'protenix_upstream_runner.py'),
    # OpenDDE v1.1.0 in protenix's venv: every pin in its pyproject (torch
    # 2.7.1, numpy 2.4.1, rdkit 2025.9.3, biotite 1.4.0, ...) is met exactly.
    'opendde': (_H + '/venv_protenix/bin/python', _H + '/OpenDDE',
                'opendde_upstream_runner.py'),
    # OpenFold3: ONE runner, the tree picks the release. openbind0 is v0.5.0
    # (worktree ~/openfold-3-v050) -- main loads its tensors but runs the other
    # triangle-attention convention (run_all_parity.sh). ~/venv_of3 serves both:
    # the two trees' conda locks are identical (pins in
    # ~/venv_of3/requirements-of3.txt; deepspeed/cuequivariance left out).
    # openfold3 = stock release 0.4.5 (a4b08032), the newest that loads the
    # preview-2 checkpoint; origin/main (v0.5.0-155-g8090aa69) refuses it.
    'openfold3': (_H + '/venv_of3/bin/python', _H + '/openfold-3-045',
                  'openfold3_upstream_runner.py'),
    'openbind0': (_H + '/venv_of3/bin/python', _H + '/openfold-3-v050',
                  'openfold3_upstream_runner.py'),
    # IntelliFold v2.0.4's native PyTorch stack. Its pyproject pins only the
    # JAX wrapper, so ~/venv_intellifold (uv, python 3.11) carries the pins of
    # the last wheel that shipped the torch stack (pypi/intellifold-2.0.2:
    # torch 2.6.0, numpy 1.24.0, rdkit 2024.3.2, accelerate 1.1.1, ...) minus
    # deepspeed, a CUDA-build kernel extra the stack only uses when asked.
    'intellifold2': (_H + '/venv_intellifold/bin/python', _H + '/IntelliFold',
                     'intellifold2_upstream_runner.py'),
    # RoseTTAFold3 (foundry @ aad357b). ~/venv_rf3 (uv, python 3.12) pins
    # foundry's pyproject deps under its own ci-constraints.txt (torch 2.7.1,
    # rdkit 2025.3.6; atomworks 2.2.1) in ~/venv_rf3/requirements-rf3.txt,
    # minus the `rf3` extra (cuequivariance: CUDA-build fused kernels).
    'rosettafold3': (_H + '/venv_rf3/bin/python',
                     _H + '/foundry_rf3/src:' + _H + '/foundry_rf3/models/rf3/src',
                     'rosettafold3_upstream_runner.py'),
    # chai-1: chai_lab 0.6.1 from PyPI in ~/venv_chai (python 3.10, torch
    # 2.6.0), isolated; its components from CHAI_DOWNLOADS_DIR.
    # CHAI_VENDOR_PY overrides the interpreter (a host whose ~/venv_chai is not this one)
    'chai1': (os.environ.get('CHAI_VENDOR_PY', _H + '/venv_chai/bin/python'), None,
              'chai1_upstream_runner.py'),
    # ESMFold2: esm 3.4.1 (biohub's own package) in ~/venv_esm341, offline
    # against the cached biohub checkpoints.
    'esmfold2': (_H + '/venv_esm341/bin/python', None, 'esmfold2_upstream_runner.py'),
    'esmfold2_fast': (_H + '/venv_esm341/bin/python', None, 'esmfold2_upstream_runner.py'),
    # the lm-tier pair: Synthyra's releases on Synthyra's own code (FastPLMs,
    # trust_remote_code), in ~/venv_synthyra built from their requirements.txt
    'esmfold2_lm600m': (_H + '/venv_synthyra/bin/python', None, 'esmfold2_upstream_runner.py'),
    'esmfold2_lm300m': (_H + '/venv_synthyra/bin/python', None, 'esmfold2_upstream_runner.py'),
    # BioEmu2 @7846769: ~/bioemu2 installed into ~/venv_bioemu2 by its own lock
    # (`uv sync --frozen --extra cuda`), checkpoint ~/bioemu2_ckpt.
    'bioemu2': (_H + '/venv_bioemu2/bin/python', None, 'bioemu2_upstream_runner.py'),
    # HelixFold3 (PaddleHelix @8e3991a, apps/protein_folding/helixfold3) in
    # ~/venv_helixfold3 (python 3.10, paddlepaddle-gpu 3.1.0 cu126 + its
    # requirements.txt); the runner puts the app on its own path.
    'helixfold3': (_H + '/venv_helixfold3/bin/python', None, 'helixfold3_upstream_runner.py'),
    # Mol-1 @bfc01fa (~/vendor_src/Mol-1, a Protenix bd54a05 fork) installed editable into
    # ~/venv_mol1 (python 3.12, torch 2.12 cu130, its requirements.txt pins minus the optional
    # opt_core / cuequivariance kernels); reference data in ~/mol1_data, checkpoint ~/mol1_weights.
    'mol1': (_H + '/venv_mol1/bin/python', None, 'mol1_upstream_runner.py'),
}


VENDOR_NA_EXIT = 3


class VendorNA(Exception):
  """The stock vendor has no way to take this input (runner exit 3)."""


def _vendor(model, *args):
  py, path, runner = VENDORS[model]
  env = {k: v for k, v in os.environ.items()
         if k not in ('PYTHONPATH', 'JAX_PLATFORMS')}
  # -I isolates from OUR tree; a vendor that is a source checkout instead gets
  # exactly that checkout on its path (-P: not the runner's own directory).
  cmd = [py, '-I'] if path is None else [py, '-P']
  if path is not None:
    env['PYTHONPATH'] = path
  if os.environ.get('VENDOR_CPU'):
    env['CUDA_VISIBLE_DEVICES'] = ''
  env.setdefault('MSA_ALL_ROWS', '1')
  if model.startswith('esmfold2'):
    env.setdefault('HF_HUB_OFFLINE', '1')
  if model == 'chai1':
    env.setdefault('CHAI_DOWNLOADS_DIR', _H + '/chai1_weights')
    env.setdefault('DISABLE_PANDERA_IMPORT_WARNING', 'True')
    # chai orders each chain's unpaired MSA rows with pandas' unstable sort, and numpy 1.26 sends that argsort to
    # x86-simd-sort on an AVX-512 CPU: equal keys then come out in another order, so which rows two chains pair in the
    # merged MSA - what the outer product reads - depends on the CPU stock runs on. Ours ports the generic introsort
    # (model_features._np126_argsort), which is what stock takes on every CPU without AVX-512; this pins stock to it
    # here. (A10 host, Xeon with AVX-512: paired_msa read trunk pair corr 0.9974 without this, PASS with it.)
    env.setdefault('NPY_DISABLE_CPU_FEATURES', 'AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX '
                                               'AVX512_CNL AVX512_ICL AVX512_SPR')
  r = subprocess.run(cmd + [os.path.join(HERE, runner), args[0], model,
                            *args[1:]],
                     capture_output=True, text=True, env=env,
                     cwd=tempfile.gettempdir())
  # Exit 3 = the STOCK vendor cannot express this input at all (openfold3 has
  # no bond input, so `covalent` is N/A there): reported as N/A, never PASS,
  # and never confused with a crash, which stays a FAILURE.
  if r.returncode == VENDOR_NA_EXIT:
    raise VendorNA(r.stderr.strip().splitlines()[-1] if r.stderr.strip()
                   else 'input not expressible in the vendor')
  if r.returncode:
    raise RuntimeError('%s %s failed:\n%s' % (model, args[0], r.stderr[-4000:]))


def _s_inputs_to_ours(model, x):
  """The vendor's s_inputs in OUR column order, where the layouts differ.

  The OF3-derived families (protenix, opendde, ...) are 449 wide with their own
  restype order; ours is 447 = their two 32-class one-hots permuted to AF3's
  order, the deletion mean, and the 384 atom-pooled channels -- the same map
  denoise_parity and conditioning_parity build their inputs with.
  """
  if x.shape[-1] == 451 and model.startswith('esmfold2'):
    # ESMFold2: 384 atom channels, then 33-class restype and profile one-hots
    # in its own alphabet (converters.esmfold2.esm_class_of_af3), then the
    # deletion mean; ours is [restype 31 | profile 31 | deletion mean | atom].
    from converters.esmfold2 import esm_class_of_af3
    cls = np.asarray(esm_class_of_af3(31))
    idx = np.concatenate([384 + cls, 417 + cls, [450], np.arange(384)])
    return x[..., idx]
  if x.shape[-1] == 449:
    from converters.openfold3 import _AF3_TO_OF3_AATYPE as remap
    # rosettafold3 (atomworks' alphabet) swaps G/C and DG/DC relative to OF3's
    # (converters/rosettafold3.py) -- its own remap, or every nucleic s_inputs
    # row reads as a disagreement.
    if model == 'rosettafold3':
      from converters.rosettafold3 import _AF3_TO_RF3_AATYPE as remap
    if model == 'helixfold3':
      # its DN rides two trailing columns of ours (model.hf3_dn_columns)
      from converters.helixfold3 import AF3_TO_HF3 as remap, HF3_UNK_DNA as dn
      return x[..., np.concatenate([384 + remap, 416 + remap, [448], np.arange(384),
                                    [384 + dn, 416 + dn]])]
    idx = np.concatenate([384 + np.asarray(remap), 416 + np.asarray(remap),
                          [448], np.arange(384)])
    return x[..., idx]
  return x


def _names(chars):
  """AF3/boltz atom-name chars (ord - 32) -> stripped strings."""
  return np.array([''.join(chr(int(c) + 32) for c in row).strip()
                   for row in chars.reshape(-1, chars.shape[-1])]
                  ).reshape(chars.shape[:-1])


def _tower_rows(model, j, hidden):
  """OUR language-model tower against the vendor's, on the same sequences (all
  chains proteins only -- the tower's input layout for anything else is the
  featuriser's business, graded by the fold rows). The vendor tower runs fp32
  under ESM_TOWER_FP32=1; otherwise it is stock bf16 and the row is INFO."""
  kinds = [k for entry in j['sequences'] for k in entry]
  if set(kinds) != {'protein'} or any(entry['protein'].get('modifications') for entry in j['sequences']):
    return []
  from alphafold3.model import esm, model_registry
  from alphafold3.model import weights as weights_lib
  tower = model_registry.ESMFOLD2_VARIANTS[model]['esmc']
  d = weights_lib.default_dir(tower)
  if not os.path.isdir(d):
    return [('lm', 'tower %s: our blob not at %s' % (tower, d), 'INFO')]
  seqs = []
  for entry in j['sequences']:
    e = entry['protein']
    seqs += [e['sequence']] * (len(e['id']) if isinstance(e['id'], list) else 1)
  ours = np.asarray(esm.embed(seqs, d, 'esmc', tower), np.float32)
  ref = np.load(hidden)['lm_hidden'].astype(np.float32)
  if ours.shape != ref.shape:
    return [('lm', 'tower hidden shape ours %s vendor %s' % (ours.shape, ref.shape), 'BAD')]
  rows = []
  for label, a, r in (('tower hidden, last layer', ours[:, -1], ref[:, -1]),
                      ('tower hidden, all layers', ours, ref)):
    c, ratio = P.compare(a, r)
    rows.append(('lm', '%s corr %.6f max|d|/rms %.2e' % (label, c, ratio),
                 P.grade(c, ratio) if os.environ.get('ESM_TOWER_FP32') else 'INFO'))
  return rows


def _chai_esm(model, j, bdir):
  """chai-1 folds from ESM2, as run_alphafold.py runs it: ours from the vendor's
  own rows (chai1_upstream_runner's esm2_rows.npz, through fold_check's ESM_EMB),
  so the fold rows grade the model, and OUR tower against those rows on its own
  (CHAI_ESM_TOWER=1, one case at a time). chai's tower is its fp16 trace, so that row is INFO. Cleared for every other
  case, so one case's rows never reach the next."""
  path = os.path.join(bdir, 'esm2_rows.npz')
  if model != 'chai1' or not os.path.exists(path):
    os.environ.pop('ESM_EMB', None)
    return []
  os.environ['ESM_EMB'] = path
  # CHAI_ESM_TOWER=1 only: our 3B tower stays resident in this process (8.75 GiB on an A10), and every later case's
  # vendor forward then runs out of memory beside it
  if not os.environ.get('CHAI_ESM_TOWER'):
    return []
  if any(entry['protein'].get('modifications') for entry in j['sequences'] if 'protein' in entry):
    return []
  from alphafold3.model import esm
  # the shipped (int8) tower, ~/ported/esm2; ESM2_DIR names another (~/ported/esm2_fp32 reads ~1e-2 against chai's)
  d = os.environ.get('ESM2_DIR') or dev_path('~/ported/esm2')
  if not os.path.isdir(d):
    return [('lm', 'tower esm2: our blob not at %s' % d, 'INFO')]
  seqs = []
  for entry in j['sequences']:
    if 'protein' in entry:
      e = entry['protein']
      seqs += [e['sequence']] * (len(e['id']) if isinstance(e['id'], list) else 1)
  ours = np.asarray(esm.embed(seqs, d, 'esm2'), np.float32)
  ref = np.load(path)['esm'].astype(np.float32)
  if ours.shape != ref.shape:
    return [('lm', 'esm2 rows shape ours %s vendor %s' % (ours.shape, ref.shape), 'BAD')]
  c, ratio = P.compare(ours, ref)
  return [('lm', 'esm2 rows (ours vs chai fp16 trace) corr %.6f max|d|/rms %.2e' % (c, ratio), 'INFO')]


# run_alphafold.py's --buckets default: OURS_BUCKET=shipped pads ours exactly as
# a real run does (the first bucket at or above the token count)
_SHIPPED_BUCKETS = [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 2560,
                    3072, 3584, 4096, 4608, 5120]


# Vendors whose MODEL wraps a cyclic polymer (boltz-schema `cyclic: true`).
# IntelliFold-2 parses the flag into cyclic_period, but its
# RelativePositionEncoding never reads it: stock, cyclic is a no-op there.
CYCLIC_VENDORS = ('boltz2',)
# the constraint kinds each vendor was trained to read (the input's `constraints`; model_features)
CONSTRAINT_VENDORS = {'boltz2': {'pocket', 'contact'}, 'chai1': {'pocket', 'contact'}, 'bioemu2': {'contact'}}
_CYCLIC_IGNORED = {'intellifold2': 'its RelativePositionEncoding ignores cyclic_period'}


def _cyclic_chains(j):
  """Chain ids of the protein entries flagged `cyclic` (a harness key, not
  AF3's: our CLI's --cyclic, the vendors' yaml `cyclic: true`)."""
  out = []
  for entry in j['sequences']:
    (kind, e), = entry.items()
    if e.get('cyclic'):
      out += e['id'] if isinstance(e['id'], list) else [e['id']]
  return out


def _strip_cyclic(j):
  j = json.loads(json.dumps(j))
  for entry in j['sequences']:
    for e in entry.values():
      e.pop('cyclic', None)
  return j


def featurise_ours(model, j, recycles, bf16):
  import fold_check
  from alphafold3.common import folding_input
  cyclic = _cyclic_chains(j)
  fi = folding_input.Input.from_json(json.dumps(_strip_cyclic(j)))
  # MODEL_DIR (fold_check's convention): an alternative blob for the same
  # model, e.g. intellifold2 converted with IF2_FP32_BLOB=1, to tell its bf16
  # trunk storage policy from a port bug.
  batch, cfg, model_dir = fold_check._fold_setup(
      model, None, model_dir=os.environ.get('MODEL_DIR'), chains=list(fi.chains),
      bonded_atom_pairs=fi.bonded_atom_pairs, seed=1, cyclic=cyclic,
      # OURS_NO_CONSTRAINTS=1: the negative control -- the vendor conditioned, ours not; a constraint case must FAIL
      constraints=() if os.environ.get('OURS_NO_CONSTRAINTS') else fi.constraints,
      # OURS_BUCKET=256: pad ours to the vendor's token bucket (chai always runs
      # at one), a probe for padding leaking into the vendor's outputs
      **({'buckets': _SHIPPED_BUCKETS if os.environ['OURS_BUCKET'] == 'shipped'
          else [int(os.environ['OURS_BUCKET'])]} if os.environ.get('OURS_BUCKET') else {}))
  cfg.global_config.bfloat16 = bf16
  cfg.num_recycles = recycles
  # Random MSA row subsampling is a draw like any other: above num_msa rows ours
  # samples with its key and every vendor with its own RNG, so a deep MSA would
  # compare two random subsets. Both sides take every row instead (the runners
  # lift their caps, MSA_ALL_ROWS); the MSA stack is row-permutation invariant.
  # LIVE rows: the featuriser pads the MSA to 16384, and lifting num_msa to the padded depth ran boltz2's MSA stack
  # over 16384 masked rows on a single-sequence input (long_chain: one 14 GiB transition buffer, OOM on 22 GB)
  rows = (int(np.asarray(batch['msa_mask']).any(-1).sum()) if 'msa_mask' in batch
          else np.asarray(batch['msa']).shape[0] if 'msa' in batch else 0)
  if rows > cfg.evoformer.num_msa:
    cfg.evoformer.num_msa = rows
  # chai runs every row to its 16384 depth, padding included, and so does ours by default - but a padded row is inert
  # (protein, msa and long_chain read the same PARITY on live rows alone), and the padded stack does not fit an A10 at
  # float32 past a few hundred tokens: long_chain's pair-weighted average is one 7.2 GiB buffer. So ours takes the live
  # rows; AF3_CHAI_NUM_MSA set by hand keeps whatever it names.
  if model == 'chai1' and 'AF3_CHAI_NUM_MSA' not in os.environ:
    cfg.evoformer.num_msa = max(rows, 1)
  cfg.return_embeddings = True
  cfg.return_distogram = True
  return batch, fold_check.LAST_RAW_BATCH, cfg, model_dir


def _pad_pair(x, batch):
  """A vendor (real tokens)^2 pair array onto our token axis, which OURS_BUCKET may pad."""
  n = np.asarray(batch['token_index']).shape[-1]
  x = np.asarray(x)
  if x.shape[0] == n:
    return x
  out = np.zeros((n, n) + x.shape[2:], x.dtype)
  out[:x.shape[0], :x.shape[1]] = x
  return out


def _inject_ref_only(rp, raw, lay):
  """The vendor's reference draw for atoms it keeps UNPLACED (layout
  `refonly.*`: chai's leaving-atom tokens), into ours by (token, name) --
  align() matches placed atoms only, so these kept our own conformer."""
  if 'refonly.atom_name' not in lay or not len(lay['refonly.atom_name']):
    return []
  names = _names(np.asarray(raw['ref_atom_name_chars']))
  mask = np.asarray(raw['ref_mask']).astype(bool)
  where = {(t, names[t, s]): (t, s) for t, s in zip(*np.nonzero(mask))}
  n = 0
  for nm, t, p in zip(lay['refonly.atom_name'], lay['refonly.atom_token'], lay['refonly.ref_pos']):
    ts = where.get((int(t), str(nm)))
    if ts is not None:
      rp[ts] = p
      n += 1
  return [('feat', 'reference draw injected for %d of %d unplaced vendor atom(s)'
           % (n, len(lay['refonly.atom_name'])), 'INFO')]


def align(raw, lay):
  """-> (ours (token, slot) for each boltz atom or (-1,-1), report rows)."""
  ours_names = _names(np.asarray(raw['ref_atom_name_chars']))
  ours_mask = np.asarray(raw['ref_mask']).astype(bool)
  n_tok_o = ours_mask.shape[0]
  if 'seq_mask' in raw:
    # ours may be padded to a bucket (OURS_BUCKET): count the real tokens
    n_tok_o = int(np.asarray(raw['seq_mask']).astype(bool).sum())
  tp = lay['token_pad']
  rows = []
  if int(tp.sum()) != n_tok_o:
    rows.append(('align', 'token count ours %d boltz %d' % (n_tok_o,
                                                            int(tp.sum())),
                 'BAD'))
  where = {}
  for t in range(n_tok_o):
    for s in np.flatnonzero(ours_mask[t]):
      where[(t, ours_names[t, s])] = (t, s)
  idx = np.full((len(lay['atom_name']), 2), -1)
  missing_ours = []
  for i, (nm, t, real) in enumerate(zip(lay['atom_name'], lay['atom_token'],
                                        lay['atom_pad'])):
    if not real:
      continue
    if (t, nm) in where:
      idx[i] = where.pop((t, nm))
    else:
      missing_ours.append('%d:%s' % (t, nm))
  # SECOND PASS, for atoms with no canonical name -- a SMILES ligand's are
  # invented independently on each side (boltz numbers them C22..., we C1...).
  # Both parse the same SMILES with RDKit and give each ligand atom its own
  # token, so within a token a single unmatched atom on each side is the same
  # atom if the ELEMENT agrees. Reported, so a name convention is never mistaken
  # for agreement on atoms.
  if missing_ours:
    o_el = np.asarray(raw['ref_element'])
    b_el = (lay['feat.ref_element'].argmax(-1)
            if lay['feat.ref_element'].ndim == 2 else lay['feat.ref_element'])
    left_o = {}
    for (t, nm), ts in where.items():
      left_o.setdefault(t, []).append(ts)
    still, by_elem = [], 0
    for item in missing_ours:
      t, nm = item.split(':', 1)
      t = int(t)
      i = next(k for k, (n2, t2, r2) in enumerate(zip(
          lay['atom_name'], lay['atom_token'], lay['atom_pad']))
               if r2 and t2 == t and n2 == nm)
      cand = left_o.get(t, [])
      if len(cand) == 1 and int(o_el[cand[0]]) == int(b_el[i]):
        idx[i] = cand[0]
        by_elem += 1
        where.pop((t, _names(np.asarray(raw['ref_atom_name_chars'])[cand[0]][None])[0]), None)
        left_o[t] = []
      else:
        still.append(item)
    if by_elem:
      rows.append(('align', '%d atom(s) matched by token+element, names differ '
                   '(SMILES naming)' % by_elem, 'INFO'))
    missing_ours = still
  if missing_ours:
    rows.append(('align', 'boltz atoms ours lacks: %s' % ' '.join(
        missing_ours[:8]), 'BAD'))
  if where:
    rows.append(('align', 'our atoms boltz lacks: %s' % ' '.join(
        '%d:%s' % k for k in list(where)[:8]), 'INFO'))
  return idx, rows


def forward_ours(batch, cfg, model_dir, pos, pos_noisy, sigma, real_sampler=False):
  import haiku as hk
  import jax
  import jax.numpy as jnp
  from alphafold3.model import model as af3_model
  from alphafold3.model import params as afp
  from alphafold3.model import model_config
  if os.environ.get('HF3_NO_BF16_BIAS'):
    # diagnostic, paired with the runner's: no bf16 rounding of the triangle bias on either side
    model_config.BF16_TRI_ATT_PAIR_BIAS = ()
  # ESMFold2's LM-pair dropout is a fresh random mask every pass at inference
  # (esm fold()'s lm_dropout 0.3); the vendor runner turns it off, so ours does
  model_config.LM_PAIR_DROPOUT = {m: 0.0 for m in model_config.LM_PAIR_DROPOUT}
  if not real_sampler:
    pos, pos_noisy = jnp.asarray(pos), jnp.asarray(pos_noisy)
    sigma = jnp.float32(sigma)

  class Injected(af3_model.Model):

    @hk.transparent
    def _sample_diffusion(self, batch, embeddings, *, sample_config,
                          diffusion_state=None):
      pc, ac = self.diffusion_module(
          positions_noisy=None, noise_level=jnp.zeros(()), batch=batch,
          embeddings=embeddings, use_conditioning=True, conditioning_only=True)
      den = self.diffusion_module(
          positions_noisy=pos_noisy, noise_level=sigma, batch=batch,
          embeddings=embeddings, use_conditioning=True,
          pair_cond=pc, atom_cond=ac)
      m = batch.predicted_structure_info.atom_mask
      return {'atom_positions': pos[None], 'mask': m[None],
              'denoised': den[None]}

  from af3_upstream_runner import tapping
  taps, outer = {}, [None]

  @hk.transform
  def fwd(b):
    outer[0] = jax.tree_util.tree_leaves(b)[0]._trace
    if os.environ.get('B2_KEEP'):
      with tapping(taps, outer):
        out = (af3_model.Model if real_sampler else Injected)(cfg)(b)
      return {**out, **taps}
    return (af3_model.Model if real_sampler else Injected)(cfg)(b)
  params = afp.get_model_haiku_params(model_dir=model_dir)
  out = jax.jit(fwd.apply)(params, jax.random.PRNGKey(0), batch)
  if os.environ.get('AF3_ESM_TRUNK_TAPS'):
    # ESMFold2's seams (evoformer._esm_tap: z_init, z_inject, z_parcae, ...)
    # record only in an EAGER run -- under jit they are tracers. Slow; debug only.
    from alphafold3.model.network import evoformer as _evo
    _evo.ESM_TRUNK_TAPS.clear()
    with jax.disable_jit():
      hk.transform(lambda b: Injected(cfg)(b)).apply(params, jax.random.PRNGKey(0), batch)
    out = dict(out, **{'esmtap/%s/%d' % (k, i): np.asarray(v)
                       for k, vs in _evo.ESM_TRUNK_TAPS.items() for i, v in enumerate(vs)})
  if os.environ.get('B2_KEEP'):
    # ONE trunk pass through the staged entry point, tapped: the fused path
    # runs the trunk inside a fori_loop, whose values cannot leave the loop.
    from alphafold3.model import staged as _staged
    st_taps, st_outer = {}, [None]

    def _trunk_tapped(b):
      st_outer[0] = jax.tree_util.tree_leaves(b)[0]._trace
      m = af3_model.Model(cfg)
      carry, k = m(b, stage='embed')
      # Every pass but the last untapped; the last one tapped, so a gap that
      # only appears with recycling is localised on the pass where it appears.
      for _ in range(int(os.environ.get('B2_TAP_PASS', 1)) - 1):
        carry, k, _ = m(b, recycle_carry=carry, key=k, stage='trunk')
      with tapping(st_taps, st_outer):
        carry, k, _ = m(b, recycle_carry=carry, key=k, stage='trunk')
      return {**{'staged/' + kk: v for kk, v in carry.items()
                 if hasattr(v, 'shape')}, **st_taps}
    tt = hk.transform(_trunk_tapped)
    out = {**out, **jax.jit(tt.apply)(params, jax.random.PRNGKey(0), batch)}
  from af3_upstream_runner import _flatten
  return _flatten(jax.device_get(out))


def _rotations(rng, n):
  """n proper rotations, Gram-Schmidt on two normal vectors (our sampler's own draw)."""
  out = np.zeros((n, 3, 3), np.float32)
  for k in range(n):
    v0, v1 = rng.normal(size=(2, 3))
    e0 = v0 / np.linalg.norm(v0)
    v1 = v1 - e0 * (v1 @ e0)
    e1 = v1 / np.linalg.norm(v1)
    out[k] = np.stack([e0, e1, np.cross(e0, e1)])
  return out


TRAJ_MAX_STEPS = 400


def trajectory_case(model, name, j, recycles):
  """The FULL sampler, both sides, on the same random draws (--trajectory).

  The single-step rows inject one noised input; this runs each side's real
  sampler at its OWN default settings, with every random draw replaced BY
  MEANING: the initial standard-normal positions, and per step the augmentation
  rotation and translation and the noise. The vendor reports the steps it took
  and its sigma schedule; a different count, schedule, noise or step scale, or
  augmentation convention then shows up as a different trajectory.
  """
  if model == 'bioemu2':
    if os.environ.get('BIOEMU2_SAMPLER'):
      return bioemu2_steer(name, j, recycles)
    return bioemu2_trajectory(name, j, recycles)
  import jax.numpy as jnp
  from alphafold3.model.network import diffusion_head as dh
  rows = []
  with tempfile.TemporaryDirectory() as tmp:
    cj = os.path.join(tmp, 'case.json')
    with open(cj, 'w') as f:
      json.dump(j, f)
    bdir = os.path.join(tmp, 'vendor')
    _vendor(model, 'featurise', cj, bdir)
    lay = dict(np.load(os.path.join(bdir, 'layout.npz')))
    # ESMFold2: ours folds from the vendor's own ESM-C hidden states, as run_case
    hidden = os.path.join(bdir, 'esmc_hidden.npz')
    if os.path.exists(hidden):
      os.environ['ESMC_HIDDEN'] = hidden
    rows += _chai_esm(model, j, bdir)
    batch, raw, cfg, model_dir = featurise_ours(model, j, recycles, 'none')
    idx, arows = align(raw, lay)
    rows += arows
    ok = idx[:, 0] >= 0
    rp = np.array(batch['ref_pos'])
    rp[idx[ok, 0], idx[ok, 1]] = lay['feat.ref_pos'][ok]
    batch = dict(batch, ref_pos=jnp.asarray(rp))
    # OpenDDE samples on its STRUCTURAL tokens (see run_case): the reference draw and every injected sampler draw
    # go into that layout through residue_atom_gather; its output comes back in residue layout
    gather = batch.get('structbook/residue_atom_gather')
    to_struct = lambda x: x
    if gather is not None:
      gather = np.asarray(gather)
      gv = gather >= 0
      s_shape = np.shape(batch['struct/pred_dense_atom_mask'])

      def to_struct(x):
        out = np.zeros((s_shape[0] * s_shape[1],) + x.shape[2:], x.dtype)
        out[gather[gv]] = x[gv]
        return out.reshape(s_shape + x.shape[2:])
      batch = dict(batch, **{'struct/ref_pos': jnp.asarray(to_struct(rp))})
    rng = np.random.default_rng(int(os.environ.get('INJECT_SEED', 0)))
    n_v = len(ok)
    x0 = rng.normal(size=(n_v, 3)).astype(np.float32)
    rot = _rotations(rng, TRAJ_MAX_STEPS)
    trans = rng.normal(size=(TRAJ_MAX_STEPS, 3)).astype(np.float32)
    eps = rng.normal(size=(TRAJ_MAX_STEPS, n_v, 3)).astype(np.float32)
    inp = os.path.join(tmp, 'traj.npz')
    np.savez(inp, traj_x0=x0, traj_rot=rot, traj_trans=trans, traj_eps=eps,
             recycling_steps=recycles, precision='fp32')
    _vendor(model, 'trajectory', bdir, inp, os.path.join(tmp, 'b.npz'))
    b = dict(np.load(os.path.join(tmp, 'b.npz')))
    floor = None
    if os.environ.get('TRAJ_FLOOR'):
      # the vendor against ITSELF with the initial positions nudged by 1e-4 (in
      # units of the unit-normal draw): how far a 200-step chaotic sampler
      # carries a difference the size of a port's per-step error. Our distance
      # from the vendor is read against that.
      # TRAJ_FLOOR=trunk instead nudges the vendor's TRUNK outputs by 1e-5 relative (TRAJ_TRUNK_NUDGE, in the runners
      # that support it), the size of a port's trunk difference, with x0 as given: how far the sampler alone carries
      # that, raw and as a rigid rotation.
      trunk = os.environ.get('TRAJ_FLOOR') == 'trunk'
      fz = os.path.join(tmp, 'traj_floor.npz')
      np.savez(fz, traj_x0=x0 if trunk else
               x0 + 1e-4 * np.random.default_rng(7).normal(size=x0.shape).astype(np.float32),
               traj_rot=rot, traj_trans=trans, traj_eps=eps, recycling_steps=recycles, precision='fp32')
      if trunk:
        # TRAJ_FLOOR_EPS sizes the nudge to match the trunk difference actually measured
        os.environ['TRAJ_TRUNK_NUDGE'] = os.environ.get('TRAJ_FLOOR_EPS', '1e-5')
      try:
        _vendor(model, 'trajectory', bdir, fz, os.path.join(tmp, 'b_floor.npz'))
      finally:
        os.environ.pop('TRAJ_TRUNK_NUDGE', None)
      floor = np.asarray(np.load(os.path.join(tmp, 'b_floor.npz'))['final']).reshape(-1, 3)
    if 'pair_state0' in b:
      # the vendor's random initial pair state (ESMFold2), injected as run_case does
      batch = dict(batch, **{'inject/pair_state0': jnp.asarray(_pad_pair(b['pair_state0'], batch))})
  # ours: the real sampler, XLA path, one sample, the same draws in our layout
  ev = cfg.heads.diffusion.eval
  ev.num_samples = 1
  levels = np.asarray(dh.noise_schedule_for(ev, cfg.global_config.model == 'chai1'))
  steps = len(levels) - 1
  dense = np.asarray(raw['pred_dense_atom_mask']).shape

  def to_ours(v):
    out = np.zeros(dense + (3,), np.float32)
    out[idx[ok, 0], idx[ok, 1]] = v[ok]
    return out
  dh.TRAJ_INJECT = dict(x0=to_struct(to_ours(x0)), rot=rot[:steps], trans=trans[:steps],
                        eps=np.stack([to_struct(to_ours(e)) for e in eps[:steps]]))
  dh.SAMPLER_OVERRIDE = None
  try:
    ours = forward_ours(batch, cfg, model_dir, None, None, None, real_sampler=True)
  finally:
    dh.TRAJ_INJECT = None
    os.environ.pop('ESMC_HIDDEN', None)
  n = int(lay['token_pad'].sum())
  vs = np.asarray(b['sigmas'], np.float64)
  rows.append(('traj', 'steps ours %d vendor %d' % (steps, int(b['steps'])),
               'PARITY' if steps == int(b['steps']) else 'BAD'))
  if len(vs) == len(levels):
    c, ratio = P.compare(levels, vs)
    rows.append(('traj', 'sigma schedule max|d| %.2e' % np.abs(levels - vs).max(), P.grade(c, ratio)))
  else:
    rows.append(('traj', 'sigma schedule length ours %d vendor %d' % (len(levels), len(vs)), 'BAD'))
  fin_o = np.asarray(ours['diffusion_samples/atom_positions'])[0][idx[ok, 0], idx[ok, 1]]
  fin_v = np.asarray(b['final']).reshape(-1, 3)[ok]
  if os.environ.get('TRAJ_DUMP'):
    # both sides' final coordinates (vendor atom order), and the vendor's floor run, for a per-atom look
    np.savez(os.environ['TRAJ_DUMP'], ours=fin_o, vendor=fin_v, floor=floor[ok] if floor is not None else fin_v,
             **{k: np.asarray(v)[ok] for k, v in lay.items()
                if np.asarray(v).ndim and np.asarray(v).shape[0] == ok.shape[0] and np.asarray(v).dtype.kind != 'O'})
  c, ratio = P.compare(fin_o, fin_v)
  rows.append(('traj', 'final coordinates corr %.6f max|d|/rms %.2e (max |d| %.3f A)'
               % (c, ratio, np.abs(fin_o - fin_v).max()), P.grade(c, ratio)))
  # the same comparison after superposition: a sampler that is only approximately rotation-equivariant can carry
  # per-step numeric differences into a rigid rotation of the whole sample (protenix1: 0.9 deg, 0.22 A raw, 0.007 A
  # superposed), which says nothing about the structure itself
  a = fin_o.astype(np.float64) - fin_o.mean(0)
  v = fin_v.astype(np.float64) - fin_v.mean(0)
  u, _, vt = np.linalg.svd(a.T @ v)
  rot_k = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
  rows.append(('traj', 'superposed rms %.4f A (raw rms %.4f A; rigid rotation %.3f deg)'
               % (np.sqrt((((a @ rot_k) - v) ** 2).sum(-1).mean()), np.sqrt(((fin_o - fin_v) ** 2).sum(-1).mean()),
                  np.degrees(np.arccos(np.clip((np.trace(rot_k) - 1) / 2, -1, 1)))), 'INFO'))
  if floor is not None:
    f = floor[ok].astype(np.float64) - floor[ok].mean(0)
    u, _, vt = np.linalg.svd(f.T @ v)
    rot_f = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    rows.append(('traj', 'vendor floor (%s) superposed rms %.4f A; rigid rotation %.3f deg'
                 % ('trunk nudged ' + os.environ.get('TRAJ_FLOOR_EPS', '1e-5') if os.environ.get('TRAJ_FLOOR') == 'trunk' else 'x0 nudged 1e-4',
                    np.sqrt((((f @ rot_f) - v) ** 2).sum(-1).mean()),
                    np.degrees(np.arccos(np.clip((np.trace(rot_f) - 1) / 2, -1, 1)))), 'INFO'))
    d_floor = np.abs(floor[ok] - fin_v).max()
    d_ours = np.abs(fin_o - fin_v).max()
    rows.append(('traj', 'vendor vs itself, %s: max |d| %.3f A; ours %.3f A (x%.2f)'
                 % ('trunk nudged ' + os.environ.get('TRAJ_FLOOR_EPS', '1e-5') if os.environ.get('TRAJ_FLOOR') == 'trunk' else 'x0 nudged 1e-4',
                    d_floor, d_ours, d_ours / max(d_floor, 1e-6)), 'INFO'))
    # graded against that floor: a final structure no further from the vendor's than the vendor's
    # own nudged run (raw max |d| and superposed rms both) is its sensitivity, not a port difference
    sup_o = np.sqrt((((a @ rot_k) - v) ** 2).sum(-1).mean())
    sup_f = np.sqrt((((f @ rot_f) - v) ** 2).sum(-1).mean())
    if d_ours <= d_floor and sup_o <= sup_f:
      rows = [(st, m, 'FLOOR' if m.startswith('final coordinates') and g not in ('PARITY', 'CLOSE') else g)
              for st, m, g in rows]
  if 'plddt' in b:
    pl_b = b['plddt'].reshape(-1)
    if pl_b.shape[0] == n_v:
      pl_o = np.asarray(ours['predicted_lddt'])[0][idx[ok, 0], idx[ok, 1]]
      pl_b = pl_b[ok] * (100.0 if pl_o.max() > 1.5 else 1.0)
      c, ratio = P.compare(pl_o, pl_b)
      rows.append(('traj', 'plddt on own final corr %.6f max|d|/rms %.2e' % (c, ratio), P.grade(c, ratio)))
  return rows


def _bioemu2_setup(j, recycles, tmp):
  """Both sides' BioEmu2 features. Ours through the shipped path (featurise_ours
  -> model_features), the vendor's SMILES conformer draw handed over."""
  if any(e.get('protein', {}).get('templates') for e in j['sequences']):
    raise VendorNA('bioemu2 has no template input')
  from alphafold3.model.pipeline import bioemu2_features as bf
  cj = os.path.join(tmp, 'case.json')
  with open(cj, 'w') as f:
    json.dump(j, f)
  bdir = os.path.join(tmp, 'vendor')
  _vendor('bioemu2', 'featurise', cj, bdir)
  ref = dict(np.load(os.path.join(bdir, 'feats.npz')))
  bf.SMILES_TEMPLATES.clear()
  for k in ref:
    if k.startswith('smiles.') and k.endswith('.coords'):
      cid = k.split('.')[1]
      bf.SMILES_TEMPLATES[cid] = bf.Template(
          names=list(ref['smiles.%s.names' % cid]), element=ref['smiles.%s.element' % cid],
          charge=ref['smiles.%s.charge' % cid], coords=ref[k], bonds=ref['smiles.%s.bonds' % cid])
  try:
    batch, raw, cfg, model_dir = featurise_ours('bioemu2', j, recycles, 'none')
  finally:
    bf.SMILES_TEMPLATES.clear()
  return bdir, ref, batch, cfg, model_dir


def _bioemu2_feature_rows(ref, batch):
  """Every array of the vendor's feature dict against ours, on the vendor's
  (unpadded) extent -- exact, not graded on a scale: a feature is equal or it
  is a bug."""
  rows, bad = [], []
  t = ref['res_type'].shape[0]
  n = ref['template_coords'].shape[0]
  for k, r in sorted(ref.items()):
    if k.startswith('smiles.') or k in ('coords', 'num_tokens', 'num_nodes', 'num_residues', 'bond_index',
                                        'bond_pad_mask', 'template_bond_index', 'bond_type', 'residue_idx',
                                        'residue_type', 'mol_type', 'distogram_index'):
      continue
    if 'bioemu2/' + k not in batch:
      bad.append('missing ' + k)
      continue
    a = np.asarray(batch['bioemu2/' + k])
    sl = tuple(slice(0, m) for m in r.shape)
    if any(x < y for x, y in zip(a.shape, r.shape)) or a.ndim != r.ndim:
      bad.append('%s shape %s vs %s' % (k, a.shape, r.shape))
      continue
    d = np.abs(a[sl].astype(np.float64) - r.astype(np.float64))
    if d.size and d.max() > 1e-6:
      bad.append('%s max|d| %.3g' % (k, d.max()))
  rows.append(('feat', '%d vendor feature arrays (%d tokens, %d atoms): %s'
               % (len([k for k in ref if not k.startswith('smiles.')]), t, n,
                  'all equal' if not bad else '; '.join(bad[:6])), 'PARITY' if not bad else 'BAD'))
  return rows


def _bioemu2_ours(batch, cfg, model_dir, noisy, sigma):
  """Our trunk through Model._bioemu2 (structure=False), then one denoise step
  on the same features."""
  import haiku as hk
  import jax
  import jax.numpy as jnp
  from alphafold3.model import model as af3_model
  from alphafold3.model import params as afp

  class M(af3_model.Model):
    def __call__(self, b):
      out = super().__call__(b, structure=False)
      f = {k[len('bioemu2/'):]: v for k, v in b.items() if k.startswith('bioemu2/')}
      for k in ('atom_pad_mask', 'token_pad_mask', 'atom_mask', 'is_dynamics_conditioned'):
        f[k] = jnp.asarray(f[k]).astype(bool)
      e = out['bioemu2']
      ctx = self.bioemu2.static_context(f, e['single'], e['pair'])
      den = self.bioemu2.denoise(f, ctx, e['s_inputs'], e['single'], noisy, sigma)
      return {**e, 'pair_cond': ctx['z'], 'denoised': den}
  fwd = hk.transform(lambda b: M(cfg)(b))
  params = afp.get_model_haiku_params(model_dir=model_dir)
  return jax.device_get(jax.jit(fwd.apply)(params, jax.random.PRNGKey(0), batch))


def bioemu2_case(name, j, recycles):
  rows = []
  with tempfile.TemporaryDirectory() as tmp:
    bdir, ref, batch, cfg, model_dir = _bioemu2_setup(j, recycles, tmp)
    rows += _bioemu2_feature_rows(ref, batch)
    n_v, t = ref['template_coords'].shape[0], ref['res_type'].shape[0]
    real = ref['atom_pad_mask'].astype(bool)
    rng = np.random.default_rng(int(os.environ.get('INJECT_SEED', 0)))
    sigma = float(os.environ.get('INJECT_SIGMA', 30.0))
    noisy = (rng.normal(size=(n_v, 3)) * np.sqrt(sigma ** 2 + 16.0 ** 2)).astype(np.float32)
    noisy[~real] = 0
    inp = os.path.join(tmp, 'in.npz')
    np.savez(inp, positions_noisy=noisy, noise_level=sigma, recycling_steps=recycles)
    _vendor('bioemu2', 'forward', bdir, inp, os.path.join(tmp, 'v.npz'))
    v = dict(np.load(os.path.join(tmp, 'v.npz')))
    n_o = np.asarray(batch['bioemu2/template_coords']).shape[0]
    noisy_o = np.zeros((n_o, 3), np.float32)
    noisy_o[:n_v] = noisy
    o = _bioemu2_ours(batch, cfg, model_dir, noisy_o, np.float32(sigma))
  pairs = [('s_inputs', o['s_inputs'][:t], v['s_inputs']),
           ('trunk single', o['single'][:t], v['s']),
           ('trunk pair', o['pair'][:t, :t], v['z']),
           ('distogram logits', o['distogram'][:t, :t], v['distogram_logits']),
           ('pair conditioning', o['pair_cond'][:t, :t], v['pair_cond']),
           ('denoised (sigma %g)' % sigma, o['denoised'][:n_v][real], v['denoised'][real])]
  for label, a, b in pairs:
    c, ratio = P.compare(np.asarray(a), np.asarray(b))
    rows.append(('model', '%s corr %.6f max|d|/rms %.2e' % (label, c, ratio), P.grade(c, ratio)))
  return rows


def bioemu2_trajectory(name, j, recycles):
  """The full sampler on both sides, every draw injected (see trajectory_case).

  BIOEMU2_PROPAGATE_NS=<lag> grades PROPAGATION instead: stock BioEmu2 first
  samples an equilibrium structure E (draws INJECT_SEED + 1), then both sides
  propagate from E by the lag on the same draws, each endpoint superposed on E."""
  from alphafold3.model.network import diffusion_head as dh
  rows = []
  lag = os.environ.get('BIOEMU2_PROPAGATE_NS')
  if lag:
    os.environ['BIOEMU2_MODE'] = 'equilibrium'
  try:
    return _bioemu2_trajectory(name, j, recycles, lag, rows, dh)
  finally:
    for k in ('BIOEMU2_X0', 'BIOEMU2_LAG_NS'):
      os.environ.pop(k, None)
    if lag:
      os.environ.pop('BIOEMU2_MODE', None)


def _bioemu2_draws(seed, n_v):
  rng = np.random.default_rng(seed)
  x0 = rng.normal(size=(n_v, 3)).astype(np.float32)
  rot = _rotations(rng, TRAJ_MAX_STEPS)
  trans = rng.normal(size=(TRAJ_MAX_STEPS, 3)).astype(np.float32)
  eps = rng.normal(size=(TRAJ_MAX_STEPS, n_v, 3)).astype(np.float32)
  return x0, rot, trans, eps


def _bioemu2_trajectory(name, j, recycles, lag, rows, dh):
  with tempfile.TemporaryDirectory() as tmp:
    bdir, ref, batch, cfg, model_dir = _bioemu2_setup(j, recycles, tmp)
    # graded here, before propagation adds the conditioning frame that stock BioEmu2
    # only sets at sampling time
    rows += _bioemu2_feature_rows(ref, batch)
    extra = {}
    if lag:
      n_v = ref['template_coords'].shape[0]
      real = ref['atom_pad_mask'].astype(bool)
      x0, rot, trans, eps = _bioemu2_draws(int(os.environ.get('INJECT_SEED', 0)) + 1, n_v)
      inp = os.path.join(tmp, 'start.npz')
      np.savez(inp, traj_x0=x0, traj_rot=rot, traj_trans=trans, traj_eps=eps, recycling_steps=recycles)
      _vendor('bioemu2', 'trajectory', bdir, inp, os.path.join(tmp, 'start_out.npz'))
      start = np.asarray(np.load(os.path.join(tmp, 'start_out.npz'))['final']).reshape(-1, 3)[real]
      np.save(os.path.join(tmp, 'x0.npy'), start)
      os.environ['BIOEMU2_X0'] = os.path.join(tmp, 'x0.npy')
      os.environ['BIOEMU2_LAG_NS'] = lag
      batch, raw, cfg, model_dir = featurise_ours('bioemu2', j, recycles, 'none')
      extra = dict(x0_frame=start, lag_ns=float(lag))
      rows.append(('traj', 'propagation by %s ns from a stock equilibrium sample' % lag, 'INFO'))
    n_v = ref['template_coords'].shape[0]
    real = ref['atom_pad_mask'].astype(bool)
    x0, rot, trans, eps = _bioemu2_draws(int(os.environ.get('INJECT_SEED', 0)), n_v)
    inp = os.path.join(tmp, 'traj.npz')
    np.savez(inp, traj_x0=x0, traj_rot=rot, traj_trans=trans, traj_eps=eps, recycling_steps=recycles,
             **extra)
    _vendor('bioemu2', 'trajectory', bdir, inp, os.path.join(tmp, 'b.npz'))
    b = dict(np.load(os.path.join(tmp, 'b.npz')))
    ev = cfg.heads.diffusion.eval
    ev.num_samples = 1
    levels = np.asarray(dh.noise_schedule_for(ev, False))
    steps = len(levels) - 1
    n_o = np.asarray(batch['bioemu2/template_coords']).shape[0]

    def to_ours(x):
      out = np.zeros((n_o, 1, 3), np.float32)
      out[:n_v, 0] = x
      return out
    dh.TRAJ_INJECT = dict(x0=to_ours(x0), rot=rot[:steps], trans=trans[:steps],
                          eps=np.stack([to_ours(e) for e in eps[:steps]]))
    dh.SAMPLER_OVERRIDE = None
    try:
      ours = forward_ours(batch, cfg, model_dir, None, None, None, real_sampler=True)
    finally:
      dh.TRAJ_INJECT = None
  rows.append(('traj', 'steps ours %d vendor %d' % (steps, int(b['steps'])),
               'PARITY' if steps == int(b['steps']) else 'BAD'))
  vs = np.asarray(b['sigmas'], np.float64)
  if len(vs) == len(levels):
    c, ratio = P.compare(levels, vs)
    rows.append(('traj', 'sigma schedule max|d| %.2e' % np.abs(levels - vs).max(), P.grade(c, ratio)))
  else:
    rows.append(('traj', 'sigma schedule length ours %d vendor %d' % (len(levels), len(vs)), 'BAD'))
  fin_o = np.asarray(ours['diffusion_samples/bioemu2_flat'])[0][:n_v][real]
  fin_v = np.asarray(b['final']).reshape(-1, 3)
  fin_v = fin_v[real] if len(fin_v) == len(real) else fin_v     # a propagated endpoint comes back real atoms only
  c, ratio = P.compare(fin_o, fin_v)
  rows.append(('traj', 'final coordinates corr %.6f max|d|/rms %.2e (max |d| %.3f A)'
               % (c, ratio, np.abs(fin_o - fin_v).max()), P.grade(c, ratio)))
  a = fin_o.astype(np.float64) - fin_o.mean(0)
  v = fin_v.astype(np.float64) - fin_v.mean(0)
  u, _, vt = np.linalg.svd(a.T @ v)
  rot_k = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
  rows.append(('traj', 'superposed rms %.4f A (raw rms %.4f A; rigid rotation %.3f deg)'
               % (np.sqrt((((a @ rot_k) - v) ** 2).sum(-1).mean()), np.sqrt(((fin_o - fin_v) ** 2).sum(-1).mean()),
                  np.degrees(np.arccos(np.clip((np.trace(rot_k) - 1) / 2, -1, 1)))), 'INFO'))
  return rows


def bioemu2_steer(name, j, recycles):
  """A steering preset (BIOEMU2_SAMPLER=edm_gd|edm_fk|edm_fkc) on both sides over
  STEER_PARTICLES particles, every draw injected, the vendor's RDKit ligand tables
  handed to ours; graded on each particle's final coordinates."""
  from alphafold3.model.network import bioemu2_steering as ns
  from alphafold3.model.pipeline import bioemu2_features as bf
  preset = os.environ['BIOEMU2_SAMPLER']
  n_part = int(os.environ.get('STEER_PARTICLES', 4))
  if os.environ.get('BIOEMU2_PROPAGATE_NS'):
    os.environ['BIOEMU2_MODE'] = 'equilibrium'      # propagation embeds in equilibrium mode
  steps = 200
  rows = []
  with tempfile.TemporaryDirectory() as tmp:
    os.environ.pop('BIOEMU2_SAMPLER')          # featurise the vendor side plainly
    try:
      bdir, ref, batch, cfg, model_dir = _bioemu2_setup(j, recycles, tmp)
    finally:
      os.environ['BIOEMU2_SAMPLER'] = preset
    n_v = ref['template_coords'].shape[0]
    real = ref['atom_pad_mask'].astype(bool)
    extra = {}
    lag = os.environ.get('BIOEMU2_PROPAGATE_NS')
    if lag:
      # steered PROPAGATION: from a stock equilibrium sample, every particle
      sx0, srot, strans, seps = _bioemu2_draws(int(os.environ.get('INJECT_SEED', 0)) + 1, n_v)
      sin = os.path.join(tmp, 'start.npz')
      np.savez(sin, traj_x0=sx0, traj_rot=srot, traj_trans=strans, traj_eps=seps, recycling_steps=recycles)
      _vendor('bioemu2', 'trajectory', bdir, sin, os.path.join(tmp, 'start_out.npz'))
      start = np.asarray(np.load(os.path.join(tmp, 'start_out.npz'))['final']).reshape(-1, 3)[real]
      np.save(os.path.join(tmp, 'x0.npy'), start)
      os.environ['BIOEMU2_X0'] = os.path.join(tmp, 'x0.npy')
      os.environ['BIOEMU2_LAG_NS'] = lag
      extra = dict(x0_frame=start, lag_ns=float(lag))
      rows.append(('traj', 'steered propagation by %s ns from a stock equilibrium sample' % lag, 'INFO'))
    rng = np.random.default_rng(int(os.environ.get('INJECT_SEED', 0)))
    x0 = rng.normal(size=(n_part, n_v, 3)).astype(np.float32)
    rot = np.stack([_rotations(rng, n_part) for _ in range(steps)])
    trans = rng.normal(size=(steps, n_part, 3)).astype(np.float32)
    eps = rng.normal(size=(steps, n_part, n_v, 3)).astype(np.float32)
    uni = rng.uniform(size=(steps, 1)).astype(np.float32)
    inp = os.path.join(tmp, 'steer.npz')
    np.savez(inp, x0=x0, rot=rot, trans=trans, eps=eps, uniform=uni, recycling_steps=recycles, **extra)
    _vendor('bioemu2', 'steer', bdir, inp, os.path.join(tmp, 'b.npz'))
    b = dict(np.load(os.path.join(tmp, 'b.npz')))
    floor = None
    if os.environ.get('STEER_FLOOR'):
      # the vendor against ITSELF with x0 nudged (1e-5 A, or STEER_FLOOR_EPS): how far the
      # steered sampler alone carries a difference of a port's size (its hinge potentials
      # have step gradients, and FK resampling at a near-tie picks another lineage)
      nudge = float(os.environ.get('STEER_FLOOR_EPS', 1e-5))
      fz = os.path.join(tmp, 'steer_floor.npz')
      np.savez(fz, x0=x0 + nudge * np.random.default_rng(7).normal(size=x0.shape).astype(np.float32),
               rot=rot, trans=trans, eps=eps, uniform=uni, recycling_steps=recycles)
      _vendor('bioemu2', 'steer', bdir, fz, os.path.join(tmp, 'b_floor.npz'))
      floor = np.asarray(np.load(os.path.join(tmp, 'b_floor.npz'))['final'])
    bf.STEERING_LIGAND_TABLES.clear()
    bf.STEERING_LIGAND_TABLES.update({k[4:]: v for k, v in b.items() if k.startswith('lig.')})
    try:
      batch, raw, cfg, model_dir = featurise_ours('bioemu2', j, recycles, 'none')
    finally:
      bf.STEERING_LIGAND_TABLES.clear()
    n_o = np.asarray(batch['bioemu2/template_coords']).shape[0]

    def pad(x):
      out = np.zeros(x.shape[:-2] + (n_o, 3), np.float32)
      out[..., :n_v, :] = x
      return out
    ns.INJECT = dict(x0=pad(x0), rot=rot, trans=trans, eps=pad(eps), uniform=uni)
    cfg.heads.diffusion.eval.num_samples = n_part
    trace = os.environ.get('BIOEMU2_FK_TRACE')
    if trace:
      ns.TRACE = []
    try:
      ours = forward_ours(batch, cfg, model_dir, None, None, None, real_sampler=True)
    finally:
      ns.INJECT = None
      if trace:
        rows_ = [r for r in ns.TRACE if bool(r[1])]
        np.savez(trace + '.ours.npz', step=np.array([r[0] for r in rows_]),
                 w=np.stack([r[2] for r in rows_]), idx=np.stack([r[3] for r in rows_]),
                 energy=np.stack([r[4] for r in rows_]))
        ns.TRACE = None
      for k in ('BIOEMU2_X0', 'BIOEMU2_LAG_NS') + (('BIOEMU2_MODE',) if lag else ()):
        os.environ.pop(k, None)
  fin_o = np.asarray(ours['diffusion_samples/bioemu2_flat'])[:, :n_v][:, real]
  fin_v = np.asarray(b['final'])
  fin_v = fin_v[:, :n_v][:, real] if fin_v.shape[1] == len(real) else fin_v   # a propagated endpoint: real atoms
  rows.append(('traj', '%s: %d particles, vendor %d steps' % (preset, n_part, int(b['steps'])), 'INFO'))
  for i in range(n_part):
    c, ratio = P.compare(fin_o[i], fin_v[i])
    a = fin_o[i].astype(np.float64) - fin_o[i].mean(0)
    v = fin_v[i].astype(np.float64) - fin_v[i].mean(0)
    u, _, vt = np.linalg.svd(a.T @ v)
    rk = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    rows.append(('traj', 'particle %d final corr %.6f max|d|/rms %.2e (max |d| %.3f A; superposed rms %.4f A)'
                 % (i, c, ratio, np.abs(fin_o[i] - fin_v[i]).max(),
                    np.sqrt((((a @ rk) - v) ** 2).sum(-1).mean())), P.grade(c, ratio)))
    if floor is not None:
      f = floor[i, :n_v][real].astype(np.float64)
      f = f - f.mean(0)
      u, _, vt = np.linalg.svd(f.T @ v)
      rf = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
      rows.append(('traj', 'particle %d vendor vs itself (x0 nudged %g): superposed rms %.4f A'
                   % (i, float(os.environ.get('STEER_FLOOR_EPS', 1e-5)), np.sqrt((((f @ rf) - v) ** 2).sum(-1).mean())), 'INFO'))
  return rows


def run_case(model, name, j, recycles, bf16, graded=False):
  if model == 'bioemu2':
    return bioemu2_case(name, j, recycles)
  rows = []
  with tempfile.TemporaryDirectory() as tmp:
    cj = os.path.join(tmp, 'case.json')
    with open(cj, 'w') as f:
      json.dump(j, f)
    bdir = os.path.join(tmp, 'vendor')
    _vendor(model, 'featurise', cj, bdir)
    lay = dict(np.load(os.path.join(bdir, 'layout.npz')))
    # ESMFold2's language model is its own model with its own gate: ours folds
    # from the hidden states the vendor's ESM-C tower produced (fold_check)
    hidden = os.path.join(bdir, 'esmc_hidden.npz')
    if os.path.exists(hidden):
      os.environ['ESMC_HIDDEN'] = hidden
      rows += _tower_rows(model, j, hidden)
    rows += _chai_esm(model, j, bdir)
    batch, raw, cfg, model_dir = featurise_ours(model, j, recycles, bf16)
    idx, arows = align(raw, lay)
    rows += arows
    ok = idx[:, 0] >= 0

    # REF_POS IS A RANDOM DRAW ON BOTH SIDES, so it is injected like the
    # coordinates: boltz picks one of the residue's stored conformers at random
    # and applies a random roto-translation per ref_space_uid group
    # (featurizerv2.py:1494); ours takes the CCD ideal geometry, centred. Our
    # batch gets boltz's exact draw. What the two DISTRIBUTIONS differ in is
    # reported, not hidden: the largest within-group distance difference.
    import jax.numpy as jnp
    b_ref = lay['feat.ref_pos']
    rp = np.array(batch['ref_pos'])
    o_ref = rp[idx[ok, 0], idx[ok, 1]]
    uid = lay['feat.ref_space_uid'][ok]
    geo = max((np.abs(np.linalg.norm(o_ref[uid == u][:, None] - o_ref[uid == u][None], axis=-1)
                      - np.linalg.norm(b_ref[ok][uid == u][:, None] - b_ref[ok][uid == u][None], axis=-1)).max()
               for u in np.unique(uid)), default=0.0)
    rows.append(('feat', 'conformer geometry (ours CCD ideal vs boltz stored '
                 'draw): max within-residue distance diff %.2f A' % geo, 'INFO'))
    rp[idx[ok, 0], idx[ok, 1]] = b_ref[ok]
    rows += _inject_ref_only(rp, raw, lay)
    batch = dict(batch, ref_pos=jnp.asarray(rp))
    # OPENDDE DIFFUSES ON STRUCTURAL TOKENS: our batch carries a second,
    # structural-token copy of the atom features under 'struct/' (the diffusion
    # module's atom encoder reads struct/ref_pos), and the sampler's positions
    # and the denoiser's output live in that (struct token, slot) layout. The
    # atoms are the same ones; residue_atom_gather[r, j] is the flat struct
    # index of residue atom (r, j). So the injected draw goes into both copies,
    # and positions cross between layouts through the gather -- a bijection on
    # real atoms, checked here.
    gather = batch.get('structbook/residue_atom_gather')
    if gather is not None:
      gather = np.asarray(gather)
      gv = gather >= 0
      if not (np.array_equal(gv, np.asarray(raw['pred_dense_atom_mask']).astype(bool))
              and len(np.unique(gather[gv])) == gv.sum()):
        rows.append(('align', 'structural gather is not a bijection on real '
                     'atoms', 'BAD'))
      s_shape = np.shape(batch['struct/pred_dense_atom_mask'])

      def to_struct(x):
        out = np.zeros((s_shape[0] * s_shape[1],) + x.shape[2:], x.dtype)
        out[gather[gv]] = x[gv]
        return out.reshape(s_shape + x.shape[2:])
      batch = dict(batch, **{'struct/ref_pos': jnp.asarray(to_struct(rp))})

    rng = np.random.default_rng(int(os.environ.get('INJECT_SEED', 0)))
    dmask = np.asarray(raw['pred_dense_atom_mask']).astype(np.float32)[..., None]
    # INJECT_POS_SCALE (default 8 A): 0 puts every injected atom at the origin, so a
    # confidence head's distance features are constant -- a bisection probe
    scale = float(os.environ.get('INJECT_POS_SCALE', 8))
    pos = (rng.normal(size=dmask.shape[:2] + (3,)) * scale * dmask).astype(np.float32)
    sigma = np.float32(os.environ.get('INJECT_SIGMA', 10.0))
    noisy = (pos + rng.normal(size=pos.shape) * sigma * dmask).astype(np.float32)
    to_b = lambda x: np.where(ok[:, None], x[idx[:, 0], idx[:, 1]], 0
                              ).astype(np.float32)
    inp = os.path.join(tmp, 'in.npz')
    np.savez(inp, positions=to_b(pos), positions_noisy=to_b(noisy),
             noise_level=sigma, recycling_steps=recycles,
             precision='bf16' if bf16 != 'none' and not graded else 'fp32')
    _vendor(model, 'forward', bdir, inp, os.path.join(tmp, 'b.npz'))
    b = dict(np.load(os.path.join(tmp, 'b.npz')))
    b_floor = None
    if os.environ.get('CASE_FLOOR'):
      # the vendor against ITSELF with its initial pair nudged by CASE_FLOOR (relative; runners
      # that honour VENDOR_NUDGE): a row ours misses by no more than this is the vendor's own
      # sensitivity -- helixfold3 rounds its triangle bias to bfloat16, so a float-level
      # difference flips roundings and recycling amplifies them
      # CASE_FLOOR_N nudges (different seeds): one draw of a chaotic spread is not its width
      os.environ['VENDOR_NUDGE'] = os.environ['CASE_FLOOR']
      b_floor = []
      try:
        for seed in range(int(os.environ.get('CASE_FLOOR_N', 3))):
          os.environ['VENDOR_NUDGE_SEED'] = str(7 + seed)
          _vendor(model, 'forward', bdir, inp, os.path.join(tmp, 'b_floor.npz'))
          b_floor.append(dict(np.load(os.path.join(tmp, 'b_floor.npz'))))
      finally:
        os.environ.pop('VENDOR_NUDGE', None)
        os.environ.pop('VENDOR_NUDGE_SEED', None)
    b_half = None
    if graded:
      # the vendor again at its STOCK precision: its own bf16 error against its
      # fp32 is the yardstick for ours (graded as a distribution, see P.HALF_MEAN)
      inp_h = os.path.join(tmp, 'in_half.npz')
      np.savez(inp_h, positions=to_b(pos), positions_noisy=to_b(noisy),
               noise_level=sigma, recycling_steps=recycles, precision='bf16')
      _vendor(model, 'forward', bdir, inp_h, os.path.join(tmp, 'b_half.npz'))
      b_half = dict(np.load(os.path.join(tmp, 'b_half.npz')))
    os.environ.pop('ESMC_HIDDEN', None)
    rows_v = sorted(k for k in b if k.startswith('msa_rows_pass'))
    if rows_v and 'px_msa_rows' in batch:
      # protenix: native's sampled MSA rows per pass against our replay of them
      ours_rows = np.asarray(batch['px_msa_rows'])
      for k in rows_v:
        i = int(k[len('msa_rows_pass'):])
        if i >= ours_rows.shape[0]:
          continue
        v = set(np.asarray(b[k]).reshape(-1).tolist())
        o = set(np.flatnonzero(ours_rows[i]).tolist())
        # graded on the SIZE: native's MSA can hold rows ours deduplicates (its unpaired copy of the query), so
        # the same draw may land on a different but identical row -- the indices are information
        rows.append(('rng', 'MSA rows pass %d: native %d, replay %d, in common %d'
                     % (i, len(v), len(o), len(v & o)), 'INFO' if len(v) == len(o) else 'BAD'))
    if 'conf_aug_rot' in b:
      # helixfold3's confidence head re-encodes a randomly rotated copy of the
      # structure: hand ours the vendor's draw
      from alphafold3.model.network import confidence_head as _ch
      _ch.HF3_CONF_AUGMENT = (b['conf_aug_rot'], b['conf_aug_trans'])
    if 'pair_state0' in b:
      # the vendor's random initial pair state (ESMFold2), injected like ref_pos
      batch = dict(batch, **{'inject/pair_state0': jnp.asarray(_pad_pair(b['pair_state0'], batch))})
    if os.environ.get('B2_KEEP'):
      import shutil
      keep = os.path.join(os.environ['B2_KEEP'], model, name)
      shutil.rmtree(keep, ignore_errors=True)
      shutil.copytree(bdir, keep)
      np.savez(os.path.join(keep, 'boltz_out.npz'), **b)
      np.savez(os.path.join(keep, 'ours_raw.npz'), **{
          k: np.asarray(v) for k, v in raw.items()
          if np.asarray(v).dtype.kind != 'O'})
  if gather is not None:
    ours = forward_ours(batch, cfg, model_dir, to_struct(pos), to_struct(noisy),
                        sigma)
    den = np.asarray(ours['diffusion_samples/denoised'])
    flat = den.reshape(den.shape[0], -1, 3)
    ours['diffusion_samples/denoised'] = np.where(
        gv[None, :, :, None], flat[:, np.where(gv, gather, 0)], 0.0)
  else:
    ours = forward_ours(batch, cfg, model_dir, pos, noisy, sigma)

  n = int(lay['token_pad'].sum())
  # ours may be padded past the vendor's tokens (OURS_BUCKET): compare the real ones
  for key in ('target_feat', 'single_embeddings'):
    if key in ours:
      ours[key] = np.asarray(ours[key])[:n]
  for key in ('pair_embeddings', 'distogram/distogram'):
    if key in ours:
      ours[key] = np.asarray(ours[key])[:n, :n]
  for key in ('full_pae', 'full_pde'):
    if key in ours:     # a model with no PDE head emits no full_pde
      ours[key] = np.asarray(ours[key])[:, :n, :n]
  # boltz2 reports pLDDT per TOKEN (token_level_confidence); ours carries the
  # token's value on each of its atoms, so read it at the first real slot.
  per_atom = b['plddt'].reshape(-1).shape[0] == len(ok)
  if per_atom:
    pl_o = ours['predicted_lddt'][0][idx[ok, 0], idx[ok, 1]]
  else:
    first = np.asarray(raw['pred_dense_atom_mask']).astype(bool).argmax(1)[:n]   # (ours may be bucket-padded)
    pl_o = ours['predicted_lddt'][0][np.arange(n), first]
  pl_scale = 100.0 if pl_o.max() > 1.5 else 1.0

  def vendor_side(bb):
    """The vendor's arrays in pair order, from one vendor run."""
    pl_b = bb['plddt'].reshape(-1)
    return [
        _s_inputs_to_ours(model, bb['s_inputs'][:n]),
        bb['s'][:n] if 's' in bb else None,
        bb['z'][:n, :n],
        bb['distogram_logits'][:n, :n].reshape(n, n, -1) if 'distogram_logits' in bb else None,
        bb['pae'][:n, :n],
        bb['pde'][:n, :n] if 'pde' in bb else None,
        # atom-level outputs, through the alignment
        bb['denoised'].reshape(-1, 3)[ok],
        (pl_b[ok] if per_atom else pl_b[:n]) * pl_scale,
    ]
  labels = ['s_inputs (target_feat)', 'trunk single', 'trunk pair', 'distogram logits', 'pae', 'pde',
            'denoise step', 'plddt']
  den_o = ours['diffusion_samples/denoised'][0][idx[ok, 0], idx[ok, 1]]
  ours_side = [ours['target_feat'], ours['single_embeddings'], ours['pair_embeddings'],
               ours['distogram/distogram'], ours['full_pae'][0],
               ours['full_pde'][0] if 'full_pde' in ours else None, den_o, pl_o]
  pairs = list(zip(labels, ours_side, vendor_side(b)))
  if graded:
    for (label, a, r), rh in zip(pairs, vendor_side(b_half)):
      if a is None or r is None or rh is None or np.shape(a) != np.shape(r):
        continue
      eo, eu = P.rel_rms_err(a, r), P.rel_rms_err(rh, r)
      P.HALF_STATS.setdefault(label, []).append((name, eo, eu, np.size(r)))
      rows.append(('half', '%s: ours bf16 err %.2e, vendor stock err %.2e (x%.2f)'
                   % (label, eo, eu, eo / eu if eu else float('inf')), 'INFO'))
    return rows
  if os.environ.get('B2_KEEP'):
    np.savez(os.path.join(os.environ['B2_KEEP'], model, name, 'ours_out.npz'),
             **{k: np.asarray(v) for k, v in ours.items()})
  floors = {}
  for bf in (b_floor or []):
    for label, f in zip(labels, vendor_side(bf)):
      floors.setdefault(label, []).append(f)
  for label, a, r in pairs:
    if r is None:   # the stock vendor has no such output (chai: no distogram)
      rows.append(('model', '%s: not produced by the stock vendor' % label, 'INFO'))
      continue
    if a is None:   # nor do we (a model with no such head)
      rows.append(('model', '%s: not produced by ours' % label, 'BAD'))
      continue
    a, r = np.asarray(a, np.float32), np.asarray(r, np.float32)
    if a.shape != r.shape:
      rows.append(('model', '%s shape ours %s boltz %s' % (label, a.shape,
                                                          r.shape), 'BAD'))
      continue
    c, ratio = P.compare(a, r)
    g = P.grade(c, ratio)
    fs = [x for x in floors.get(label, []) if x is not None]
    if fs:
      cmps = [P.compare(np.asarray(x, np.float32), r) for x in fs]
      cf, rf = min(c_ for c_, _ in cmps), max(r_ for _, r_ in cmps)
      rows.append(('floor', '%s: vendor vs itself (pair nudged %s, %d draws) worst corr %.6f max|d|/rms %.2e'
                   % (label, os.environ['CASE_FLOOR'], len(fs), cf, rf), 'INFO'))
      if g not in ('PARITY', 'CLOSE') and c >= cf and ratio <= rf:
        g = 'FLOOR'
    rows.append(('model', '%s corr %.6f max|d|/rms %.2e' % (label, c, ratio), g))
  return rows


def main(argv):
  rec = 1
  if '--recycles' in argv:
    rec = int(argv[argv.index('--recycles') + 1])
  # --bf16 MODE: ours in bfloat16 ('all' or 'intermediate'); the vendor side
  # stays as it is. For measuring a vendor whose precision cannot be changed
  # (chai's exported modules are bf16 inside) against our own bf16 floor.
  bf16 = argv[argv.index('--bf16') + 1] if '--bf16' in argv else 'none'
  # --bf16-graded MODE: ours in bfloat16 MODE against the vendor at fp32, graded
  # against the vendor's OWN stock-precision error as a distribution over cases
  graded = '--bf16-graded' in argv
  trajectory = '--trajectory' in argv
  if graded:
    bf16 = argv[argv.index('--bf16-graded') + 1]
  pos = [a for i, a in enumerate(argv) if not a.startswith('--') and not a.isdigit()
         and not (i and argv[i - 1] in ('--bf16', '--bf16-graded'))]
  model, names = pos[0], pos[1:]
  if model not in VENDORS:
    raise SystemExit('no vendor runner for %r; have %s' % (model, sorted(VENDORS)))
  allc = P.cases()
  # a head-to-tail cyclic 13-mer (an arbitrary sequence): a harness case, not
  # AF3's -- AF3 has no cyclic input, so af3_native_parity lacks it
  allc['cyclic'] = {'name': 'cyclic', 'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 3,
                    'sequences': [{'protein': {'id': 'A', 'sequence': 'GKRTWPLAVYNDG', 'cyclic': True,
                                               'modifications': [], 'unpairedMsa': '', 'pairedMsa': '',
                                               'templates': []}}]}
  # the same entities in an order a vendor may rearrange (helixfold3 puts protein, RNA, DNA,
  # ligand): a ligand first, DNA before protein. A harness case, not AF3's.
  prot = 'GSWKELAAKLAKEAAEKGDPEVAKRALELAKKAAEEGDPELAKKAAELAKKA'
  allc['reorder'] = {'name': 'reorder', 'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 3,
                     'sequences': [{'ligand': {'id': 'A', 'ccdCodes': ['ATP']}},
                                   {'dna': {'id': 'B', 'sequence': 'GCGATCGCAT', 'modifications': []}},
                                   {'protein': {'id': 'C', 'sequence': prot, 'modifications': [],
                                                'unpairedMsa': '>q\n%s\n' % prot, 'pairedMsa': '', 'templates': []}},
                                   {'rna': {'id': 'D', 'sequence': 'GCGGAUUUAGCUCAG', 'modifications': [],
                                            'unpairedMsa': ''}}]}
  # pocket/contact conditioning (the input's `constraints`): harness cases, read only by the vendors trained on them
  # (CONSTRAINT_VENDORS). A pocket for the SMILES ligand (chai takes no CCD ligand); a pocket on ATP plus a contact
  # to one of its atoms (boltz2's ligand-atom token); residue contacts across a heterodimer.
  pocket = [{'pocket': {'binder': 'B', 'contacts': [['A', 9], ['A', 12], ['A', 13], ['A', 16], ['A', 30]],
                        'maxDistance': 6.0}}]
  allc['pocket'] = dict(allc['ligand_smiles'], name='pocket', constraints=pocket)
  allc['pocket_contact'] = dict(allc['ligand_ccd'], name='pocket_contact', constraints=pocket + [
      {'contact': {'token1': ['A', 41], 'token2': ['B', 1, 'PG'], 'maxDistance': 8.0}}])
  allc['contact_ligand'] = dict(allc['ligand_ccd'], name='contact_ligand', constraints=[
      {'contact': {'token1': ['A', 41], 'token2': ['B', 1, 'PG'], 'maxDistance': 8.0}}])
  allc['contact'] = dict(allc['heterodimer'], name='contact', constraints=[
      {'contact': {'token1': ['A', 10], 'token2': ['B', 44], 'maxDistance': 8.0}},
      {'contact': {'token1': ['A', 30], 'token2': ['B', 8], 'maxDistance': 12.0}}])
  # a case may also be a fold-input JSON file, named by its path
  for n in names or ():
    if n.endswith('.json') and n not in allc:
      allc[n] = json.load(open(n))
  names = names or list(allc)
  failed = []
  for n in names:
    print('== %s %s' % (model, n), flush=True)
    try:
      if _cyclic_chains(allc[n]) and model not in CYCLIC_VENDORS:
        raise VendorNA(_CYCLIC_IGNORED.get(model, '%s takes no cyclic polymer input' % model))
      kinds = {k for c in allc[n].get('constraints') or () for k in c}
      if kinds - CONSTRAINT_VENDORS.get(model, set()):
        raise VendorNA('%s was not trained with %s conditioning' % (model, '/'.join(sorted(kinds))))
      rows = (trajectory_case(model, n, allc[n], rec) if trajectory
              else run_case(model, n, allc[n], rec, bf16, graded=graded))
    except VendorNA as e:   # the vendor cannot take this input: N/A, not FAIL
      print('  %s: N/A (%s)' % (n, e), flush=True)
      continue
    except Exception as e:  # a crash is a FAILURE, never a skip
      import traceback
      traceback.print_exc()
      print('  ERROR %s: %s' % (type(e).__name__, str(e).splitlines()[0][:200]))
      failed.append(n)
      continue
    for stage, msg, g in rows:
      print('  %-6s %-7s %s' % (stage, g, msg))
    bad = [r for r in rows if r[2] not in ('PARITY', 'CLOSE', 'INFO', 'FLOOR')]
    print('  %s: %s' % (n, 'PASS' if not bad else 'FAIL (%d rows)' % len(bad)),
          flush=True)
    if bad:
      failed.append(n)
  if graded:
    # bf16 is graded as a DISTRIBUTION over the cases run (af3_native_parity's
    # HALF_MEAN / HALF_MAX against the vendor's own stock-precision error)
    print('== bf16 (%s) graded over %d case(s)' % (bf16, len(names)))
    for stage, msg, g in P.grade_half():
      print('  %-6s %-7s %s' % (stage, g, msg))
      if g not in ('PARITY', 'CLOSE', 'INFO'):
        failed.append('half:' + msg.split(' ')[0])
  print('__VERDICT %s %s' % ('FAIL' if failed else 'PASS', ' '.join(failed)))
  return 1 if failed else 0


if __name__ == '__main__':
  sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
  # the repo root: converters/ (fold_check's parse_ca, rf3's alphabet remap)
  sys.path.insert(0, os.path.join(HERE, '..', '..'))
  argv, sys.argv = sys.argv[1:], sys.argv[:1]
  sys.exit(main(argv))
