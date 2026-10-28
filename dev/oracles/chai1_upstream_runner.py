"""The STOCK chai-1 half of vendor_native_parity.py (chai1).

  python chai1_upstream_runner.py featurise chai1 CASE.json OUT_DIR
  python chai1_upstream_runner.py forward   chai1 OUT_DIR IN.npz OUT.npz

Run in ~/venv_chai (chai_lab 0.6.1 from PyPI, nothing else on the path), with
CHAI_DOWNLOADS_DIR naming the downloaded components. Everything is chai's own,
driven the way `chai_lab.chai1.run_inference` drives it:
make_all_atom_feature_context -> run_folding_on_context, which collates the
features and calls the exported TorchScript components in sequence.

chai ships TorchScript, so nothing INSIDE a component can be hooked; what
crosses BETWEEN them is recorded by wrapping `load_exported`. The sampler is
inline in run_folding_on_context, so instead of overriding it:
  * the diffusion wrapper's first call runs the real module ONCE on the
    injected (noised coordinates, sigma) and records it -- what it returns to
    chai's loop is irrelevant, the loop is cut to one step;
  * the confidence wrapper scores the injected coordinates.

INPUTS. chai's FASTA takes polymers by sequence, a modified residue inline as
`(CCD)`, and a ligand ONLY as SMILES (its residue is always `LIG`): a CCD
ligand goes in as that CCD entry's SMILES, so its atom names are chai's, and
the harness aligns those atoms by token + element. A glycan goes in as chai's
glycan entity, a covalent bond as a restraints-CSV `covalent` row. An MSA is
written as chai's own ColabFold path writes one (colabfold.py): the query as
the paired block, the unpaired rows not already in it, a source per header.
ESM2 embeddings are ON, as chai-1 is run: chai's own traced ESM2 3B (fp16, on
the GPU) embeds each protein chain, and `featurise` writes its rows -- one per
protein residue, in token order -- to esm2_rows.npz, which the harness hands
ours (fold_check's ESM_EMB) so both fold from the same embeddings; our own tower
is graded against those rows separately. CHAI_NO_ESM=1 runs both without
(zeros), the way every chai number before 2026-10-28 was measured.

There is no distogram: stock chai-lab ships no distogram head.
"""
import json
import os
import sys

import numpy as np

# SMILES for the CCD ligands the cases use (chai takes none by code).
_CCD_SMILES = {
    'ATP': 'c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O'
           '[P@](=O)(O)OP(=O)(O)O)O)O)N',
    'MG': '[Mg+2]',
}


class _Stop(Exception):
  pass


def _case_inputs(j, d):
  """AF3 fold-input JSON -> (fasta path, msa dir or None, restraints path or None)."""
  from chai_lab.data.parsing.msas.aligned_pqt import expected_basename
  import pandas as pd
  lines, restraints, msa_dir = [], [], None
  glycan_chains = set()
  for pair in j.get('bondedAtomPairs') or ():
    for a in pair:
      for entry in j['sequences']:
        (kind, e), = entry.items()
        if kind == 'ligand' and a[0] in (e['id'] if isinstance(e['id'], list) else [e['id']]):
          glycan_chains.add(a[0])
  for entry in j['sequences']:
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    for cid in ids:
      if kind in ('protein', 'rna', 'dna'):
        seq = list(e['sequence'])
        for m in e.get('modifications', []):
          pos = m.get('ptmPosition', m.get('basePosition'))
          seq[pos - 1] = '(%s)' % m.get('ptmType', m.get('modificationType'))
        lines += ['>%s|name=%s' % (kind, cid), ''.join(seq)]
        if kind == 'protein' and (len((e.get('unpairedMsa') or '').splitlines()) > 2
                                  or len((e.get('pairedMsa') or '').splitlines()) > 2):
          msa_dir = msa_dir or os.path.join(d, 'msa')
          os.makedirs(msa_dir, exist_ok=True)
          path = os.path.join(msa_dir, expected_basename(e['sequence']))
          if not os.path.exists(path):
            _write_pqt(e['sequence'], e.get('unpairedMsa') or '', path, pd, e.get('pairedMsa') or '')
      elif kind == 'ligand':
        if cid in glycan_chains:
          lines += ['>glycan|name=%s' % cid, _glycan_string(e['ccdCodes'], cid, j.get('bondedAtomPairs'))]
        elif 'smiles' in e:
          lines += ['>ligand|name=%s' % cid, e['smiles']]
        else:
          (code,) = e['ccdCodes']
          # chai takes a ligand ONLY as SMILES, so a CCD ligand gets chai's
          # SMILES atom order and names where ours builds it from the CCD -- no
          # stock equivalent of the input. Even a single-atom ion differs: its
          # atom NAME is an input feature (ion case: s_inputs off at the MG token
          # alone). ligand_smiles is the test of chai's ligand path.
          print('VENDOR_NA: chai1 has no CCD-ligand input (SMILES only); see ligand_smiles',
                file=sys.stderr)
          sys.exit(3)
          lines += ['>ligand|name=%s' % cid, _CCD_SMILES[code]]
  for k, (a, b) in enumerate(j.get('bondedAtomPairs') or ()):
    if a[0] == b[0] and a[0] in glycan_chains:
      continue  # a glycosidic bond: in the glycan string
    def ref(x):
      # chai's res_idx: residue one-letter + 1-based index, '@' atom; a ligand
      # or glycan names only its atom
      for entry in j['sequences']:
        (kind, e), = entry.items()
        if x[0] in (e['id'] if isinstance(e['id'], list) else [e['id']]):
          if kind == 'protein':
            return '%s%d@%s' % (e['sequence'][x[1] - 1], x[1], x[2])
          # a ligand or glycan side names only its atom: chai reads res_idx[0] as a
          # one-letter residue NAME, so '1@C1' asked for residue '1' (UNK) and matched nothing
          return '@%s' % x[2]
      raise KeyError(x)
    restraints.append(dict(restraint_id='bond%d' % k, chainA=a[0], res_idxA=ref(a), chainB=b[0], res_idxB=ref(b),
                           connection_type='covalent', confidence=1.0, min_distance_angstrom=0.0,
                           max_distance_angstrom=0.0, comment='bondedAtomPairs'))
  # pocket/contact constraints: chai names a residue by one-letter code + 1-based index; a pocket's chainA is the
  # binder (chain level), chainB the pocket residue (restraints.PairwiseInteraction)
  seqs = {i: e['sequence'] for entry in j['sequences'] for kind, e in entry.items() if kind == 'protein'
          for i in (e['id'] if isinstance(e['id'], list) else [e['id']])}
  res = lambda t: '%s%d' % (seqs[t[0]][t[1] - 1], t[1])
  for k, c in enumerate(j.get('constraints') or ()):
    (kind, e), = c.items()
    rows = ([(e['token1'][0], res(e['token1']), e['token2'][0], res(e['token2']))] if kind == 'contact' else
            [(e['binder'], '', t[0], res(t)) for t in e['contacts']])
    for m, (ca, ra, cb, rb) in enumerate(rows):
      restraints.append(dict(restraint_id='%s%d_%d' % (kind, k, m), chainA=ca, res_idxA=ra, chainB=cb, res_idxB=rb,
                             connection_type=kind, confidence=1.0, min_distance_angstrom=0.0,
                             max_distance_angstrom=e['maxDistance'], comment='constraints'))
  fasta = os.path.join(d, 'input.fasta')
  with open(fasta, 'w') as f:
    f.write('\n'.join(lines) + '\n')
  rpath = None
  if restraints:
    rpath = os.path.join(d, 'restraints.csv')
    pd.DataFrame(restraints).to_csv(rpath, index=False)
  _TEMPLATES.clear()
  for entry in j['sequences']:
    (kind, e), = entry.items()
    if kind != 'protein':
      continue
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    for t in e.get('templates') or ():
      _TEMPLATES.append((ids, e['sequence'], t))
  return fasta, msa_dir, rpath


_TEMPLATES = []


def _template_m8(case_dir, cache_folder):
  """chai's template input: an m8 hits file whose hit CIF chai fetches by PDB id
  into its cache folder -- pre-placed there (<ID>.cif.gz) so the run is offline
  and reads exactly the template the case carries. One hit per query chain,
  aligned over the case's own query/template index range; chai realigns the hit
  sequence itself (kalign), as it does for a server hit."""
  if not _TEMPLATES:
    return None
  import gzip
  import gemmi
  os.makedirs(cache_folder, exist_ok=True)
  lines = []
  for k, (ids, seq, t) in enumerate(_TEMPLATES):
    doc = gemmi.cif.read_string(t['mmcif'])
    pdb_id = doc.sole_block().find_value('_entry.id').strip().upper()
    chain = doc.sole_block().find_value('_entity_poly.pdbx_strand_id').split(',')[0].strip()
    with gzip.open(os.path.join(cache_folder, '%s.cif.gz' % pdb_id), 'wt') as f:
      f.write(t['mmcif'])
    q, tt = t['queryIndices'], t['templateIndices']
    for cid in ids:
      lines.append('\t'.join(map(str, [cid, '%s_%s' % (pdb_id.lower(), chain), 100.0, len(q), 0, 0,
                                      min(q) + 1, max(q) + 1, min(tt) + 1, max(tt) + 1, 1e-30, 500, ''])))
  path = os.path.join(case_dir, 'templates.m8')
  with open(path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
  return path


def _glycan_string(codes, cid, bonds):
  """chai's glycan string (glycans.py: `NAG(4-1 NAG)`, a sugar then each child
  as `(<src O>-<dst C> CHILD)`) from the case's intra-chain O->C bonds; a bare
  run of codes would be sugars with no glycosidic bond at all."""
  children = {i: [] for i in range(1, len(codes) + 1)}
  for a, b in bonds or ():
    if a[0] == b[0] == cid:
      (src, sa), (dst, da) = ((a[1], a[2]), (b[1], b[2])) if a[2].startswith('O') else ((b[1], b[2]), (a[1], a[2]))
      assert sa.startswith('O') and da.startswith('C'), (a, b)
      children[src].append((int(sa[1:]), int(da[1:]), dst))

  def emit(i):
    return codes[i - 1] + ''.join('(%d-%d %s)' % (s, d, emit(c)) for s, d, c in children[i])
  return emit(1)


def _a3m_rows(a3m):
  rows, header = [], None
  for line in a3m.splitlines():
    if line.startswith('>'):
      header = line[1:]
    elif line.strip():
      rows.append((header, line.strip()))
  return rows


def _write_pqt(query, a3m, path, pd, paired_a3m=''):
  """One chain's aligned.pqt as chai's ColabFold path writes it (colabfold.py):
  the pair a3m's rows keyed by their row index (the query is row 0, key '0';
  all-gap padding rows skipped), then the single a3m's rows past its query that
  are not already in the paired block."""
  from chai_lab.data.parsing.msas.data_source import MSADataSource
  rows = _a3m_rows(a3m) or [('query', query)]
  prows = _a3m_rows(paired_a3m)
  if prows:
    paired = [(str(i), h, s) for i, (h, s) in enumerate(prows) if set(s) != {'-'}]
  else:
    # no pair a3m: the paired block is the query alone
    paired = [('0', rows[0][0], rows[0][1])]
  in_paired = {s for _, _, s in paired}
  single = [(h, s) for i, (h, s) in enumerate(rows) if i > 0 and s not in in_paired]
  headers = [h for _, h, _ in paired] + [h for h, _ in single]
  sources = ['query'] + [MSADataSource.UNIREF90.value if h.startswith('UniRef') else MSADataSource.BFD_UNICLUST.value
                         for h in headers[1:]]
  df = pd.DataFrame(dict(sequence=[s for _, _, s in paired] + [s for _, s in single], source_database=sources,
                         pairing_key=[k for k, _, _ in paired] + [''] * len(single), comment=''))
  df.to_parquet(path)


def _np(t):
  import torch
  t = t.detach()
  if t.dtype == torch.bfloat16:
    t = t.float()
  return t.cpu().numpy()


def _fp32_graphs(module, from_dtypes=(15,)):
  """Rewrite every bfloat16 dtype argument (ScalarType 15) in a TorchScript
  module's graphs to float32 (6). chai's exported components keep fp32 weights
  and cast to bf16 inside the graph (the trunk: 1360 aten::to casts); this is the
  only way to run chai itself at fp32, since nothing inside can be hooked.
  Opt-in (CHAI_FP32=1): it is a precision experiment on the vendor, not chai."""
  import re
  changed = 0

  def walk(block, graph):
    nonlocal changed
    for node in block.nodes():
      for sub in node.blocks():
        walk(sub, graph)
      try:
        schema = node.schema()
      except RuntimeError:
        continue
      if schema == '(no schema)':
        continue
      args = [a.strip().split(' ')[-1].split('=')[0] for a in
              re.search(r'\((.*)\)', schema).group(1).split(',') if a.strip()]
      for i, inp in enumerate(node.inputs()):
        if i < len(args) and args[i] == 'dtype':
          src = inp.node()
          if src.kind() == 'prim::Constant' and src.hasAttribute('value') \
              and src.kindOf('value') == 'i' and src.i('value') in from_dtypes:
            with graph.insert_point_guard(node):
              node.replaceInput(i, graph.insertConstant(6))
            changed += 1

  # the bucketed entry points chai calls (chai1.AVAILABLE_MODEL_SIZES); they are
  # reachable by getattr but not listed by dir()
  for size in ('', 256, 384, 512, 768, 1024, 1536, 2048):
    try:
      g = getattr(module, 'forward_%s' % size if size else 'forward').graph
    except (AttributeError, RuntimeError):
      continue
    walk(g, g)
  # ...and every submodule's own methods: a call like `(to_qkv).forward2(x)` runs
  # that module's graph, with casts of its own. Each bucket calls its own
  # numbered copy (forward, forward1, forward2, ...), so walk them all, not just
  # `forward` (which left bucket 512's to_qkv in bf16)
  # (no dedupe by id(): the graph wrappers are temporaries and their ids are
  # reused, which silently skipped most submodules; a second pass finds nothing)
  for sub in module.modules():
    try:
      names = sub._c._method_names()
    except AttributeError:
      continue
    for name in names:
      try:
        g = sub._c._get_method(name).graph
      except RuntimeError:
        continue
      walk(g, g)
  return changed


def _run(case_dir, recycles, on_inputs, inject=None, traj=None):
  """chai's own featurise + fold, with every exported component wrapped.

  traj (the `trajectory` command): chai's real 200-step sampler, one sample,
  with chai1's torch.randn and center_random_augmentation replaced by the
  harness's draws (see trajectory); nothing injected into the components."""
  from pathlib import Path
  import torch
  import chai_lab.chai1 as C
  from chai_lab.utils.tensor_utils import set_seed

  with open(os.path.join(case_dir, 'case.json')) as f:
    j = json.load(f)
  fasta, msa_dir, rpath = _case_inputs(j, case_dir)
  set_seed([1])
  out = Path(case_dir) / 'chai_out'
  import shutil
  shutil.rmtree(out, ignore_errors=True)
  m8 = _template_m8(case_dir, str(out / 'templates'))
  ctx = C.make_all_atom_feature_context(
      Path(fasta), output_dir=out, use_esm_embeddings=not os.environ.get('CHAI_NO_ESM'),
      esm_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
      msa_directory=Path(msa_dir) if msa_dir else None,
      constraint_path=Path(rpath) if rpath else None,
      templates_path=Path(m8) if m8 else None)
  got = {}
  orig_collate = C.Collate

  class Collate(orig_collate):
    def __call__(self, contexts):
      batch = super().__call__(contexts)
      got['inputs'] = batch['inputs']
      if os.environ.get('CHAI_DUMP_FEATURES'):
        # chai's raw (pre-embedding) feature tensors, for a field-by-field diff
        np.savez(os.path.join(case_dir, 'chai_features.npz'),
                 **{k: _np(v)[0] for k, v in batch['features'].items()
                    if hasattr(v, 'shape') and v.numel() < 5e7},
                 inp_msa_mask=_np(batch['inputs']['msa_mask'])[0])
      on_inputs(batch['inputs'])
      return batch
  C.Collate = Collate

  class Wrap:
    def __init__(self, name, inner):
      self.name, self.inner, self.calls = name, inner, 0

    def forward(self, **kw):
      self.calls += 1
      if inject is not None and self.name == 'diffusion_module.pt' and self.calls == 1:
        dev = kw['atom_noised_coords'].device
        kw = dict(kw, atom_noised_coords=torch.tensor(inject['noisy'], device=dev)[None, None],
                  noise_sigma=torch.full((1, 1), inject['sigma'], device=dev))
        out = self.inner.forward(**kw)
        got['denoised'] = _np(out).reshape(-1, 3)   # (1, [ds,] atoms, 3) -> (atoms, 3)
        return out
      if self.name == 'confidence_head.pt' and inject is not None:
        kw = dict(kw, atom_coords=torch.tensor(inject['positions'], device=kw['atom_coords'].device)[None])
      if self.name == 'confidence_head.pt' and traj is not None:
        got.setdefault('final', _np(kw['atom_coords']).reshape(-1, 3))
        raise _Stop()
      if self.name == 'trunk.pt' and os.environ.get('CHAI_TRUNK_INIT'):
        # chai's own trunk on SUPPLIED initial reprs (ours): how far chai's output
        # moves for that input difference alone -- the gap a correct trunk shows
        sub = np.load(os.environ['CHAI_TRUNK_INIT'])
        for key, name in (('token_pair_trunk_initial_repr', 'z_init'), ('token_single_trunk_initial_repr', 's_init')):
          t = kw[key].clone()
          v = torch.tensor(sub[name], dtype=t.dtype, device=t.device)
          n = v.shape[0]
          if t.dim() == 4:
            t[0, :n, :n] = v
          else:
            t[0, :n] = v
          kw[key] = t
          if self.calls == 1 and key == 'token_pair_trunk_initial_repr':
            kw['token_pair_trunk_repr'] = t       # the first pass's recycled input IS the initial repr
          if self.calls == 1 and key == 'token_single_trunk_initial_repr':
            kw['token_single_trunk_repr'] = t
      out = self.inner.forward(**kw)
      if self.name == 'feature_embedding.pt':
        got['emb.TOKEN'] = _np(out['TOKEN'])[0]
        got['emb.MSA'] = _np(out['MSA'][:, :8])[0]          # the first rows: live ones lead
        got['emb.ATOM'] = _np(out['ATOM'])[0]
      if self.name == 'token_embedder.pt':
        got['s_inputs'] = _np(out[0])[0]
      elif self.name == 'trunk.pt':
        got['s'], got['z'] = _np(out[0])[0], _np(out[1])[0]
        if self.calls == 1:
          # the trunk's own inputs, for a stage split: is the gap in the
          # embedder or in the trunk?
          got['emb.z_init'] = _np(kw['token_pair_trunk_initial_repr'])[0]
          got['emb.s_init'] = _np(kw['token_single_trunk_initial_repr'])[0]
          got['emb.trunk_pass0_z'] = got['z']
          got['emb.trunk_pass0_s'] = got['s']
      elif self.name == 'confidence_head.pt':
        got['pae_logits'], got['pde_logits'], got['plddt_logits'] = (_np(x)[0] for x in out)
      return out

  orig = C.load_exported

  def load(key, device):
    w = orig(key, device)
    if os.environ.get('CHAI_FP32'):
      n = _fp32_graphs(w.jit_module)
      print('CHAI_FP32: %s: %d bf16 dtype arguments -> fp32' % (key, n), file=sys.stderr)
    return Wrap(key, w)
  C.load_exported = load
  orig_torch, orig_aug = C.torch, C.center_random_augmentation
  if traj is not None:
    C.torch, C.center_random_augmentation = traj
  try:
    C.run_folding_on_context(ctx, output_dir=out, num_trunk_recycles=recycles + 1,
                             num_diffn_timesteps=200 if traj is not None else 2,
                             num_diffn_samples=1, seed=1, device=torch.device('cuda:0'), low_memory=False)
  except _Stop:
    pass
  finally:
    C.load_exported, C.Collate = orig, orig_collate
    C.torch, C.center_random_augmentation = orig_torch, orig_aug
  return got


def _layout(inputs):
  real = _np(inputs['atom_exists_mask'])[0].astype(bool)
  names = [n for n, r in zip(inputs['atom_ref_name'][0] if isinstance(inputs['atom_ref_name'][0], list)
                             else inputs['atom_ref_name'], real) if r]
  return dict(
      atom_name=np.array([str(n) for n in names]),
      atom_token=_np(inputs['atom_token_index'])[0][real],
      atom_pad=np.ones(int(real.sum()), bool),
      token_pad=_np(inputs['token_exists_mask'])[0].astype(bool)[_np(inputs['token_exists_mask'])[0].astype(bool)],
      atom_slot=np.flatnonzero(real),
      **{'feat.ref_pos': _np(inputs['atom_ref_pos'])[0][real],
         'feat.ref_space_uid': _np(inputs['atom_ref_space_uid'])[0][real],
         'feat.ref_element': np.eye(128, dtype=np.int64)[_np(inputs['atom_ref_element'])[0][real]],
         'atomfeat.ref_charge': _np(inputs['atom_ref_charge'])[0][real],
         'atomfeat.ref_name_chars': _np(inputs['atom_ref_name_chars'])[0][real]},
      # atoms chai keeps but does not place (a ligand's declared-bond leaving
      # atom: reference features, atom_exists false): their reference draw, so
      # the harness can inject it into ours as it does for placed atoms
      **_ref_only(inputs, real))


def _ref_only(inputs, real):
  if 'atom_ref_mask' not in inputs:
    return {}
  ref = _np(inputs['atom_ref_mask'])[0].astype(bool) & ~real
  names = inputs['atom_ref_name'][0] if isinstance(inputs['atom_ref_name'][0], list) else inputs['atom_ref_name']
  return {'refonly.atom_name': np.array([str(n) for n, r in zip(names, ref) if r]),
          'refonly.atom_token': _np(inputs['atom_token_index'])[0][ref],
          'refonly.ref_pos': _np(inputs['atom_ref_pos'])[0][ref]}


def _esm_rows(inputs):
  """chai's ESM2 rows, one per protein RESIDUE in token order: chai gives each of
  a residue's tokens its residue's row (get_esm_embedding_context), which is the
  per-residue form model_features accepts."""
  emb = _np(inputs['esm_embeddings'])[0]
  real = _np(inputs['token_exists_mask'])[0].astype(bool)
  protein = real & (_np(inputs['token_entity_type'])[0] == 0)   # EntityType.PROTEIN
  asym, res = _np(inputs['token_asym_id'])[0], _np(inputs['token_residue_index'])[0]
  seen, keep = set(), []
  for t in np.flatnonzero(protein):
    if (asym[t], res[t]) not in seen:
      seen.add((asym[t], res[t]))
      keep.append(t)
  return emb[keep].astype(np.float32)


def featurise(model, case_json, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  import shutil
  shutil.copy(case_json, os.path.join(out_dir, 'case.json'))

  def on_inputs(inputs):
    np.savez(os.path.join(out_dir, 'layout.npz'), **_layout(inputs))
    rows = _esm_rows(inputs)
    if not os.environ.get('CHAI_NO_ESM') and len(rows):
      np.savez(os.path.join(out_dir, 'esm2_rows.npz'), esm=rows)
    raise _Stop
  _run(out_dir, 0, on_inputs)


def forward(model, out_dir, in_npz, out_npz):
  z = np.load(in_npz)
  if str(z['precision']) == 'bf16':
    # the vendor's STOCK precision (bf16 grading): chai's in-graph bf16 casts stay
    os.environ.pop('CHAI_FP32', None)
  lay = np.load(os.path.join(out_dir, 'layout.npz'))
  slot = lay['atom_slot']

  def padded(x):
    out = np.zeros((5888 if slot.max() < 5888 else int(slot.max()) + 1, 3), np.float32)
    out[slot] = x
    return out
  state = {}

  def on_inputs(inputs):
    n_atoms = inputs['atom_exists_mask'].shape[-1]
    state['n'] = n_atoms
    now = _np(inputs['atom_ref_pos'])[0][slot]
    if not np.array_equal(now, lay['feat.ref_pos']):
      raise RuntimeError('chai re-drew a different reference conformer (max |d| %.3g)'
                         % np.abs(now - lay['feat.ref_pos']).max())

  def pad_to(x, n):
    out = np.zeros((n, 3), np.float32)
    out[slot] = x
    return out
  # the padded atom count is only known after collation; chai pads to 23 * the token bucket
  n_tok_bucket = None
  inject = {'sigma': float(z['noise_level'])}

  class Lazy(dict):
    def __getitem__(self, k):
      n = state['n']
      if k == 'noisy':
        return pad_to(z['positions_noisy'], n)
      if k == 'positions':
        return pad_to(z['positions'], n)
      return dict.__getitem__(self, k)
  got = _run(out_dir, int(z['recycling_steps']), on_inputs, inject=Lazy(inject))
  inputs = got['inputs']
  tok = _np(inputs['token_exists_mask'])[0].astype(bool)
  n = int(tok.sum())
  centres = lambda lo, hi, k: (np.linspace(lo, hi, k + 1)[:-1] + np.linspace(lo, hi, k + 1)[1:]) / 2

  def expect(logits, c):
    p = np.exp(logits - logits.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True) * c).sum(-1)
  pl_atoms = expect(got['plddt_logits'], centres(0, 1, got['plddt_logits'].shape[-1]))
  np.savez(out_npz,
           s_inputs=got['s_inputs'][:n], s=got['s'][:n], z=got['z'][:n, :n],
           pae=expect(got['pae_logits'][:n, :n], centres(0, 32, 64)),
           pde=expect(got['pde_logits'][:n, :n], centres(0, 32, 64)),
           plddt=pl_atoms[slot], denoised=got['denoised'][slot],
           **{'tap.emb_token': got['emb.TOKEN'][:n], 'tap.emb_atom': got['emb.ATOM'][slot],
              'tap.z_init': got['emb.z_init'][:n, :n], 'tap.s_init': got['emb.s_init'][:n],
              'tap.emb_msa': got['emb.MSA'][:, :n]})


def trajectory(model, out_dir, in_npz, out_npz):
  """chai's REAL sampler (run_folding_on_context at its default 200 timesteps,
  one sample: 199 Heun steps on the midpoint schedule), every random draw the
  harness's (vendor_native_parity --trajectory): chai1's two torch.randn calls
  (initial positions, per-step noise) and center_random_augmentation (which
  rotates as R @ x; the replacement applies the harness's x @ R, keeping
  chai's masked centroid). chai samples over its PADDED atom axis: the draws
  go to the real atoms (atom_exists_mask, the layout's order), zero elsewhere."""
  import torch
  import chai_lab.chai1 as C
  from chai_lab.model.utils import calc_centroid
  z = np.load(in_npz)
  x0 = torch.tensor(z['traj_x0'], dtype=torch.float32)
  eps = torch.tensor(z['traj_eps'], dtype=torch.float32)
  rot = torch.tensor(z['traj_rot'], dtype=torch.float32)
  trans = torch.tensor(z['traj_trans'], dtype=torch.float32)
  n = {'randn': 0, 'aug': 0}
  box = {}

  def scatter(v, shape, device):
    out = torch.zeros(shape, dtype=torch.float32)
    out.reshape(-1, shape[-2], 3)[:, box['real']] = v
    return out.to(device)

  class _Torch:
    def __getattr__(self, k):
      return getattr(torch, k)

    def randn(self, *size, device=None, **kw):
      shape = tuple(size[0]) if len(size) == 1 and not isinstance(size[0], int) else size
      k = n['randn']
      n['randn'] += 1
      return scatter(x0 if k == 0 else eps[k - 1], shape, device)

  def aug(atom_coords, atom_single_mask, s_trans=1.0, rotations=None):
    k = n['aug']
    n['aug'] += 1
    centroid = calc_centroid(atom_coords, mask=atom_single_mask)[:, None]
    x = atom_coords - centroid
    return x @ rot[k].to(x) + s_trans * trans[k].to(x)

  def on_inputs(inputs):
    box['real'] = torch.from_numpy(_np(inputs['atom_exists_mask'])[0].astype(bool))
  got = _run(out_dir, int(z['recycling_steps']), on_inputs, traj=(_Torch(), aug))
  from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule
  sig = InferenceNoiseSchedule(s_max=C.DiffusionConfig.S_tmax, s_min=4e-4, p=7.0,
                               sigma_data=C.DiffusionConfig.sigma_data).get_schedule(
                                   device=torch.device('cpu'), num_timesteps=200)
  np.savez(out_npz, sigmas=sig.float().numpy(), steps=n['aug'],
           final=got['final'][box['real'].numpy()])


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward, 'trajectory': trajectory}[cmd](*rest)
