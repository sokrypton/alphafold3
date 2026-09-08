"""L6: fold a NON-PROTEIN or multi-entity target and score it against a reference.

`fold_check.py` folds one protein chain and scores CA-RMSD. Everything else this
library claims to support -- RNA, DNA, ligands, complexes -- had its screens in a
different repository, so this repo could not re-run them at all. This is that
screen, in-tree, on the same featurisation and the same per-model conventions a
real run uses.

  PYTHONPATH=src:. python dev/oracles/modality_check.py <model> <case>

Each case names an input and a rule for WHICH atoms to score, because the right
atom differs by modality and scoring the wrong one is a silent way to report
success: protein CA, nucleic C1', a ligand its own heavy atoms after aligning on
the protein it binds. (Scoring an RNA on the wrong atom is not hypothetical --
it is the trap recorded in memory rna-parity-status.)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HOME = os.path.expanduser('~')

# name -> (mmCIF, entity spec). `chains` is built lazily so folding_input is
# imported inside main, after sys.argv has been neutralised for absl.
CASES = {
    'rna_1ehz': dict(cif=_HOME + '/1EHZ.cif', kind='rna', chain='A'),
    # 1LMB is the lambda-repressor operator complex: chains 1 and 2 are a DNA
    # duplex, 3 and 4 the protein dimer. Two cases out of one file -- the duplex
    # alone, and the whole complex, which is the only case here that tests an
    # INTERFACE (scored in one shared frame, so a correct-but-misplaced chain
    # fails).
    'dna_1lmb': dict(cif=_HOME + '/1LMB.cif', kind='multi',
                     entities=[('1', 'dna'), ('2', 'dna')]),
    'complex_1lmb': dict(cif=_HOME + '/1LMB.cif', kind='multi',
                         entities=[('3', 'protein'), ('4', 'protein'),
                                   ('1', 'dna'), ('2', 'dna')]),
    # The ligand case reads its input from the SAME JSON the earlier screens
    # used, MSA included: streptavidin folded from a single sequence lands at
    # 3-5 A, which measures the missing MSA rather than the ligand.
    'ligand_1stp': dict(cif=_HOME + '/1STP.cif', kind='ligand', chain='A',
                        ligand='BTN', json=_HOME + '/1stp_msa.json'),
    # Ubiquitin phosphorylated at Ser20 -- the PTM case. AF3 atomises a modified
    # residue, so this exercises a path no plain-protein fold reaches, and
    # `--write` checks SEP actually survives into the output file.
    'ptm_5k9p': dict(cif=_HOME + '/5K9P.cif', kind='ptm', chain='A',
                     ptms=[('SEP', 20)]),
    # The SAME target with the modification removed -- the control for the PTM
    # case. Without it a bad PTM number cannot be told apart from a model that
    # simply folds this target badly.
    'plain_5k9p': dict(cif=_HOME + '/5K9P.cif', kind='ptm', chain='A',
                       ptms=[]),
    'protein_6mrr': dict(cif=None, kind='protein', chain='A'),
}

_RNA = {'A': 'A', 'G': 'G', 'C': 'C', 'U': 'U'}
_DNA = {'DA': 'A', 'DG': 'G', 'DC': 'C', 'DT': 'T'}
_AA3 = None      # filled from converters.pdb.A3 on first use


def read_cif_atoms(path):
  """-> list of dicts for the _atom_site loop. A minimal reader on purpose.

  The runtime venv has neither gemmi nor biopython, and adding one to run a
  screen would put a dependency in the way of the screen being run at all.
  """
  cols, rows, in_loop = [], [], False
  for line in open(path):
    line = line.rstrip('\n')
    if line.startswith('_atom_site.'):
      cols.append(line.split('.', 1)[1].strip())
      in_loop = True
      continue
    if in_loop:
      if not line.strip() or line.startswith('#'):
        if rows:
          break
        continue
      if line.startswith('loop_') or line.startswith('_'):
        break
      parts = line.split()
      if len(parts) == len(cols):
        rows.append(dict(zip(cols, parts)))
  if not rows:
    raise SystemExit('no _atom_site rows in %s' % path)
  return rows


def _f(r, k, alt):
  return r.get(k, r.get(alt, ''))


def reference(case):
  """-> (sequence, {(seq_id, atom_name): xyz}) for the scored polymer chain."""
  rows = read_cif_atoms(case['cif'])
  want = case['chain']
  seq, coords, seen = [], {}, set()
  for r in rows:
    ch = _f(r, 'auth_asym_id', 'label_asym_id')
    if ch != want:
      continue
    comp = _f(r, 'auth_comp_id', 'label_comp_id')
    name = _f(r, 'auth_atom_id', 'label_atom_id').strip('"')
    sid = _f(r, 'auth_seq_id', 'label_seq_id')
    alt = r.get('label_alt_id', '.')
    if alt not in ('.', '?', 'A'):
      continue
    xyz = np.array([float(r['Cartn_x']), float(r['Cartn_y']),
                    float(r['Cartn_z'])])
    coords[(sid, name)] = xyz
    if (sid,) not in seen:
      letter = _letter(comp, case['kind'])
      if letter is not None:
        seen.add((sid,))
        seq.append((sid, letter))
  return seq, coords


def _letter(comp, kind):
  """Residue -> one-letter code, resolving MODIFIED residues to their parent.

  1EHZ is tRNA-Phe: 14 of its 76 residues are modified (PSU, 2MG, H2U, 1MA,
  7MG, 5MC, 5MU, OMC, OMG, YG ...). Skipping them silently -- what a plain
  A/G/C/U table does -- gave a 62-residue "sequence" whose residue indices no
  longer lined up with the coordinates, i.e. a screen that would fold the wrong
  molecule and score it against a shifted reference. The parent comes from the
  CCD we already ship (`mon_nstd_parent_comp_id`), which is the same field AF3
  uses to give an atomised residue its parent restype.
  """
  table = {'rna': _RNA, 'dna': _DNA}.get(kind)
  if table is not None:
    if comp in table:
      return table[comp]
    parent = _ccd_parent(comp)
    return table.get(parent)
  from converters.pdb import A3
  if comp in A3:
    return A3[comp]
  parent = _ccd_parent(comp)
  return A3.get(parent)


_CCD = None


def _ccd_parent(comp):
  global _CCD
  if _CCD is None:
    from alphafold3.constants import decoded_ccd
    _CCD = decoded_ccd.get_ccd()
  entry = _CCD.get(comp)
  if entry is None:
    return None
  vals = entry.get('_chem_comp.mon_nstd_parent_comp_id', [])
  parent = (vals[0] if vals else '').strip('"').split(',')[0]
  return None if parent in ('', '?', '.') else parent


def ligand_coords(case):
  """-> (names, elements, xyz) for the ligand's heavy atoms, in FILE order.

  Pairing predicted to reference ligand atoms cannot go by name: our featurised
  batch stores a CCD ligand's atom names as the ELEMENT alone
  (`ref_structure.atom_name_chars` reads [35,0,0,0] = 'C' for every carbon of
  BTN), so there is nothing to match 'C11' against. What survives is ORDER --
  both sides list the ligand in its CCD atom order -- and that is checkable:
  the element sequences must agree atom for atom, which the caller asserts
  before scoring anything.
  """
  rows = read_cif_atoms(case['cif'])
  names, elems, xyz = [], [], []
  for r in rows:
    comp = _f(r, 'auth_comp_id', 'label_comp_id')
    if comp != case['ligand']:
      continue
    name = _f(r, 'auth_atom_id', 'label_atom_id').strip('"')
    el = r.get('type_symbol', name[:1]).strip()
    if el == 'H':
      continue
    if names and name in names:      # a second copy of the ligand
      break
    names.append(name)
    elems.append(el)
    xyz.append([float(r['Cartn_x']), float(r['Cartn_y']), float(r['Cartn_z'])])
  if not names:
    raise SystemExit('no %s atoms in %s' % (case['ligand'], case['cif']))
  return names, elems, np.asarray(xyz)


def kabsch(a, b):
  """RMSD of a onto b after optimal superposition."""
  a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
  ac, bc = a - a.mean(0), b - b.mean(0)
  u, _, vt = np.linalg.svd(ac.T @ bc)
  d = np.sign(np.linalg.det(u @ vt))
  r = u @ np.diag([1.0, 1.0, d]) @ vt
  return float(np.sqrt(((ac @ r - bc) ** 2).sum(1).mean())), r, a.mean(0), b.mean(0)


def check_output(model_name, batch, out, outdir, case, ref_seq):
  """Write the model's own mmCIF and check it describes what we asked for.

  The structure WRITER is as much a part of multimodal support as the network:
  a model can place a ligand correctly and still emit a file that drops it, and
  no RMSD above would notice. This re-reads what we wrote and checks the entity
  content, the coordinates and the pLDDT range.
  """
  import fold_check
  from alphafold3.model import model as af3_model

  os.makedirs(outdir, exist_ok=True)
  # The writer needs the string-typed layout columns the forward pass strips.
  raw = fold_check.LAST_RAW_BATCH or batch
  # get_inference_result reads `.flags` on the arrays it is handed, which a jax
  # ArrayImpl does not have -- so bring the model output back to numpy first.
  import jax
  out_np = jax.tree_util.tree_map(
      lambda x: np.asarray(x) if hasattr(x, 'shape') else x, out)
  # run_alphafold stamps the params' identifier onto the result before writing;
  # a harness that calls the model directly has to supply one or the writer
  # raises KeyError('__identifier__').
  out_np = dict(out_np)
  out_np.setdefault('__identifier__', model_name.encode())
  results = list(af3_model.Model.get_inference_result(
      batch=raw, result=out_np, target_name=model_name))
  path = os.path.join(outdir, '%s_%s.cif' % (model_name, case['kind']))
  with open(path, 'w') as f:
    f.write(results[0].predicted_structure.to_mmcif())
  rows = read_cif_atoms(path)
  chains = sorted({_f(r, 'auth_asym_id', 'label_asym_id') for r in rows})
  comps = {}
  for r in rows:
    c = _f(r, 'auth_comp_id', 'label_comp_id')
    comps[c] = comps.get(c, 0) + 1
  xyz = np.array([[float(r['Cartn_x']), float(r['Cartn_y']),
                   float(r['Cartn_z'])] for r in rows])
  bfac = np.array([float(r.get('B_iso_or_equiv', 'nan')) for r in rows])
  print('  wrote %s: %d atoms, chains %s, %d component types'
        % (os.path.basename(path), len(rows), chains, len(comps)))
  print('    finite coords %s | pLDDT within [0,100] %s'
        % (bool(np.isfinite(xyz).all()),
           bool(np.isfinite(bfac).all() and (bfac >= 0).all()
                and (bfac <= 100).all())))
  for ptm, _pos in case.get('ptms', ()):
    print('    modified residue %s in output: %d atoms' % (ptm, comps.get(ptm, 0)))
    assert comps.get(ptm), 'the writer dropped the modified residue %s' % ptm
  if case.get('ligand'):
    n_lig = comps.get(case['ligand'], 0)
    print('    ligand %s in output: %d atoms' % (case['ligand'], n_lig))
    assert n_lig, 'the writer dropped the ligand'
  assert np.isfinite(xyz).all(), 'non-finite coordinates in the written file'
  return path


def multi_reference(case):
  """-> [(chain, kind, [(seq_id, letter)], {(seq_id, atom): xyz})] per entity.

  Copies of one entity are trimmed to their COMMON residue ids. 1LMB's two
  protein chains are the same protein with different resolved ranges (87 vs 92
  residues), so without this they are not interchangeable and the symmetry-aware
  complex score below cannot try the swap -- which matters, because a model that
  builds a perfect dimer with the copies swapped otherwise scores 17 A while
  every chain reads ~1 A.
  """
  out = []
  for chain, kind in case['entities']:
    sub = dict(case, chain=chain, kind=kind)
    seq, coords = reference(sub)
    out.append([chain, kind, seq, coords])
  by_kind = {}
  for ent in out:
    by_kind.setdefault(ent[1], []).append(ent)
  for kind, group in by_kind.items():
    if len(group) < 2:
      continue
    common = set.intersection(*[{sid for sid, _l in e[2]} for e in group])
    trimmed = [[(sid, l) for sid, l in e[2] if sid in common] for e in group]
    # ONLY when the trim leaves something and leaves the copies identical. The
    # two DNA strands of 1LMB are complementary with disjoint numbering, so the
    # intersection is EMPTY -- trimming to it would have silently dropped the
    # DNA from the complex score entirely.
    if common and len({''.join(l for _s, l in t) for t in trimmed}) == 1:
      for e, t in zip(group, trimmed):
        e[2] = t
      print('  %d %s chains trimmed to %d common residues (identical there)'
            % (len(group), kind, len(common)))
  return [tuple(e) for e in out]


def run_multi(args, case):
  """Fold a multi-entity target and score each entity AND the whole complex."""
  import fold_check
  from alphafold3.common import folding_input
  from alphafold3.model import feat_batch

  ents = multi_reference(case)
  ids = [chr(ord('A') + i) for i in range(len(ents))]
  chains = []
  for cid, (_ch, kind, seq, _c) in zip(ids, ents):
    letters = ''.join(l for _, l in seq)
    if kind == 'protein':
      chains.append(folding_input.ProteinChain(
          id=cid, sequence=letters, ptms=[], unpaired_msa='>q\n%s\n' % letters,
          paired_msa='', templates=[]))
    elif kind == 'dna':
      chains.append(folding_input.DnaChain(id=cid, sequence=letters,
                                           modifications=[]))
    elif kind == 'rna':
      chains.append(folding_input.RnaChain(id=cid, sequence=letters,
                                           modifications=[], unpaired_msa=''))
    else:
      raise SystemExit('unknown entity kind %r' % kind)
  print('%s / %s: %s'
        % (args.model, args.case,
           ', '.join('%s=%s(%d)' % (cid, k, len(sq))
                     for cid, (_c, k, sq, _x) in zip(ids, ents))))

  out, batch = fold_check.fold(args.model, '', model_dir=args.model_dir,
                               seed=args.seed, chains=chains)
  pos = np.asarray(out['diffusion_samples']['atom_positions'])
  fb = feat_batch.Batch.from_data_dict(batch)
  names = np.asarray(fb.ref_structure.atom_name_chars)

  def atom_name(t, a):
    v = names[t, a]
    if v.ndim == 2:
      v = v.argmax(-1)
    return ''.join(chr(32 + int(i)) for i in v).strip()

  # Tokens run entity by entity in the order the chains were given, one token
  # per polymer residue -- so the offset is a running sum, not a search.
  off, per, joint_got, joint_ref = 0, [], [], []
  for cid, (_ch, kind, seq, coords) in zip(ids, ents):
    want = "C1'" if kind in ('dna', 'rna') else 'CA'
    got, ref = [], []
    for i, (sid, _l) in enumerate(seq):
      t = off + i
      for a in range(pos.shape[2]):
        if atom_name(t, a) == want and (sid, want) in coords:
          got.append(pos[:, t, a, :])
          ref.append(coords[(sid, want)])
          break
    off += len(seq)
    if not got:
      print('  %s: no %s atoms matched' % (cid, want))
      continue
    got = np.stack(got, 1)
    ref = np.stack(ref)
    rs = [kabsch(got[i], ref)[0] for i in range(got.shape[0])]
    per.append((cid, kind, want, min(rs)))
    joint_got.append(got)
    joint_ref.append(ref)
    print('  %s (%s): %d %s atoms, best %.3f, mean %.3f'
          % (cid, kind, len(ref), want, min(rs), float(np.mean(rs))))
  if len(joint_got) > 1:
    # SYMMETRY. 1LMB's two protein chains are the same protein, so a model that
    # builds a perfect dimer with the copies swapped scores ~17 A here while
    # every chain is individually ~1 A. Try the assignments that identical
    # sequences allow and keep the best -- otherwise this measures chain
    # labelling, not docking.
    import itertools

    seqs = [''.join(l for _, l in sq) for _c, _k, sq, _x in ents]
    groups = {}
    for i, q in enumerate(seqs):
      groups.setdefault(q, []).append(i)
    swappable = [g for g in groups.values() if len(g) > 1]
    perms = [list(range(len(ents)))]
    for g in swappable:
      new_perms = []
      for base in perms:
        for order in itertools.permutations(g):
          p = list(base)
          for slot, src in zip(g, order):
            p[slot] = base[src]
          new_perms.append(p)
      perms = new_perms
    r = np.concatenate(joint_ref)
    best, best_perm = None, None
    for perm in perms:
      g = np.concatenate([joint_got[i] for i in perm], 1)
      if g.shape[1] != r.shape[0]:
        continue
      rs = [kabsch(g[i], r)[0] for i in range(g.shape[0])]
      if best is None or min(rs) < best[0]:
        best, best_perm = (min(rs), float(np.mean(rs))), perm
    print('  WHOLE COMPLEX in one frame: %d atoms, best %.3f, mean %.3f'
          ' (%d assignment%s tried%s)'
          % (len(r), best[0], best[1], len(perms),
             '' if len(perms) == 1 else 's',
             '' if best_perm == list(range(len(ents)))
             else ', best is a swap'))
  if args.write:
    check_output(args.model, batch, out, args.write, case, [])
  return 0


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('case', choices=sorted(CASES))
  ap.add_argument('--model_dir', default=None)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--write', metavar='DIR', default=None,
                  help="also write the model's own mmCIF and validate it")
  args = ap.parse_args(argv)
  sys.argv = sys.argv[:1]          # absl parses argv lazily

  import fold_check
  from alphafold3.common import folding_input

  case = CASES[args.case]
  if case['kind'] == 'multi':
    return run_multi(args, case)
  if args.case == 'protein_6mrr':
    seq, native = fold_check.parse_ca(_HOME + '/6MRR.pdb')
    chains = None
    ref_seq = [(str(i), c) for i, c in enumerate(seq)]
  else:
    ref_seq, ref_atoms = reference(case)
    seq = ''.join(c for _, c in ref_seq)
    kind = case['kind']
    if kind == 'rna':
      # unpaired_msa='' means "no MSA", which featurisation requires to be
      # stated explicitly (None is rejected). Single-sequence RNA is the
      # comparable setting: none of these ports is given an RNA MSA here.
      chains = [folding_input.RnaChain(id='A', sequence=seq, modifications=[],
                                       unpaired_msa='')]
    elif kind == 'dna':
      chains = [folding_input.DnaChain(id='A', sequence=seq, modifications=[])]
    elif kind == 'ptm':
      # NO_MSA=1 drops the self-MSA, which is what a native protenix run with
      # `--use_msa false` does -- the last input difference between the two
      # sides when comparing against native.
      msa = '' if os.environ.get('NO_MSA') else '>q\n%s\n' % seq
      chains = [folding_input.ProteinChain(
          id='A', sequence=seq, ptms=list(case['ptms']),
          unpaired_msa=msa, paired_msa='', templates=[])]
      print('  modifications: %s (msa: %s)'
            % (case['ptms'], 'none' if not msa else 'self'))
    elif kind == 'ligand':
      if case.get('json'):
        fi = folding_input.Input.from_json(open(case['json']).read())
        chains = list(fi.chains)
        have = {getattr(c, 'id', None) for c in chains}
        if not any(isinstance(c, folding_input.Ligand) for c in chains):
          chains.append(folding_input.Ligand(id='B',
                                             ccd_ids=[case['ligand']]))
        print('  input from %s: %d chains %s'
              % (os.path.basename(case['json']), len(chains), sorted(have)))
      else:
        chains = [folding_input.ProteinChain(id='A', sequence=seq, ptms=[],
                                             unpaired_msa='', paired_msa='',
                                             templates=[]),
                  folding_input.Ligand(id='B', ccd_ids=[case['ligand']])]
    else:
      raise SystemExit('unknown kind %r' % kind)

  print('%s / %s: %d residues' % (args.model, args.case, len(ref_seq)))
  out, batch = fold_check.fold(args.model, seq, model_dir=args.model_dir,
                               seed=args.seed, chains=chains)
  pos = np.asarray(out['diffusion_samples']['atom_positions'])   # (S, T, A, 3)
  if args.case == 'protein_6mrr':
    ca = pos[:, :, 1, :]
    rs = [kabsch(ca[i], native)[0] for i in range(ca.shape[0])]
    print('  %d samples, CA-RMSD %s  best %.3f'
          % (len(rs), ' '.join('%.3f' % r for r in rs), min(rs)))
    return 0

  # Dense slot -> atom name, from the batch itself: the scored atom must be
  # located by NAME, not by a slot index guessed per modality (a nucleotide's
  # C1' is not at a fixed slot across residue types).
  from alphafold3.model import feat_batch
  fb = feat_batch.Batch.from_data_dict(batch)
  names = np.asarray(fb.ref_structure.atom_name_chars)   # (T, A, 4) or (T, A, 4, 64)

  def atom_name(t, a):
    v = names[t, a]
    if v.ndim == 2:            # one-hot over 64 printable characters
      v = v.argmax(-1)
    return ''.join(chr(32 + int(i)) for i in v).strip()

  n_tok = pos.shape[1]
  score_atom = {'rna': "C1'", 'dna': "C1'", 'ligand': 'CA',
                'ptm': 'CA'}[case['kind']]
  # Token index is NOT residue index once anything is atomised: AF3 splits a
  # modified residue into one token per atom, so every residue after it shifts.
  # Read the residue number off the batch and match on THAT. Scoring ubiquitin
  # phospho-Ser20 by position gave 12.4 A -- a misalignment, not a bad fold.
  # GROUP tokens into residues, then walk residues in order. Matching on the
  # residue NUMBER fails because the batch numbers residues within the chain
  # while the reference carries the mmCIF's auth_seq_id (1STP chain A does not
  # start at 1), and matching on token POSITION fails because an atomised
  # residue becomes one token per atom. Grouping by (asym_id, residue_index)
  # and taking the n-th group is right under both.
  res_index = np.asarray(fb.token_features.residue_index)
  asym_id = np.asarray(fb.token_features.asym_id)
  groups, seen = [], None
  for t in range(n_tok):
    key = (int(asym_id[t]), int(res_index[t]))
    if key != seen:
      groups.append([t])
      seen = key
    else:
      groups[-1].append(t)
  got, ref = [], []
  for g, (sid, _letter) in enumerate(ref_seq):
    if g >= len(groups) or (sid, score_atom) not in ref_atoms:
      continue
    for t in groups[g]:
      hit = False
      for a in range(pos.shape[2]):
        if atom_name(t, a) == score_atom:
          got.append(pos[:, t, a, :])
          ref.append(ref_atoms[(sid, score_atom)])
          hit = True
          break
      if hit:
        break
  got = np.stack(got, 1)                                        # (S, n, 3)
  ref = np.stack(ref)
  rs = [kabsch(got[i], ref)[0] for i in range(got.shape[0])]
  print('  %d samples, %s-RMSD %s  best %.3f  mean %.3f'
        % (len(rs), score_atom, ' '.join('%.3f' % r for r in rs), min(rs),
           float(np.mean(rs))))

  if args.write:
    check_output(args.model, batch, out, args.write, case, ref_seq)

  if case['kind'] == 'ligand':
    # NOT `names`: that is the atom-name array `atom_name` closes over, and
    # shadowing it made every ligand run die inside the scorer.
    lig_names, elems, lref = ligand_coords(case)
    # Ligand tokens are the atomised tail of the token axis, one atom each.
    lig_tokens = [t for t in range(len(ref_seq), n_tok)
                  if atom_name(t, 0) and int(np.asarray(
                      fb.predicted_structure_info.atom_mask)[t].sum()) == 1]
    ours = [atom_name(t, 0) for t in lig_tokens]
    # HOW the two sides pair up depends on the model: most featurise a CCD
    # ligand with its real atom names ('C11', 'O11', ...), which pair exactly;
    # rosettafold3's featurisation rewrites them to the ELEMENT ('C', 'O'),
    # leaving only order to match on. Try names, fall back to order, and in the
    # fallback require the element sequences to agree atom for atom -- that is
    # the check that makes pairing-by-order safe rather than assumed.
    if ours == lig_names:
      order = list(range(len(ours)))
    elif ours == elems:
      order = list(range(len(ours)))
      print('  ligand paired BY ORDER (this model names ligand atoms by '
            'element); element sequences agree')
    else:
      print('  LIGAND ATOMS DO NOT PAIR, not scoring the ligand:')
      print('    ours %s' % ours[:20])
      print('    ref names %s' % lig_names[:20])
      print('    ref elems %s' % elems[:20])
      return 1
    lref = lref[order]
    # The pose is only meaningful in the protein's frame: superpose on the
    # polymer atoms, apply that transform to the predicted ligand, compare.
    best = None
    for i in range(got.shape[0]):
      _, r, gm, rm = kabsch(got[i], ref)
      lg = np.stack([pos[i, t, 0, :] for t in lig_tokens])
      lg = (lg - gm) @ r + rm
      d = float(np.sqrt(((lg - lref) ** 2).sum(1).mean()))
      best = d if best is None else min(best, d)
    print('  ligand %s: %d atoms, best in-frame RMSD %.3f'
          % (case['ligand'], len(lig_names), best))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
