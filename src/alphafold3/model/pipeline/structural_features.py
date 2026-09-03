"""OpenDDE structural-token featurization.

OpenDDE re-tokenizes each protein residue into a backbone structural token
(atoms N, CA, C, O, OXT; role protein_bb=1; representative atom CA) and a
sidechain structural token (the remaining atoms; role protein_sc=2;
representative atom CB). Glycine (or any residue with an empty backbone/sidechain
split) stays a single backbone token. See opendde/data/tokenizer.py.

This module post-processes AF3's residue-level token layout (the AtomLayout pair
produced by features.tokenizer) into the structural-token layout, plus the
per-token bookkeeping the StructuralTokenExpander needs (parent residue, role,
twin index, chain-adjacent parents). The resulting AtomLayouts feed the same
features.*.compute_features builders, so the whole structural Batch is built with
the existing machinery.
"""

import numpy as np

from alphafold3.model.atom_layout import atom_layout


PROTEIN_BACKBONE_ATOMS = frozenset(['N', 'CA', 'C', 'O', 'OXT'])
NUCLEIC_BACKBONE_ATOMS = frozenset([
    'P', 'OP1', 'OP2', 'OP3', 'O1P', 'O2P', 'O3P', "O5'", "C5'", "C4'", "O4'",
    "C3'", "O3'", "C2'", "O2'", "C1'", 'O5*', 'C5*', 'C4*', 'O4*', 'C3*', 'O3*',
    'C2*', 'O2*', 'C1*', 'O5T', 'O3T'])
# STRUCTURAL_TOKEN_ROLES: atom=0 protein_bb=1 protein_sc=2 dna_bb=3 dna_base=4
# rna_bb=5 rna_base=6 (opendde/data/tokenizer.py).
ROLE_ATOM = 0
ROLE_PROTEIN_BB = 1
ROLE_PROTEIN_SC = 2
_ROLE = {'protein': (1, 2), 'dna': (3, 4), 'rna': (5, 6)}
NO_TWIN = -1
# standard residues that get split (else per-atom), from opendde/data/constants.py.
_STD = {
    'protein': frozenset(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                          'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                          'THR', 'TRP', 'TYR', 'VAL', 'UNK']),
    'rna': frozenset(['A', 'G', 'C', 'U', 'N']),
    'dna': frozenset(['DA', 'DG', 'DC', 'DT', 'DN']),
}
_BACKBONE = {'protein': PROTEIN_BACKBONE_ATOMS, 'dna': NUCLEIC_BACKBONE_ATOMS,
             'rna': NUCLEIC_BACKBONE_ATOMS}
# representative-atom priority per (mol_type, is_backbone).
_CENTRE = {('protein', True): ['CA', 'N', 'C'], ('protein', False): ['CB'],
           ('dna', True): ["C4'", 'C4*', "C1'", 'C1*'], ('dna', False): ['N1', 'N9', "C1'"],
           ('rna', True): ["C4'", 'C4*', "C1'", 'C1*'], ('rna', False): ['N1', 'N9', "C1'"]}

# A nucleic BASE subtoken's centre depends on the ring system, which the single
# _CENTRE list above cannot express: opendde's
# Tokenizer._base_centre_atom_preferences (data/tokenizer.py) is
#   purine     -> N9, C4, C8, N7, C5
#   pyrimidine -> N1, C2, C6, C5, C4
#   otherwise  -> C1', C1*, N9, N1
# Purines carry an N1 as well (in the six-membered ring), so the plain
# ['N1', 'N9', ...] priority silently picked N1 for every A/G/DA/DG where opendde
# picks N9 -- the wrong representative atom on about half of all nucleotides.
# Invisible to protein (this branch is nucleic-only), which is why every opendde
# gate passed while 1EHZ tRNA folded to 5.6 A against native opendde's 1.1 A.
_PURINE = frozenset(['A', 'G', 'DA', 'DG'])
_PYRIMIDINE = frozenset(['C', 'U', 'DC', 'DT'])


def _base_centre(res_name):
  """opendde's per-residue base-centre preference list."""
  if res_name in _PURINE:
    return ['N9', 'C4', 'C8', 'N7', 'C5']
  if res_name in _PYRIMIDINE:
    return ['N1', 'C2', 'C6', 'C5', 'C4']
  return ["C1'", 'C1*', 'N9', 'N1']


def _mol_type(chain_type):
  """AF3 chain_type string -> {protein, dna, rna, ligand}."""
  ct = (chain_type or '').lower()
  if 'polypeptide' in ct:
    return 'protein'
  if 'deoxyribo' in ct:
    return 'dna'
  if 'ribonucleotide' in ct:
    return 'rna'
  return 'ligand'


def _empty_row(max_atoms):
  return {
      'atom_name': np.full(max_atoms, '', dtype=object),
      'res_id': np.zeros(max_atoms, dtype=np.int64),
      'chain_id': np.full(max_atoms, '', dtype=object),
      'atom_element': np.full(max_atoms, '', dtype=object),
      'res_name': np.full(max_atoms, '', dtype=object),
      'chain_type': np.full(max_atoms, '', dtype=object),
  }


def _fill_row(dst, i, src, j):
  for f in ('atom_name', 'res_id', 'chain_id', 'atom_element', 'res_name', 'chain_type'):
    getattr(dst, f)[i] = getattr(src, f)[j] if getattr(src, f) is not None else (
        0 if f == 'res_id' else '')


def build_structural_layout(all_tokens, all_token_atoms_layout):
  """Residue-level AtomLayouts -> structural-token-level AtomLayouts + bookkeeping.

  Args:
    all_tokens: AtomLayout (n_res,), one representative atom per residue token.
    all_token_atoms_layout: AtomLayout (n_res, max_atoms_per_token).

  Returns:
    dict with:
      all_tokens: AtomLayout (n_struct,)
      all_token_atoms_layout: AtomLayout (n_struct, max_atoms_per_token)
      parent_residue_idx: (n_struct,) int64
      subtoken_role_id: (n_struct,) int64
      twin_token_idx: (n_struct,) int64  (bb<->sc within a residue, -1 if none)
      prev_parent_residue_idx / next_parent_residue_idx: (n_struct,) int64
  """
  n_res, max_atoms = all_token_atoms_layout.shape
  A = all_token_atoms_layout
  bb_set = np.array(sorted(PROTEIN_BACKBONE_ATOMS), dtype=object)

  # Per structural token, the (residue, [atom slot indices]) it draws from.
  rows_atom_layout = []          # list of {field: (max_atoms,) array}
  rows_repr = []                 # list of (residue, atom slot) for representative
  rows_src = []                  # per structural token: (residue, [residue atom slots])
  parent, role, chain_of = [], [], []

  def choose(idxs, names, priority):
    for p in priority:
      for k in idxs:
        if names[k] == p:
          return k
    return idxs[0]

  ct_field = A.chain_type
  res_field = A.res_name
  for r in range(n_res):
    names = A.atom_name[r]
    valid = np.array([bool(n) for n in names])
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
      continue
    mt = _mol_type(ct_field[r][valid_idx[0]] if ct_field is not None else None)
    res_name = (res_field[r][valid_idx[0]] if res_field is not None else '')
    # Standard protein/dna/rna residue with >1 atom -> backbone/sidechain (or
    # backbone/base) split; everything else (ligand, modified residue, single
    # atom) -> one "atom" token (role 0) per atom. See opendde tokenize_structural.
    if mt in _ROLE and res_name in _STD[mt] and valid_idx.size > 1:
      bb_set = _BACKBONE[mt]
      bb_role, child_role = _ROLE[mt]
      is_bb = np.array([names[k] in bb_set for k in range(max_atoms)])
      bb_idx = [k for k in valid_idx if is_bb[k]]
      child_idx = [k for k in valid_idx if not is_bb[k]]
      if not child_idx or not bb_idx:                 # glycine / empty split
        groups = [(bb_role, list(valid_idx), _CENTRE[(mt, True)])]
      else:
        child_centre = (_base_centre(res_name) if mt in ('rna', 'dna')
                        else _CENTRE[(mt, False)])
        groups = [(bb_role, bb_idx, _CENTRE[(mt, True)]),
                  (child_role, child_idx, child_centre)]
    else:
      groups = [(ROLE_ATOM, [k], [names[k]]) for k in valid_idx]
    for role_id, idxs, priority in groups:
      row = _empty_row(max_atoms)
      for slot, k in enumerate(idxs):
        for f in ('atom_name', 'res_id', 'chain_id', 'atom_element', 'res_name', 'chain_type'):
          arr = getattr(A, f)
          if arr is not None:
            row[f][slot] = arr[r][k]
      rows_atom_layout.append(row)
      rows_src.append((r, list(idxs)))
      rep_k = choose(idxs, names, priority)
      rows_repr.append((r, rep_k))
      parent.append(r)
      role.append(role_id)
      chain_of.append(all_tokens.chain_id[r])

  n_struct = len(parent)
  parent = np.array(parent, dtype=np.int64)
  role = np.array(role, dtype=np.int64)

  def stack(field):
    return np.stack([row[field] for row in rows_atom_layout], axis=0)
  struct_atoms = atom_layout.AtomLayout(
      atom_name=stack('atom_name'), res_id=stack('res_id'),
      chain_id=stack('chain_id'), atom_element=stack('atom_element'),
      res_name=stack('res_name'), chain_type=stack('chain_type'))

  def repr_field(field):
    src = getattr(A, field)
    return np.array([getattr(A, field)[r][k] if src is not None else ''
                     for (r, k) in rows_repr],
                    dtype=object if field != 'res_id' else np.int64)
  struct_tokens = atom_layout.AtomLayout(
      atom_name=repr_field('atom_name'), res_id=repr_field('res_id'),
      chain_id=repr_field('chain_id'), atom_element=repr_field('atom_element'),
      res_name=repr_field('res_name'), chain_type=repr_field('chain_type'))

  # twin: bb<->sc of the same residue.
  twin = np.full(n_struct, NO_TWIN, dtype=np.int64)
  for r in range(n_res):
    members = np.where(parent == r)[0]
    if members.size == 2:
      twin[members[0]] = members[1]
      twin[members[1]] = members[0]

  # chain-adjacent parents: prev/next residue in the SAME chain, else -1.
  chain_of = np.array(chain_of, dtype=object)
  res_chain = np.array([all_tokens.chain_id[r] for r in range(n_res)], dtype=object)
  prev_parent = np.full(n_struct, -1, dtype=np.int64)
  next_parent = np.full(n_struct, -1, dtype=np.int64)
  for i in range(n_struct):
    r = parent[i]
    if r - 1 >= 0 and res_chain[r - 1] == res_chain[r]:
      prev_parent[i] = r - 1
    if r + 1 < n_res and res_chain[r + 1] == res_chain[r]:
      next_parent[i] = r + 1

  # Gather to reconstruct the residue-level atom layout from structural coords:
  # residue_atom_gather[r, j] = flat index (t * max_atoms + s) of the structural
  # (token t, slot s) that holds residue r's atom slot j; -1 for padding. Lets the
  # structural diffusion output be scattered back to the residue layout, which the
  # existing get_predicted_structure / save_pdb path already understands.
  residue_atom_gather = np.full((n_res, max_atoms), -1, dtype=np.int64)
  for t, (r, idxs) in enumerate(rows_src):
    for s, j in enumerate(idxs):
      residue_atom_gather[r, j] = t * max_atoms + s

  # residue_rep_token[r] = the structural token that stands for residue r when a
  # per-token quantity has to come back to the residue layout (PAE, PDE and the
  # pTM they feed). A residue expands into several structural subtokens, so this
  # is a choice, not a reconstruction: we take its FIRST, which is the subtoken
  # carrying the residue's own backbone role. Residues with no structural token
  # at all (padding) get 0 and are masked out by seq_mask downstream.
  residue_rep_token = np.zeros((n_res,), dtype=np.int64)
  seen = np.zeros((n_res,), dtype=bool)
  for t, r in enumerate(parent):
    if 0 <= r < n_res and not seen[r]:
      residue_rep_token[r] = t
      seen[r] = True

  return {
      'all_tokens': struct_tokens,
      'all_token_atoms_layout': struct_atoms,
      'parent_residue_idx': parent,
      'subtoken_role_id': role,
      'twin_token_idx': twin,
      'prev_parent_residue_idx': prev_parent,
      'next_parent_residue_idx': next_parent,
      'residue_atom_gather': residue_atom_gather,
      'residue_rep_token': residue_rep_token,
  }


def build_structural_batch(all_tokens, all_token_atoms_layout, *, ccd,
                           chemical_components_data, queries_subset_size,
                           keys_subset_size, padding_shapes, struct_num_tokens,
                           ref_max_modified_date=None, conformer_max_iterations=None,
                           random_seed=0):
  """Build the diffusion-facing structural-token feature set.

  Re-tokenizes the residue-level layouts into structural tokens and drives the SAME
  AF3 feature builders (AtomCrossAtt / TokenFeatures / PredictedStructureInfo /
  PseudoBetaInfo / RefStructure) on them, so the structural Batch is produced with
  the tested machinery. `struct_num_tokens` is the padded token bucket for the
  structural set (>= n_struct); `padding_shapes` supplies num_atoms / other dims.

  Returns a dict of the built feature dataclasses plus the expander bookkeeping
  arrays (parent_residue_idx, subtoken_role_id, twin_token_idx, prev/next_parent).
  MSA/Templates/bonds/frames are the caller's responsibility (the diffusion needs
  none of them; confidence needs frames).
  """
  import dataclasses
  import numpy as np
  from alphafold3.model import features as MF

  info = build_structural_layout(all_tokens, all_token_atoms_layout)
  sat = info['all_tokens']
  satl = info['all_token_atoms_layout']
  pad = dataclasses.replace(padding_shapes, num_tokens=struct_num_tokens)

  atom_cross_att = MF.AtomCrossAtt.compute_features(
      all_token_atoms_layout=satl, queries_subset_size=queries_subset_size,
      keys_subset_size=keys_subset_size, padding_shapes=pad)
  token_features = MF.TokenFeatures.compute_features(
      all_tokens=sat, padding_shapes=pad)
  predicted_structure_info = MF.PredictedStructureInfo.compute_features(
      all_tokens=sat, all_token_atoms_layout=satl, padding_shapes=pad)
  # AF3 re-derives the pseudo-beta atom from each token's own atoms and res_name:
  # CB/CA for protein, and for a NUCLEIC token the base ring atoms C4 (purine) or
  # C2 (pyrimidine). A structural token holds only the backbone OR only the base,
  # so an RNA/DNA BACKBONE token contains neither and falls through to "first
  # valid atom" -- P -- with a warning per token. Protein is unaffected: its
  # backbone token has CA and its sidechain token has CB.
  #
  # We already chose the right representative when the tokens were built
  # (_CENTRE / _base_centre, matching opendde's Tokenizer), and `sat` IS that
  # one-atom-per-token layout. Native opendde agrees: its
  # structural_distogram_rep_atom_mask puts the representative at C4' for a
  # backbone token and N9 for a purine base token. So build the gather from our
  # own representatives instead of letting AF3 guess.
  pseudo_beta_layout = sat.copy_and_pad_to((pad.num_tokens,))
  pseudo_beta_info = MF.PseudoBetaInfo(
      token_atoms_to_pseudo_beta=atom_layout.compute_gather_idxs(
          source_layout=satl, target_layout=pseudo_beta_layout))
  ref_structure, _ = MF.RefStructure.compute_features(
      all_token_atoms_layout=satl, ccd=ccd, padding_shapes=pad,
      chemical_components_data=chemical_components_data,
      random_state=np.random.RandomState(random_seed),
      ref_max_modified_date=ref_max_modified_date,
      conformer_max_iterations=conformer_max_iterations)
  frames = MF.Frames.compute_features(
      all_tokens=sat, all_token_atoms_layout=satl, ref_structure=ref_structure,
      padding_shapes=pad)

  # Pad the expander bookkeeping to the token bucket (pad tokens are inert -- their
  # parent/role point at 0/backbone but token_features.mask masks them out).
  def pad_tok(a, fill=0):
    out = np.full((struct_num_tokens,) + a.shape[1:], fill, dtype=a.dtype)
    out[:a.shape[0]] = a
    return out

  return {
      'atom_cross_att': atom_cross_att,
      'token_features': token_features,
      'predicted_structure_info': predicted_structure_info,
      'pseudo_beta_info': pseudo_beta_info,
      'ref_structure': ref_structure,
      'frames': frames,
      'parent_residue_idx': pad_tok(info['parent_residue_idx']),
      'subtoken_role_id': pad_tok(info['subtoken_role_id']),
      'twin_token_idx': pad_tok(info['twin_token_idx'], fill=NO_TWIN),
      'prev_parent_residue_idx': pad_tok(info['prev_parent_residue_idx'], fill=-1),
      'next_parent_residue_idx': pad_tok(info['next_parent_residue_idx'], fill=-1),
      'n_struct': info['parent_residue_idx'].shape[0],
      # residue-shaped (n_res, max_atoms) -- not padded to the structural bucket.
      'residue_atom_gather': info['residue_atom_gather'],
      'residue_rep_token': info['residue_rep_token'],
  }


def structural_to_residue_positions(struct_atom_positions, residue_atom_gather):
  """Scatter structural-token diffusion coords back to the residue atom layout.

  struct_atom_positions: (..., n_struct_tokens, max_atoms, 3) diffusion output.
  residue_atom_gather: (n_res, max_atoms) int, t*max_atoms+s per residue atom (-1 pad).
  Returns (..., n_res, max_atoms, 3) so the standard get_predicted_structure /
  save_pdb path (which expects the residue layout) can consume an OpenDDE fold.
  """
  import numpy as np
  x = np.asarray(struct_atom_positions)
  lead = x.shape[:-3]
  flat = x.reshape(lead + (-1, 3))                       # (..., n_struct*max_atoms, 3)
  g = np.asarray(residue_atom_gather)
  idx = np.clip(g, 0, flat.shape[-2] - 1).reshape(-1)
  out = flat[..., idx, :].reshape(lead + g.shape + (3,))
  out = np.where((g >= 0)[..., None], out, 0.0)          # zero padded atoms
  return out.astype(np.float32)


def assemble_structural_batch(residue_batch, sb):
  """Combine build_structural_batch() output `sb` with a residue Batch into a full
  feat_batch.Batch on structural tokens.

  The diffusion and confidence heads read only token_features / ref_structure /
  predicted_structure_info / pseudo_beta_info / atom_cross_att / frames -- all supplied
  structural. MSA / templates / bond / convert_model_output are never read on this path,
  so they are carried over from the residue batch unchanged (their residue token dim is
  harmless as long as nothing indexes them). Returns (structural_batch, bookkeeping).
  """
  import dataclasses
  from alphafold3.model import feat_batch

  if not isinstance(residue_batch, feat_batch.Batch):
    residue_batch = feat_batch.Batch.from_data_dict(residue_batch)
  structural = dataclasses.replace(
      residue_batch,
      token_features=sb['token_features'],
      ref_structure=sb['ref_structure'],
      predicted_structure_info=sb['predicted_structure_info'],
      pseudo_beta_info=sb['pseudo_beta_info'],
      atom_cross_att=sb['atom_cross_att'],
      frames=sb['frames'],
  )
  bookkeeping = {k: sb[k] for k in (
      'parent_residue_idx', 'subtoken_role_id', 'twin_token_idx',
      'prev_parent_residue_idx', 'next_parent_residue_idx', 'n_struct',
      'residue_atom_gather', 'residue_rep_token')}
  return structural, bookkeeping


def attach_structural_batch(batch_dict, refeaturise, *,
                            struct_num_tokens=None, pad_multiple=32):
  """OpenDDE: augment a residue-level batch dict with the structural-token batch.

  Re-runs featurisation under a lightweight capture of the AF3 tokenizer / AtomCrossAtt
  / RefStructure internals, builds the structural batch, and attaches its as_data_dict()
  under a 'struct/' prefix plus the expander bookkeeping under 'structbook/'. These extra
  keys flow through the runner jit as ordinary leaves; Model extracts and rebuilds the
  structural Batch on the opendde path.

  struct_num_tokens is the padded structural-token bucket. If None it defaults to n_struct
  rounded up to `pad_multiple` (>= n_struct). For a design where the size never changes,
  pass pad_multiple=1 for the minimal (exact n_struct) layout -- no wasted O(N^2) pair
  memory and no recompile cost to worry about; larger pad_multiple (default 32) keeps
  shapes stable across differently sized inputs. An explicit struct_num_tokens overrides
  both. Returns the augmented dict.
  """
  import numpy as np
  from alphafold3.model import features as MF

  cap = {}
  o_tok, o_aca, o_ref = (MF.tokenizer, MF.AtomCrossAtt.compute_features.__func__,
                         MF.RefStructure.compute_features.__func__)

  def p_tok(flat, **kw):
    at, atl, sti = o_tok(flat, **kw)
    cap['at'], cap['atl'] = at, atl
    return at, atl, sti

  def p_aca(cls, all_token_atoms_layout, queries_subset_size, keys_subset_size,
            padding_shapes):
    cap['qss'], cap['kss'], cap['pad'] = queries_subset_size, keys_subset_size, padding_shapes
    return o_aca(cls, all_token_atoms_layout, queries_subset_size, keys_subset_size,
                 padding_shapes)

  def p_ref(cls, all_token_atoms_layout, ccd, padding_shapes, chemical_components_data,
            random_state, ref_max_modified_date, conformer_max_iterations,
            ligand_ligand_bonds=None):
    cap['ccd'], cap['ccd_data'] = ccd, chemical_components_data
    cap['refmax'], cap['confiter'] = ref_max_modified_date, conformer_max_iterations
    return o_ref(cls, all_token_atoms_layout, ccd, padding_shapes,
                 chemical_components_data, random_state, ref_max_modified_date,
                 conformer_max_iterations, ligand_ligand_bonds)

  MF.tokenizer = p_tok
  MF.AtomCrossAtt.compute_features = classmethod(p_aca)
  MF.RefStructure.compute_features = classmethod(p_ref)
  try:
    # `refeaturise` MUST reproduce the ORIGINAL featurisation exactly -- this
    # re-run is what the structural layout is built from, so if it drops (say)
    # the modifications, a chain carrying a PTM is tokenised here as its
    # unmodified self. The residue-level batch then has 85 tokens
    # (a modified residue is atomised, one token per atom) while
    # residue_atom_gather has 76 rows, and every structural gather is silently
    # off -- it surfaced as an IndexError only because the scorer happened to
    # walk past the end.
    refeaturise()
  finally:
    MF.tokenizer = o_tok
    MF.AtomCrossAtt.compute_features = classmethod(o_aca)
    MF.RefStructure.compute_features = classmethod(o_ref)

  n_struct = build_structural_layout(cap['at'], cap['atl'])['parent_residue_idx'].shape[0]
  if struct_num_tokens is None:
    struct_num_tokens = int(np.ceil(n_struct / pad_multiple) * pad_multiple)
  sb = build_structural_batch(
      cap['at'], cap['atl'], ccd=cap['ccd'], chemical_components_data=cap['ccd_data'],
      queries_subset_size=cap['qss'], keys_subset_size=cap['kss'],
      padding_shapes=cap['pad'], struct_num_tokens=struct_num_tokens,
      ref_max_modified_date=cap['refmax'], conformer_max_iterations=cap['confiter'])
  structural, book = assemble_structural_batch(batch_dict, sb)

  out = dict(batch_dict)
  for k, v in structural.as_data_dict().items():
    out['struct/' + k] = v
  for k in ('parent_residue_idx', 'subtoken_role_id', 'twin_token_idx',
            'prev_parent_residue_idx', 'next_parent_residue_idx'):
    out['structbook/' + k] = book[k]
  # The two residue-shaped gathers are built at the TRUE residue count, but the
  # residue batch they map back into is padded to a token bucket -- so pad them
  # to match, or every consumer broadcasts 76 against 128. A padded row gathers
  # nothing (-1) and stands for token 0, and seq_mask discards it either way.
  n_tokens = np.asarray(batch_dict['aatype']).shape[0]
  gather, rep = book['residue_atom_gather'], book['residue_rep_token']
  pad = n_tokens - gather.shape[0]
  if pad < 0:
    raise ValueError(f'{gather.shape[0]} residues do not fit {n_tokens} tokens')
  out['structbook/residue_atom_gather'] = np.pad(
      gather, ((0, pad), (0, 0)), constant_values=-1)
  out['structbook/residue_rep_token'] = np.pad(rep, (0, pad))
  out['structbook/n_struct'] = np.asarray(book['n_struct'])
  return out


def _gather_struct_feature(x_struct, residue_atom_gather):
  """Scatter a structural per-(token,slot) feature (..., n_struct, max_atoms) to the
  residue layout (..., n_res, max_atoms) using residue_atom_gather (t*max_atoms+s)."""
  import numpy as np
  x = np.asarray(x_struct)
  lead = x.shape[:-2]
  flat = x.reshape(lead + (-1,))                         # (..., n_struct*max_atoms)
  g = np.asarray(residue_atom_gather)
  idx = np.clip(g, 0, flat.shape[-1] - 1).reshape(-1)
  out = flat[..., idx].reshape(lead + g.shape)
  return np.where(g >= 0, out, 0.0)


def opendde_predicted_structure(result, aug, sample=0):
  """Build an all-atom Structure (with pLDDT b-factors) from an OpenDDE fold.

  result: AF3Runner.predict output on an opendde-augmented batch (structural-token
  diffusion_samples + predicted_lddt). aug: the augmented batch dict (carries the
  structbook gather + the residue-level keys). Reconstructs residue-layout coords and
  per-atom pLDDT, then reuses the standard get_predicted_structure. Returns a Structure.
  """
  import numpy as np
  from alphafold3.model import feat_batch
  from alphafold3.model.model import get_predicted_structure

  g = aug['structbook/residue_atom_gather']
  sp = np.asarray(result['diffusion_samples']['atom_positions'])
  sp = sp[sample] if sp.ndim == 4 else sp                # (n_struct, max_atoms, 3)
  res_pos = structural_to_residue_positions(sp, g)       # (n_res, max_atoms, 3)

  res_result = {'diffusion_samples': {'atom_positions': res_pos}}
  pl = result.get('predicted_lddt')
  if pl is not None:
    pl = np.asarray(pl)
    pl = pl[sample] if pl.ndim == 3 else pl              # (n_struct*max_atoms, bins)
    nb = pl.shape[-1]
    centres = (np.arange(nb) + 0.5) * (100.0 / nb)
    p = np.exp(pl - pl.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    exp_lddt = (p * centres).sum(-1)                     # (n_struct*max_atoms,)
    max_atoms = g.shape[1]
    res_result['predicted_lddt'] = _gather_struct_feature(
        exp_lddt.reshape(-1, max_atoms), g)              # (n_res, max_atoms)

  residue_batch = feat_batch.Batch.from_data_dict(
      {k: v for k, v in aug.items() if '/' not in k})
  return get_predicted_structure(res_result, residue_batch)
