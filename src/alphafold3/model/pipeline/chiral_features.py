'''chiral-centre features (RoseTTAFold3's `chiral_feats`)

RF3's diffusion atom encoder adds a chirality term to the atom query:

    Q_L = process_ch(chiral_grads(R_noisy)) + Q_L

where `chiral_grads` is the gradient, w.r.t. the noisy coordinates, of the squared
error between each chiral centre's *improper* dihedral and its ideal tetrahedral
value. It is the one signal in the network that is not reflection-symmetric, so it
is what lets the model tell an L-amino acid from a D one and refine towards ideal
tetrahedral geometry (the same trick as RF2AA's SE(3) vector features).

This module builds the static half of that -- which quadruples of atoms form the
chiral centres, and what the signed ideal angle is -- in numpy at featurisation
time. The gradient itself is computed in-graph; see atom_cross_attention.

Convention (matched to atomworks' AddRF2AAChiralFeatures, which is what RF3 trains
on; gated row-for-row against native RF3 featurised batches for BOTH a protein
(6MRR, 213/213 rows) and an atomised ligand (biotin, 9/9)):
  * a centre is an RDKit tetrahedral stereocentre of the residue's CCD component.
  * with 3 explicit heavy neighbours (i<j<k, the 4th substituent being an implicit
    hydrogen) it emits THREE rows -- (c,i,j,k), (c,i,k,j), (c,j,k,i) -- one per pair
    of neighbours that shares a plane with the centre.
  * the ideal angle is arcsin(1/sqrt(3)) ~ 35.26 deg, SIGNED by the sign of the
    dihedral in the reference conformer. The sign is the whole point: it encodes
    which enantiomer this centre is.
  * with 4 explicit heavy neighbours all four tetrahedral sides are enumerated the
    same way, giving twelve rows (RF3 trains with take_first_chiral_subordering
    false).

Indices are into the FLAT dense-atom layout (token * max_atoms_per_token + slot),
which is the layout ref_structure.positions carries.
'''

from __future__ import annotations

import functools

import numpy as np

# arcsin(1/sqrt(3)): the angle between a face of a regular tetrahedron and the plane
# through two of that face's vertices and the centre. atomworks' _IDEAL_DIHEDRAL_ANGLE.
IDEAL_DIHEDRAL_ANGLE = float(np.arcsin(1.0 / np.sqrt(3.0)))


def _dihedral(a, b, c, d, eps=1e-4):
  '''signed dihedral of the plane pair (abc)/(bcd), atomworks' get_dih convention.

  Kept bit-compatible with atomworks (same eps, same atan2(y+eps, x+eps)) because we
  only take its SIGN, and near-planar centres would otherwise flip.
  '''
  b0, b1, b2 = a - b, c - b, d - c
  b1n = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + eps)
  v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
  w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
  x = np.sum(v * w, axis=-1)
  y = np.sum(np.cross(b1n, v) * w, axis=-1)
  return np.arctan2(y + eps, x + eps)


@functools.lru_cache(maxsize=512)
def _component_centres(res_name: str):
  '''[(centre_atom_name, (neighbour names...))] for one CCD component, cached.

  Cached because it costs an RDKit mol build + stereo perception per residue TYPE,
  and a chain re-uses at most ~20 of them.
  '''
  from rdkit import Chem as rd_chem

  from alphafold3.data.tools import rdkit_utils
  from alphafold3.constants.decoded_ccd import get_ccd

  ccd_cif = get_ccd().get(res_name)
  if not ccd_cif:
    return ()
  mol = None
  for cif in (ccd_cif, _model_conformer_cif(ccd_cif)):
    if cif is None:
      continue
    try:
      mol = rdkit_utils.mol_from_ccd_cif(cif, force_parse=False)
      mol = rdkit_utils.sanitize_mol(mol, sort_alphabetically=False,
                                     remove_hydrogens=True)
      break
    except Exception:
      mol = None
  if mol is None:
    return ()                              # unparseable component: no chiral term
  # includeUnassigned=False: only centres RDKit is confident are stereocentres.
  # force=True re-perceives rather than trusting cached properties on the mol.
  centres = rd_chem.FindMolChiralCenters(mol, force=True, includeUnassigned=False,
                                         useLegacyImplementation=True)
  out = []
  for idx, _chirality in centres:
    atom = mol.GetAtomWithIdx(idx)
    nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(nbrs) not in (3, 4):
      continue
    name = lambda i: mol.GetAtomWithIdx(i).GetProp('atom_name')
    try:
      out.append((name(idx), tuple(name(i) for i in nbrs)))
    except KeyError:
      continue                             # component without atom_name props
  return tuple(out)


def _model_conformer_cif(ccd_cif):
  """The same component with its MODEL coordinates standing in for the ideal ones.

  Some CCD entries have no ideal conformer at all -- every
  `pdbx_model_Cartn_*_ideal` is `?` -- and RDKit needs 3D coordinates to perceive a
  tetrahedral stereocentre, so those components come back with NO centres. DAR
  (D-arginine) is one, which is why 5KX0's residues 19 and 22 got no chirality
  signal while the other nine D-residues did.

  The observed (model) coordinates answer the only question asked of the conformer
  here: WHICH atoms are stereocentres. The SIGN never comes from this molecule -- it
  comes from the batch's own `ref_pos` -- so a model conformer of the wrong
  enantiomer would still give the right restraint.

  Returns None when the component has no model coordinates either.
  """
  ideal = ('_chem_comp_atom.pdbx_model_Cartn_x_ideal',
           '_chem_comp_atom.pdbx_model_Cartn_y_ideal',
           '_chem_comp_atom.pdbx_model_Cartn_z_ideal')
  model = ('_chem_comp_atom.model_Cartn_x',
           '_chem_comp_atom.model_Cartn_y',
           '_chem_comp_atom.model_Cartn_z')
  if not all(k in ccd_cif for k in model):
    return None
  if any('?' in ccd_cif[k] for k in model):
    return None
  patched = dict(ccd_cif)
  for want, have in zip(ideal, model):
    patched[want] = ccd_cif[have]
  return patched


def _plane_pair_keys(centre, bonded, order=None):
  '''atomworks' plane-pair enumeration.

  3 neighbours -> the single tetrahedral side, and its 3 (pair-in-plane, remaining)
  splits, so 3 rows. 4 neighbours -> all four sides split the same way, so 12.

  RF3 trains with `take_first_chiral_subordering: false` (configs/datasets/base.yaml),
  which is the enumerate-everything branch, so there is no take-the-first-side case to
  implement. Only 4-neighbour centres are affected -- every centre on a protein, a
  nucleotide or biotin has an implicit hydrogen and three heavy neighbours.

  `order` maps a neighbour name to its ATOM INDEX, which is what atomworks sorts on.
  Sorting on the NAME instead agrees only when the two orders coincide, which they do
  for most amino acids and did for every centre of the 6MRR batch this was first
  checked against -- but not for biotin's C2, whose neighbours C4/C7/S1 sort to
  indices 15/6/8. That emitted a different plane-pair split, and with it different
  SIGNS, i.e. the wrong enantiomer's restraint. Verified row-for-row against a native
  RF3 featurised biotin batch.
  '''
  import itertools
  key_fn = (lambda n: order[n]) if order else (lambda n: n)
  bonded = sorted(bonded, key=key_fn)      # atomworks sorts to fix i<j
  keys = []
  for side in itertools.combinations(bonded, 3):
    for pair in itertools.combinations(side, 2):
      remaining = next(a for a in side if a not in pair)
      keys.append((centre, *pair, remaining))
  return keys


def _residue_slot_map(batch, mask, num_dense):
  """[(res_name, {atom_name: flat dense index})] per residue, in token order.

  Prefers `token_atoms_layout`, which carries res_name / chain_id / res_id / atom_name
  for every dense slot and is therefore right for polymers and atomised ligands alike.
  Falls back to the polymer `aatype` vocabulary with one residue per token for batches
  built by hand without a layout -- correct for proteins, which is all such batches are.
  """
  layout = batch.get('token_atoms_layout')
  if layout is not None:
    layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout
  if layout is None:
    from alphafold3.constants import residue_names

    names = np.asarray(batch['ref_atom_name_chars'])
    aatype = np.asarray(batch['aatype']).reshape(-1)
    vocab = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP
    out = []
    for t in range(mask.shape[0]):
      if not mask[t].any() or int(aatype[t]) >= len(vocab):
        continue
      slot = {}
      for s in range(num_dense):
        if mask[t, s]:
          nm = ''.join(chr(int(c) + 32) for c in names[t, s]).strip()
          if nm:
            slot[nm] = t * num_dense + s
      out.append((vocab[int(aatype[t])], slot))
    return out

  # dict preserves first-appearance order, so a polymer (one residue per token) keeps
  # the token ordering the previous per-token loop produced.
  groups = {}
  for t in range(mask.shape[0]):
    for s in range(num_dense):
      if not mask[t, s]:
        continue
      atom_name = str(layout.atom_name[t, s])
      if not atom_name:
        continue
      key = (str(layout.chain_id[t, s]), int(layout.res_id[t, s]))
      res_name, slot = groups.setdefault(key, (str(layout.res_name[t, s]), {}))
      slot[atom_name] = t * num_dense + s
  return list(groups.values())


def build_chiral_features(batch, max_centres=None):
  '''chiral_centers (n,4) int32 + chiral_angles (n,) float32 for a featurised batch.

  Reads only what the batch already carries: ref_atom_name_chars gives the atom name
  in each dense slot, aatype the residue type, ref_pos the reference conformer that
  fixes each centre's sign. Returns flat dense-atom indices.

  max_centres pads to a fixed length so the shape is static across sequences of the
  same size. Padding rows are marked by angle == 0 (a real centre is always
  +-IDEAL_DIHEDRAL_ANGLE) and carry atom indices [0,1,2,3] rather than zeros: the
  model masks them out of the loss, but a DEGENERATE quadruple would still make the
  dihedral's gradient NaN, and NaN * 0 is NaN, so the padding has to be geometrically
  harmless as well as masked.
  '''
  mask = np.asarray(batch['ref_mask']).astype(bool)        # (T, D)
  pos = np.asarray(batch['ref_pos'], np.float32)           # (T, D, 3)
  num_token, num_dense = mask.shape[-2], mask.shape[-1]
  flat_pos = pos.reshape(-1, 3)

  # Group dense slots by RESIDUE, via token_atoms_layout. Two things make this the
  # only correct source, and both only bite on ligands:
  #   * the component name. `aatype` is a POLYMER vocabulary, so an atomised ligand
  #     reads back as UNK and _component_centres finds nothing -- biotin, which has
  #     three stereocentres, produced zero chiral rows.
  #   * the grouping. AF3 atomises a ligand into ONE TOKEN PER ATOM, so a centre and
  #     its neighbours live in different tokens; anything scoped to a single token's
  #     dense slots can never assemble a centre. A polymer residue is one token, which
  #     is why this was invisible for proteins.
  residues = _residue_slot_map(batch, mask, num_dense)

  rows = []
  for res_name, slot in residues:
    centres = _component_centres(res_name)
    if not centres:
      continue
    for centre_name, bonded_names in centres:
      if centre_name not in slot or any(b not in slot for b in bonded_names):
        continue                            # atom not present in this residue's layout
      for key in _plane_pair_keys(centre_name, bonded_names, order=slot):
        idxs = [slot[n] for n in key]
        p = flat_pos[idxs]
        sign = np.sign(_dihedral(p[0], p[1], p[2], p[3]))
        if sign == 0:
          continue                          # degenerate reference geometry
        rows.append((idxs, float(sign) * IDEAL_DIHEDRAL_ANGLE))

  if not rows:
    centres_arr = np.zeros((0, 4), np.int32)
    angles_arr = np.zeros((0,), np.float32)
  else:
    centres_arr = np.asarray([r[0] for r in rows], np.int32)
    angles_arr = np.asarray([r[1] for r in rows], np.float32)

  if max_centres is not None:
    n = centres_arr.shape[0]
    if n > max_centres:
      centres_arr, angles_arr = centres_arr[:max_centres], angles_arr[:max_centres]
    elif n < max_centres:
      pad = max_centres - n
      filler = np.tile(np.arange(4, dtype=np.int32)[None, :], (pad, 1))
      centres_arr = np.concatenate([centres_arr, filler], axis=0)
      angles_arr = np.concatenate([angles_arr, np.zeros((pad,), np.float32)], axis=0)
  return centres_arr, angles_arr


def attach_chiral_features(batch, max_centres=None):
  '''add chiral_centers / chiral_angles to a featurised batch, in place.'''
  centres, angles = build_chiral_features(batch, max_centres=max_centres)
  batch['chiral_centers'] = centres
  batch['chiral_angles'] = angles
  return batch


def apply_atomized_element_names(batch):
  '''overwrite ref_atom_name_chars with the ELEMENT symbol on atomized tokens.

  RF3's featuriser (atomworks, which logs "Using element type for atom names of
  atomized tokens") does not give an atomized token its CCD atom name: it gives it
  the element symbol, so biotin's C11/O11/O12 are all just "C"/"O"/"O". Verified
  against a native RF3 featurised biotin batch, where every ligand row decodes to a
  one-character element.

  This matters more than it looks. The atom-name characters are 256 of the 389
  columns of RF3's fused per-atom feature Linear, so feeding CCD names there puts
  two thirds of the input vector off-distribution for EVERY ligand atom -- while
  leaving proteins untouched, because a polymer token is not atomized and keeps its
  real atom names either way.

  RF3-only: every other ported family trains on AF3's CCD atom names.
  '''
  from alphafold3.constants import periodic_table

  layout = batch.get('token_atoms_layout')
  if layout is None:
    return batch
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout

  standard = _standard_residues()
  mask = np.asarray(batch['ref_mask']).astype(bool)
  element = np.asarray(batch['ref_element'])
  chars = np.array(batch['ref_atom_name_chars'], copy=True)
  symbol = _element_symbols(periodic_table)

  num_token, num_dense = mask.shape[-2], mask.shape[-1]
  for t in range(num_token):
    for s in range(num_dense):
      if not mask[t, s] or str(layout.res_name[t, s]) in standard:
        continue
      name = symbol.get(int(element[t, s]), '')
      if not name:
        continue
      chars[t, s] = _pad_name_chars(name)
  batch['ref_atom_name_chars'] = chars
  return batch


def apply_restype_alignment_on_atomized(batch):
  '''give an atomized token an alignment of its OWN restype, not of a GAP.

  AF3 builds `profile` from the MSA, and a ligand has no MSA row, so its column is
  the GAP token: AF3's 31-class vocab is [20 amino acids, UNK=20, GAP=21, nucleics],
  and every ligand token comes out one-hot at 21 while its `aatype` says 20. RF3's
  featuriser (atomworks) instead gives a token with no alignment the one-hot of its
  own restype, so native RF3 sees 20 in BOTH channels -- verified against a native
  featurised biotin batch (restype.argmax == profile.argmax == 20 on all 16 rows).

  The same split runs through `msa` itself, which is the other half of what the MSA
  module reads: our ligand row is 21, native's is 20.

  Neither channel is small. `profile` is 31 of the 449 columns of RF3's s_inputs, and
  s_inputs feeds to_s_init, both z-init projections and the MSA embedder; `msa` is the
  MSA module's own input. A ligand entered the trunk mislabelled through all of them.

  BOTH channels use the SAME test: a column is rewritten only when EVERY alignment row
  in it is a gap, which cannot happen for a polymer because row 0 is the query sequence
  itself. So a real gap inside a real alignment is never touched.

  The profile used to use a different, weaker test -- `profile.argmax != aatype`, i.e.
  "the profile disagrees with the restype, so there must be no alignment". That is true
  for a SINGLE-SEQUENCE MSA, which is all it was ever checked on ("0 mismatched rows on
  a single-sequence 6MRR"), and false the moment a real alignment arrives: a polymer
  column whose dominant symbol is a gap, or simply another residue, legitimately
  disagrees with the query restype. The rule then fired on REAL polymer columns and
  replaced their profile with a one-hot of the query -- discarding the alignment while
  leaving the MSA rows themselves intact, which is a combination the model never saw in
  training. On 1STP with a 2144-row alignment that cost ~4 A of backbone accuracy and
  made rosettafold3 WORSE with an MSA than without one.

  RF3-only: every other ported family trains on AF3's gap convention.
  '''
  from alphafold3.constants import residue_names

  gap_idx = residue_names.POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP[residue_names.GAP]
  aatype = np.asarray(batch['aatype'])
  profile = np.array(batch['profile'], copy=True)
  msa = batch.get('msa')

  if msa is not None:
    # a token with no alignment of its own: gap in EVERY row, including the query
    unaligned = (np.asarray(msa) == gap_idx).all(axis=0) & (aatype != gap_idx)
  else:
    # no MSA to consult -- fall back to the old restype test, which is exact when
    # the alignment is the query alone (the only case reachable without `msa`)
    unaligned = profile.argmax(-1) != aatype

  if unaligned.any():
    rows = np.zeros((int(unaligned.sum()), profile.shape[-1]), profile.dtype)
    rows[np.arange(len(rows)), aatype[unaligned]] = 1.0
    profile[unaligned] = rows
  batch['profile'] = profile

  if msa is not None:
    msa = np.array(msa, copy=True)
    msa[:, unaligned] = aatype[unaligned].astype(msa.dtype)
    batch['msa'] = msa
  return batch


def _pad_name_chars(name):
  '''ascii-32 codes for a 4-character left-aligned atom name (space -> 0).'''
  out = np.zeros((4,), np.int32)
  for i, c in enumerate(name[:4]):
    out[i] = ord(c) - 32
  return out


@functools.lru_cache(maxsize=1)
def _standard_residues():
  from alphafold3.model import features as af3_features
  return frozenset(af3_features._STANDARD_RESIDUES)


@functools.lru_cache(maxsize=1)
def _element_symbols(periodic_table):
  # uppercase, matching the mmCIF _chem_comp_atom.type_symbol convention the
  # atom-name channel is otherwise fed from.
  return {v: k.upper() for k, v in periodic_table.ATOMIC_NUMBER.items()}


def apply_unknown_restype_on_atomized(batch):
  '''an atomised polymer token is UNKNOWN to RF3, not its parent residue type.

  AlphaFold 3 atomises a modified residue but keeps the PARENT restype on every one
  of its tokens: D-aspartate reads back as ASP in `aatype`, in `profile` and in every
  `msa` row. atomworks does not -- it gives an atomised token the unknown restype, and
  a native RF3 batch for 5KX0 has 20 (UNK) in all three channels on all 93 atomised
  tokens, against our 3/14/6 (ASP/PRO/GLU).

  This is the D-amino-acid bug. 5KX0 is a de novo cyclic peptide with 11 D-residues,
  and the parent restype is the one feature that ASSERTS the wrong enantiomer: it
  names the L residue whose backbone the model has seen a million times, and it does
  so in the three channels that dominate s_inputs. The reference conformer and the
  chiral centres both say D -- correctly, and byte-for-byte as native builds them --
  but they are outvoted. Native builds 11/11 D-residues as D; with the parent restype
  we built 2/11.

  Protein tokens only. The evidence here is a protein batch, and AF3's 31-class vocab
  carries ONE nucleotide-unknown where atomworks carries two (UNKNOWN_RNA and
  UNKNOWN_DNA), so which of them a modified base should map to is not something this
  input can settle -- and mapping a modified base to its PARENT is what made three
  ports fold tRNA. Leaving nucleics alone keeps that.

  RF3-only: every other ported family trains on AlphaFold 3's parent-restype
  convention. Boltz-2 wants the unknown restype too, but on a residue it keeps as ONE
  token, and it needs the `is_modified` flag alongside -- see _mark_modified_residues.
  '''
  from alphafold3.constants import residue_names

  layout = batch.get('token_atoms_layout')
  if layout is None:
    return batch
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout

  standard = _standard_residues()
  mask = np.asarray(batch['ref_mask']).astype(bool)
  is_protein = np.asarray(batch['is_protein']).astype(bool).reshape(-1)
  unknown = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP).index(
      residue_names.UNK)

  num_token, num_dense = mask.shape[-2], mask.shape[-1]
  atomized = np.zeros((num_token,), bool)
  for t in range(num_token):
    if not is_protein[t]:
      continue
    for s in range(num_dense):
      if mask[t, s]:
        atomized[t] = str(layout.res_name[t, s]) not in standard
        break
  if not atomized.any():
    return batch

  aatype = np.array(batch['aatype'])
  aatype[atomized] = unknown
  batch['aatype'] = aatype

  profile = np.array(batch['profile'], copy=True)
  profile[atomized] = 0.0
  profile[atomized, unknown] = 1.0
  batch['profile'] = profile

  msa = batch.get('msa')
  if msa is not None:
    msa = np.array(msa, copy=True)
    msa[:, atomized] = unknown
    batch['msa'] = msa
  return batch
