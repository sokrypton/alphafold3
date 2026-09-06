# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Per-model featurisation conventions, applied to a featurised batch.

`global_config.model` decides which forward branches run; a few families also
need the INPUT built their way. Those differences are conventions their weights
were trained under, not preferences -- getting one wrong is silent, and shows up
as a fold that is merely mediocre.

Applied on the finished batch rather than threaded through the featuriser: each
one rewrites a handful of plain index arrays, so the layout machinery upstream
stays untouched and stock AlphaFold 3 is byte-identical to before.

    batch = model_features.apply(batch, model_registry.get('opendde'), ...)
"""

from __future__ import annotations

import numpy as np


def _key_window(batch, policy, k_size=None, prefix=''):
  """Rewrite the atom-attention key window on a FINISHED batch.

  Three models needed three windows and got three near-identical functions. The
  arithmetic is the same in all of them -- each block of `q_size` queries takes
  `k_size` keys centred on it, starting at `block_start - (k_size - q_size)//2`,
  and the same four gather arrays are rewritten -- so what actually differs is
  the END POLICY, and for ESMFold2 the width:

    'slide'  AF3's own (features.py AtomCrossAtt, "Shift subsets with
             out-of-bound indices"): shift an out-of-range window bodily back
             inside the real atom count, so every block keeps k_size valid keys.
    'wrap'   chai-1: the window modulo the padded atom count (`% num_atoms` in
             its traced graph), so the first block's leading keys land in
             padding and are masked out -- its end blocks genuinely see fewer
             real neighbours.
    'pad'    opendde/protenix (rearrange_qk_to_dense_trunk): clip and mask, so
             edge blocks see fewer neighbours AND the ones they see sit at
             DIFFERENT key slots, which misaligns the per-slot pair bias too.

  Rewritten on the finished batch rather than threaded through AF3's
  featurisation: the key gathers are plain index arrays over the flat queries
  layout, so the window is the only thing that has to change and the rest of the
  layout machinery stays untouched.
  """
  q_mask = np.asarray(batch[f'{prefix}token_atoms_to_queries:gather_mask'])
  num_subsets, q_size = np.asarray(
      batch[f'{prefix}token_atoms_to_queries:gather_idxs']).shape
  n_padded = num_subsets * q_size
  kk = f'{prefix}queries_to_keys:gather_idxs'
  if k_size is None:
    k_size = np.asarray(batch[kk]).shape[1]
  else:
    # only when a width is REQUESTED: a tiny input cannot hold it, and the
    # window is then the whole list. Taking this on the width read back from the
    # batch would silently reshape arrays the other two policies keep.
    k_size = min(k_size, n_padded)
  starts = np.arange(num_subsets) * q_size + (q_size - k_size) // 2
  if policy == 'slide':
    starts = np.clip(starts, 0, n_padded - k_size)
  window = starts[:, None] + np.arange(k_size)[None, :]

  flat_mask = q_mask.reshape(-1)
  if policy == 'wrap':
    window = window % n_padded
    keep = flat_mask[window]
  elif policy == 'pad':
    in_bounds = (window >= 0) & (window < n_padded)
    window = np.clip(window, 0, n_padded - 1)
    keep = in_bounds & flat_mask[window]
  elif policy == 'slide':
    keep = flat_mask[window]
  else:
    raise ValueError(f'unknown key-window policy {policy!r}')

  batch[kk] = window.astype(np.asarray(batch[kk]).dtype)
  batch[f'{prefix}queries_to_keys:gather_mask'] = keep
  tok = np.asarray(batch[f'{prefix}tokens_to_queries:gather_idxs']).reshape(-1)
  tk = f'{prefix}tokens_to_keys:gather_idxs'
  batch[tk] = tok[window].astype(np.asarray(batch[tk]).dtype)
  tok_mask = np.asarray(
      batch[f'{prefix}tokens_to_queries:gather_mask']).reshape(-1)[window]
  batch[f'{prefix}tokens_to_keys:gather_mask'] = (
      keep if policy == 'pad' else tok_mask)
  return batch


def _circular_key_window(batch, prefix=''):
  """chai-1's atom-attention key window wraps; AF3's is shifted in-bounds.

  Both models give each block of 32 queries a 128-key window centred on it, and
  both compute the same window: block_start - 48 .. block_start + 79. They only
  differ at the ends. AF3 slides an out-of-range window bodily back inside the
  real atom count, so every block keeps 128 valid keys. chai takes the window
  modulo the padded atom count (`% num_atoms` in its traced graph), so the first
  block's leading 48 keys land in padding and are masked out -- the end blocks
  genuinely see fewer real neighbours.

  Rewritten on the finished batch rather than threaded through AF3's
  featurisation: the three key gathers are plain index arrays over the flat
  queries layout, so the window is the only thing that has to change and the
  rest of the layout machinery stays untouched.
  """
  return _key_window(batch, 'wrap', prefix=prefix)

def _padded_key_window(batch, prefix=''):
  """opendde/protenix zero-PAD the atom-attention key window; AF3 slides it.

  Both give each block of 32 queries a 128-key window centred on it:
  block_start - 48 .. block_start + 79. AF3 then shifts any window that runs off
  either end bodily back inside the real atom count, so every block keeps 128
  valid keys (features.py AtomCrossAtt: "Shift subsets with out-of-bound
  indices"). opendde's rearrange_qk_to_dense_trunk instead pads the key sequence
  with (n_keys - n_queries) // 2 = 48 zeros on the left and the matching amount
  on the right, and masks those slots -- its edge blocks genuinely see fewer
  neighbours, and the neighbours they do see sit at DIFFERENT key slots than
  ours, so the per-slot pair bias is misaligned too.

  Verified against native's own bookkeeping: for 1EHZ its pad_info reads
  k_pad_left 48 / k_pad_right 54, and its v_lm disagreed with ours on 2.5% of
  window entries -- all of them in blocks 0, 1 and the last two, which is exactly
  the set of blocks where sliding and padding differ.

  Only the four edge blocks change, which is a small share of a long chain but a
  large one of a short chain: 4 of 51 query blocks for 1EHZ's 1626 atoms against
  4 of 18 for 6MRR's 574.
  """
  return _key_window(batch, 'pad', prefix=prefix)

def _drop_atoms_by_name(batch, names):
  """Mask out atoms another model's tokenizer does not create.

  AF3 gives the C-terminal residue an OXT; chai-1 does not -- its 6MRR run has
  573 atoms where ours has 574, and the extra one is exactly that OXT (verified
  against chai's own AtomRefPos/AtomNameOneHot dump). One atom in 574 sounds
  ignorable, but it lands in the last token's pooled atom representation and in
  the atom-pair window around it, and the trunk amplifies s_init error several
  fold per pass.

  The atom layout is COMPACTED (real atoms first, 18 full blocks of 32 here), so
  dropping an atom that is not last would renumber every atom after it. This
  only masks: the dense slot keeps its coordinates and simply stops being real,
  which is what every consumer already keys off.
  """
  chars = np.asarray(batch['ref_atom_name_chars'])
  decode = lambda row: ''.join(
      chr(int(c) + 32) if 0 <= c < 64 else '' for c in row).strip()
  ref_mask = np.array(batch['ref_mask'])
  n_atoms_per_token = ref_mask.shape[1]
  drop = {(t, a) for t in range(ref_mask.shape[0])
          for a in range(n_atoms_per_token)
          if ref_mask[t, a] and decode(chars[t, a]) in names}
  if not drop:
    return 0
  flat = {t * n_atoms_per_token + a for t, a in drop}
  for t, a in drop:
    ref_mask[t, a] = False
  batch['ref_mask'] = ref_mask
  if 'pred_dense_atom_mask' in batch:
    pdm = np.array(batch['pred_dense_atom_mask'])
    for t, a in drop:
      pdm[t, a] = False
    batch['pred_dense_atom_mask'] = pdm

  # queries carry the dropped atoms too; clear them before anything gathered
  # THROUGH the queries (the key window) is rebuilt from them.
  q_idxs = np.asarray(batch['token_atoms_to_queries:gather_idxs'])
  q_mask = np.array(batch['token_atoms_to_queries:gather_mask'])
  dead_queries = set()
  for r, c in zip(*np.nonzero(q_mask)):
    if int(q_idxs[r, c]) in flat:
      q_mask[r, c] = False
      dead_queries.add(r * q_idxs.shape[1] + c)
  batch['token_atoms_to_queries:gather_mask'] = q_mask

  for key, dead in (('queries_to_token_atoms', flat),
                    ('token_atoms_to_pseudo_beta', flat),
                    ('queries_to_keys', dead_queries)):
    idxs = np.asarray(batch[f'{key}:gather_idxs'])
    mask = np.array(batch[f'{key}:gather_mask'])
    if key == 'queries_to_token_atoms':
      # this one is indexed BY (token, atom), so drop by position
      for t, a in drop:
        mask[t, a] = False
    else:
      mask &= ~np.isin(idxs, list(dead))
    batch[f'{key}:gather_mask'] = mask
  return len(drop)


def _override_ref_conformers(batch, conformers):
  """Replace ref_pos with another model's reference conformers, by atom name.

  AF3 (and so featurise_spec) takes reference coordinates from the CCD ideal
  values. chai-1 instead ships a pre-generated RDKit cache
  (`conformers_v1.apkl`), and the difference is not cosmetic: rebuilding chai's
  ATOM feature stream from our conformers lands at 40% relative error, while the
  same code with chai's own positions is exact to the bfloat16 floor. The
  feature goes straight into a Linear, so it is not pose-invariant.

  Worth knowing that this asymmetry is deliberate on chai's side and applies to
  POLYMERS ONLY: its tokenizer returns the cached conformer unchanged for
  standard residues but calls `center_random_augment()` -- a random rotation
  plus a random translation -- for ligands and non-standard residues. So a
  ligand's reference pose is fresh noise on every run by design, and only the
  polymer conformers have to be matched.

  `conformers` maps a residue code to (atom_names, positions). Atoms whose names
  are absent (AF3's terminal OXT, which chai's conformers do not carry) keep the
  coordinates they already had.
  """
  from alphafold3.constants import residue_names

  pos = np.array(batch['ref_pos'])
  mask = np.asarray(batch['ref_mask'])
  chars = np.asarray(batch['ref_atom_name_chars'])
  aatype = np.asarray(batch['aatype'])
  decode = lambda row: ''.join(
      chr(int(c) + 32) if 0 <= c < 64 else '' for c in row).strip()

  names_3 = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP
  n_replaced = 0
  for t in range(pos.shape[0]):
    code = names_3[int(aatype[t])] if int(aatype[t]) < len(names_3) else None
    entry = conformers.get(code)
    if entry is None:
      continue
    ref_names, ref_pos = entry
    lookup = {n: i for i, n in enumerate(ref_names)}
    for a in range(pos.shape[1]):
      if not mask[t, a]:
        continue
      j = lookup.get(decode(chars[t, a]))
      if j is not None:
        pos[t, a] = ref_pos[j]
        n_replaced += 1
  batch['ref_pos'] = pos.astype(np.asarray(batch['ref_pos']).dtype)
  return n_replaced


def load_chai_conformers(path):
  """Read the standard-residue conformers exported from chai's apkl cache."""
  z = np.load(path, allow_pickle=True)
  codes = sorted({k.split('/')[0] for k in z.files})
  return {c: ([str(x) for x in z[f'{c}/names']], np.asarray(z[f'{c}/pos']))
          for c in codes}


def _zero_msa(batch):
  """chai-1's single-sequence path: an all-gap MSA, profile and deletion zero.

  Zeroing the PROFILE too, not just the mask: chai's no-MSA path feeds an
  all-gap MSA whose profile and deletion mean are identically zero, whereas AF3's
  featuriser would still hold the query's own profile. Leaving it in adds a term
  chai does not have. (Checked against chai's own captured feature stream from a
  no-MSA run: every row is the GAP class with profile and deletion_mean zero.)
  """
  for key in ('msa_mask', 'profile', 'deletion_mean'):
    batch[key] = np.zeros_like(np.asarray(batch[key]))
  return batch


def _attach_esm(batch, esm):
  """Put ESM2 embeddings on the protein tokens, in order; everything else zero.

  chai-1's TOKEN feature stream is mostly ESM2 -- zeroing it drops the token
  embedding to corr 0.327 of chai's, and a natural protein folds to 5.70 A where
  chai reaches 0.642. `esm` is (num_protein_tokens, 2560) in token order, chai's
  own traced ESM whose raw last hidden state IS the feature verbatim.
  """
  is_protein = np.asarray(batch['is_protein']).astype(bool)
  out = np.zeros(is_protein.shape + (np.asarray(esm).shape[-1],), np.float32)
  idxs = np.flatnonzero(is_protein)
  rows = np.asarray(esm, np.float32)
  if len(idxs) != len(rows):
    raise ValueError(f'esm has {len(rows)} rows but the batch has {len(idxs)} '
                     'protein tokens')
  out[idxs] = rows
  batch['esm_embeddings'] = out
  return batch


def _wide_key_window(batch, k_size, prefix=''):
  """Rebuild the atom-attention key window at a WIDER size than AF3's 128.

  AF3 gives each block of 32 queries 128 keys, which is a +/-48 window. ESMFold2
  attends +/-64, so 128 keys cannot hold its window however they are centred:
  the graph masks to +/-64 by rank, but a query can only see keys the SUBSET
  contains, and the subset was ten short of the window for a typical interior
  query (129 keys wanted, 119 median seen). Silent -- attention over a truncated
  key set is still well-formed attention.

  Rewritten on the finished batch, like _circular_key_window: the key gathers
  are index arrays over the flat queries layout, so the window is the only thing
  that changes.
  """
  return _key_window(batch, 'slide', k_size=k_size, prefix=prefix)

def _attach_lm_pair(batch, lm_pair):
  """ESMFold2's language-model PAIR representation, from alphafold3.model.esm.

  (num_tokens, num_tokens, c_z), built outside the fold from ESM-C's hidden
  states. It has to be in the SAME token order the featuriser laid out, which is
  why it is keyed to the batch rather than to the input JSON.

  ESM-C is a PROTEIN language model, so a pair rep covering only the protein
  tokens is scattered into the full token grid with zeros elsewhere -- the same
  thing `_attach_esm` does for chai-1's ESM2, and the only sensible reading of
  "no language-model information about this token". Note that native ESMFold2
  has no non-protein path at all (its featuriser takes sequences), so a ligand
  or nucleic-acid chain is out of distribution for the model however this term
  is filled.
  """
  lm = np.asarray(lm_pair, np.float32)
  n = np.asarray(batch['token_index']).shape[-1]
  if lm.shape[:2] != (n, n):
    is_protein = np.asarray(batch['is_protein']).astype(bool)
    idxs = np.flatnonzero(is_protein)
    if lm.shape[:2] != (len(idxs), len(idxs)):
      raise ValueError(
          f'lm_pair is {lm.shape[:2]} but the batch has {n} tokens '
          f'({len(idxs)} of them protein)')
    full = np.zeros((n, n, lm.shape[-1]), np.float32)
    full[np.ix_(idxs, idxs)] = lm
    lm = full
  batch['lm_pair'] = lm
  return batch


def apply(batch, spec, *, refeaturise=None, model_dir=None, esm=None,
          has_msa=True, fold_input=None, cyclic=None, lm_pair=None):
  """Apply `spec`'s featurisation conventions to a featurised batch, in place.

  spec is a model_registry.ModelSpec; the work is driven by spec.featurise.
  A model with no entries there (alphafold3, openfold3, intellifold2) passes
  through untouched.

  refeaturise is a zero-argument callable that repeats the featurisation that
  produced `batch`. Only opendde needs it: its structural-token layout is built
  by re-running the featuriser under a capture of the tokenizer's internals.

  model_dir is where this model's weights live; chai-1 reads its own standard
  residue conformers from there.

  cyclic is not a per-model convention at all -- it wraps the shared relative
  position encoding, so every family here honours it. It is applied whether or
  not the model declares anything.
  """
  knobs = spec.featurise
  if cyclic:
    cyclic_period(batch, cyclic, fold_input=fold_input)
  if not knobs:
    return batch

  if knobs.get('modified_as_one_token'):
    if fold_input is None:
      raise ValueError(f'{spec.name} needs fold_input= to find its modified '
                       'residues')
    # The single token is produced at featurisation time
    # (flatten_non_standard_residues=False); what is left is the pair of signals
    # boltz2 reads on it.
    _mark_modified_residues(batch, fold_input, unknown_restype=True)

  if knobs.get('atomized_backbone_bonds'):
    _atomized_backbone_bonds(batch)
  if knobs.get('atomized_unknown_restype'):
    from alphafold3.model.pipeline import chiral_features
    # before restype_alignment, which reads `aatype` to rewrite profile and msa.
    chiral_features.apply_unknown_restype_on_atomized(batch)
  if knobs.get('atomized_element_names'):
    from alphafold3.model.pipeline import chiral_features
    chiral_features.apply_atomized_element_names(batch)
  if knobs.get('restype_alignment'):
    from alphafold3.model.pipeline import chiral_features
    chiral_features.apply_restype_alignment_on_atomized(batch)
  if knobs.get('drop_atoms'):
    _drop_atoms_by_name(batch, knobs['drop_atoms'])
  if knobs.get('std_conformers'):
    if model_dir is None:
      raise ValueError(f'{spec.name} needs model_dir to find its conformers')
    import os
    _override_ref_conformers(
        batch, load_chai_conformers(
            os.path.join(os.path.expanduser(str(model_dir)),
                         knobs['std_conformers'])))
  if knobs.get('atom_keys_subset_size'):
    _wide_key_window(batch, knobs['atom_keys_subset_size'])
  if knobs.get('circular_keys'):
    _circular_key_window(batch)
  if knobs.get('padded_keys'):
    _padded_key_window(batch)
  if knobs.get('chirals'):
    from alphafold3.model.pipeline import chiral_features
    chiral_features.attach_chiral_features(batch)
  if knobs.get('zero_msa_without_alignment') and not has_msa:
    _zero_msa(batch)
  if knobs.get('lm_pair') and lm_pair is not None:
    _attach_lm_pair(batch, lm_pair)
  if knobs.get('esm') and esm is not None:
    _attach_esm(batch, esm)

  if knobs.get('opendde'):
    if refeaturise is None:
      raise ValueError('opendde needs refeaturise= to build its structural '
                       'token layout')
    from alphafold3.model.pipeline import structural_features
    batch = structural_features.attach_structural_batch(
        batch, refeaturise, struct_num_tokens=knobs.get('struct_num_tokens'))
    if knobs.get('padded_keys'):
      # the structural-token layout has its own atom windows, and it is the one
      # the diffusion runs on -- both need the convention.
      _padded_key_window(batch, prefix='struct/')
  return batch


def _modified_residue_positions(fold_input):
  """-> {(asym_id, 1-based residue index)} of every modified polymer residue.

  asym_id is 1-based and assigned in chain order, so chain i is asym_id i + 1.
  """
  positions = set()
  for index, chain in enumerate(fold_input.chains):
    mods = (getattr(chain, 'ptms', None)
            or getattr(chain, 'modifications', None) or ())
    for _code, residue in mods:
      positions.add((index + 1, int(residue)))
  return positions


def _mark_modified_residues(batch, fold_input, unknown_restype=False):
  """Add `is_modified`, and optionally give those tokens the unknown restype.

  Only Boltz-2 reads either, and it needs BOTH: its modified-residue conditioning
  and the unknown restype are the joint signature it was trained to read as "this
  token is a modified residue, take its shape from the reference conformer".
  Measured on phospho-ubiquitin with the residue kept as one token: restype alone
  leaves the phosphate 6.42 A out, the flag alone 6.53 A, and the two together
  3.10 A with a chemically correct 1.61 A OG-P bond. Every other family ignores
  both and keeps AlphaFold 3's convention, the PARENT restype -- which is why
  the restype change is opt-in rather than implied by there being modifications.

  Tokens are found by (chain, residue index) rather than by position: under
  AlphaFold 3's convention a modified residue is ATOMISED into one token per
  atom, so several tokens share one residue index.
  """
  from alphafold3.constants import residue_names

  wanted = _modified_residue_positions(fold_input)
  residue_index = np.asarray(batch['residue_index'])
  asym_id = np.asarray(batch['asym_id'])
  flag = np.zeros(residue_index.shape, np.int32)
  for token in range(residue_index.shape[0]):
    if (int(asym_id[token]), int(residue_index[token])) in wanted:
      flag[token] = 1
  batch['is_modified'] = flag
  if unknown_restype and flag.any():
    unknown = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP).index(
        residue_names.UNK)
    aatype = np.array(batch['aatype'])
    aatype[flag.astype(bool)] = unknown
    batch['aatype'] = aatype
  return batch


def cyclic_period(batch, chains=True, fold_input=None):
  """Wrap the relative-position encoding, so a chain has no N- or C-terminus.

  Not an AlphaFold 3 feature and not specific to any one model: the encoding is
  shared, so every family in this package honours it (see
  featurization.create_relative_encoding). A period of 0 means "not cyclic", so
  leaving this off is byte-identical to before.

  chains=True makes every polymer chain wrap at its own length; a sequence of
  chain ids wraps only those (fold_input maps the ids to asym_ids); a dict gives
  explicit per-chain periods, keyed by asym_id.

  The period is the chain's RESIDUE count, not its token count: a ligand
  contributes many tokens at one residue index, and wrapping on token count
  would misplace every polymer offset in the same chain.
  """
  asym_id = np.asarray(batch['asym_id'])
  residue_index = np.asarray(batch['residue_index'])
  period = np.zeros(asym_id.shape, np.int32)
  def wrap_at_own_length(a):
    selected = asym_id == a
    period[selected] = len(np.unique(residue_index[selected]))

  if chains is True:
    for a in np.unique(asym_id[asym_id > 0]):
      wrap_at_own_length(a)
  elif isinstance(chains, dict):
    for key, value in chains.items():
      period[asym_id == int(key)] = int(value)
  else:
    if fold_input is None:
      raise ValueError('naming cyclic chains by id needs fold_input=')
    # asym_id is 1-based and assigned in chain order.
    by_id = {chain.id: index + 1
             for index, chain in enumerate(fold_input.chains)}
    unknown = set(chains) - set(by_id)
    if unknown:
      raise ValueError(f'no chain(s) {sorted(unknown)} in this input; it has '
                       f'{sorted(by_id)}')
    for chain_id in chains:
      wrap_at_own_length(by_id[chain_id])
  batch['cyclic_period'] = period
  return batch


# Backbone atom pair that links residue i to residue i+1, per polymer class.
_LINK_ATOMS = (('C', 'N'), ("O3'", 'P'))


def _atomized_backbone_bonds(batch):
  """Restore the peptide bonds AlphaFold 3 drops at every atomised residue.

  AF3 extracts inter-residue bonds only where one side is a LIGAND chain
  (inter_chain_bonds.get_polymer_ligand_and_ligand_ligand_bonds), so a residue that
  is atomised INSIDE a polymer chain keeps its own CCD bonds and its declared
  covalent links but loses the backbone bond to each neighbour. AF3 does not miss
  them -- its relative-position encoding still places the atoms on one chain,
  because they share a residue index -- but atomworks emits them, so RF3 was trained
  with them present. On 5KX0, a 26-residue cyclic peptide with 11 atomised
  D-residues, native RF3's token_bonds carries 97 pairs and ours carried 83: every
  one of the 14 missing was a junction bond at an atomised residue.

  Bonds are written into the free (masked-off) rows of the ligand-ligand gather, so
  the batch keeps its shapes. Both directions are written: atomworks' matrix is
  symmetric, and AF3's featurisation gives only one direction.

  Not applied to the head-to-tail bond of a cyclic chain -- see cyclic_period, which
  is how every model in this package is told about that, and which is shared rather
  than RF3-specific.
  """
  layout = batch.get('token_atoms_layout')
  if layout is None:
    return batch
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout

  key = 'tokens_to_ligand_ligand_bonds'
  idxs = np.array(batch[f'{key}:gather_idxs'])
  mask = np.array(batch[f'{key}:gather_mask'])
  order = batch.get('ligand_ligand_bond_order')
  ref_mask = np.asarray(batch['ref_mask']).astype(bool)
  num_token, num_dense = ref_mask.shape[-2], ref_mask.shape[-1]

  standard = _standard_residue_names()
  # residues in token order: (chain, res_id) -> [res_name, {atom_name: token}]
  residues = {}
  for t in range(num_token):
    for s in range(num_dense):
      if not ref_mask[t, s]:
        continue
      name = str(layout.atom_name[t, s])
      if not name:
        continue
      rkey = (str(layout.chain_id[t, s]), int(layout.res_id[t, s]))
      entry = residues.setdefault(rkey, [str(layout.res_name[t, s]), {}])
      entry[1].setdefault(name, t)

  wanted = []
  keys = list(residues)
  for first, second in zip(keys, keys[1:]):
    if first[0] != second[0] or second[1] != first[1] + 1:
      continue                              # different chain, or a numbering gap
    (name_a, atoms_a), (name_b, atoms_b) = residues[first], residues[second]
    if name_a in standard and name_b in standard:
      continue                              # AF3 does not drop this one
    for tail, head in _LINK_ATOMS:
      if tail in atoms_a and head in atoms_b:
        wanted.append((atoms_a[tail], atoms_b[head]))
        break

  present = {(int(a), int(b))
             for (a, b), ok in zip(idxs, mask.all(axis=1)) if ok}
  new = []
  for a, b in wanted:
    for pair in ((a, b), (b, a)):
      if pair not in present:
        present.add(pair)
        new.append(pair)
  if not new:
    return batch

  free = np.flatnonzero(~mask.all(axis=1))
  if free.size < len(new):
    # AF3 sizes this gather from the bond table, so a batch whose only bonds are
    # the ones we are adding arrives with no spare rows. Grow it rather than drop
    # bonds; the bond axis is not a bucketed dimension.
    grow = len(new) - free.size
    idxs = np.concatenate([idxs, np.zeros((grow, 2), idxs.dtype)])
    mask = np.concatenate([mask, np.zeros((grow, 2), mask.dtype)])
    if order is not None:
      order = np.concatenate([np.asarray(order),
                              np.zeros((grow,), np.asarray(order).dtype)])
    free = np.flatnonzero(~mask.all(axis=1))
  rows = free[:len(new)]
  idxs[rows] = np.asarray(new, idxs.dtype)
  mask[rows] = True
  batch[f'{key}:gather_idxs'] = idxs
  batch[f'{key}:gather_mask'] = mask
  if order is not None:
    order = np.array(order)
    order[rows] = 1                          # a backbone link is a single bond
    batch['ligand_ligand_bond_order'] = order
  return batch


def _standard_residue_names():
  from alphafold3.model.pipeline import chiral_features
  return chiral_features._standard_residues()
