'''AlphaFold 3 fold inputs -> AlphaFold 2 input features.

The point of this file is that one JSON drives every model in the repo. AF3's
`folding_input.Input` is the common front door; AF2 wants a completely different
feature dict, so this is the bridge.

It is deliberately thin, because `AF2Runner` builds most of the dict itself from
`params['seq']`: `update_seq_gamma` writes msa / cluster_profile / target_feat /
aatype, and `update_aatype` writes the atom14/atom37 layout tables. What is left
for a featuriser is the part the sequence cannot imply -- how long the complex
is, where the chain boundaries fall, and the chain-id encodings the multimer
trunk reads.

SCOPE. AF2 is protein-only: no ligands, no DNA or RNA, no covalent bonds. That
is a strict subset of what `folding_input.Input` can express, so a fold input
carrying anything else RAISES here rather than being silently reduced to its
protein chains. A prediction quietly missing its ligand looks exactly like a
prediction that has one.
'''

from __future__ import annotations

import numpy as np

from alphafold3.af2.common import residue_constants as rc

# msa_feat channel layout, from ColabDesign v1 main's prep_input_features:
#   23 one-hot (20 + UNK + GAP + MASK), 1 has_deletion, 1 deletion_value,
#   23 profile, 1 deletion_mean  ->  49
MSA_FEAT_DIM = 49
NUM_ATOM37 = 37
NUM_ATOM14 = 14


def blank_features(L: int, N: int = 1, T: int = 1, eN: int = 1) -> dict:
  '''zero-filled AF2 features, matching v1 main's prep_input_features(L, N, T, eN)

  Every array here is a placeholder that either the runner fills in from the
  sequence parameters (msa_feat, target_feat, aatype) or a featuriser overwrites
  (residue_index, the template block, the chain ids).
  '''
  return {
      'aatype': np.zeros(L, int),
      'target_feat': np.zeros((L, 20)),
      'msa_feat': np.zeros((N, L, MSA_FEAT_DIM)),

      'seq_mask': np.ones(L),
      'msa_mask': np.ones((N, L)),
      'msa_row_mask': np.ones(N),

      'atom14_atom_exists': np.zeros((L, NUM_ATOM14)),
      'atom37_atom_exists': np.zeros((L, NUM_ATOM37)),
      'residx_atom14_to_atom37': np.zeros((L, NUM_ATOM14), int),
      'residx_atom37_to_atom14': np.zeros((L, NUM_ATOM37), int),
      'residue_index': np.arange(L),

      'extra_deletion_value': np.zeros((eN, L)),
      'extra_has_deletion': np.zeros((eN, L)),
      'extra_msa': np.zeros((eN, L), int),
      'extra_msa_mask': np.zeros((eN, L)),
      'extra_msa_row_mask': np.zeros(eN),

      'template_aatype': np.zeros((T, L), int),
      'template_all_atom_mask': np.zeros((T, L, NUM_ATOM37)),
      'template_all_atom_positions': np.zeros((T, L, NUM_ATOM37, 3)),
      'template_mask': np.zeros(T),
      'template_pseudo_beta': np.zeros((T, L, 3)),
      'template_pseudo_beta_mask': np.zeros((T, L)),

      'asym_id': np.zeros(L),
      'sym_id': np.zeros(L),
      'entity_id': np.zeros(L),
      'all_atom_positions': np.zeros((N, NUM_ATOM37, 3)),
  }


def protein_chains(fold_input):
  '''the fold input's protein chains, or a refusal naming what AF2 cannot take.

  AF2 has no representation for a ligand, a nucleotide or an inter-chain bond.
  Dropping them to fold "the protein part" would answer a different question
  than the one asked, and the caller would have no way to tell -- so this raises
  instead. Callers wanting the protein subset must say so by editing the input.
  '''
  from alphafold3.common import folding_input

  rejected = []
  chains = []
  for chain in fold_input.chains:
    if isinstance(chain, folding_input.ProteinChain):
      chains.append(chain)
    else:
      rejected.append(f'{type(chain).__name__} {chain.id!r}')
  if getattr(fold_input, 'bonded_atom_pairs', None):
    rejected.append(f'{len(fold_input.bonded_atom_pairs)} bonded atom pair(s)')
  if rejected:
    raise ValueError(
        'AlphaFold 2 is protein-only and this input carries '
        + ', '.join(rejected)
        + '. AF2 has no representation for these, so folding it would silently '
          'answer a different question. Use an AF3-family model, or remove them '
          'from the input explicitly.')
  if not chains:
    raise ValueError('no protein chains in fold input')
  return chains


def featurise_input(fold_input, num_seq: int = 1, num_templates: int = 1,
                    chain_break: int | None = None,
                    use_msa: bool = True) -> tuple[dict, str]:
  '''-> (AF2 input dict, the concatenated one-letter sequence)

  The sequence comes back alongside because `AF2Runner.predict` takes it
  separately: the runner writes it into `params['seq']` as a peaked one-hot, so
  the same code path serves prediction and design and there is no second,
  prediction-only featurisation to drift.

  `chain_break` sets how residue_index crosses a chain boundary. None restarts
  numbering per chain, which is what AF2-multimer's relative position encoding
  expects (it reads asym_id for the chain identity). An integer reproduces v1
  main's "residue index offset hack" -- every protocol there concatenates chains
  with `prev_end + 50` -- which is what the monomer/ptm models were designed
  with. It is explicit rather than baked because design results depend on it.
  '''
  chains = protein_chains(fold_input)
  seqs = [c.sequence for c in chains]
  lengths = [len(s) for s in seqs]
  L = sum(lengths)

  inputs = blank_features(L, N=num_seq, T=num_templates)

  # entity_id groups chains by IDENTICAL sequence (AF2-multimer's notion of the
  # same molecular entity); sym_id counts the copies within one entity; asym_id
  # is the chain ordinal. All three are 1-based, matching AF2's own featuriser.
  entity_of = {}
  copies_seen = {}
  residue_index, asym_id, sym_id, entity_id = [], [], [], []
  offset = 0
  for i, (chain, seq) in enumerate(zip(chains, seqs)):
    n = len(seq)
    if chain_break is None:
      idx = np.arange(n)
    else:
      idx = np.arange(n) + offset
      offset += n + int(chain_break)
    residue_index.append(idx)
    eid = entity_of.setdefault(seq, len(entity_of) + 1)
    copies_seen[eid] = copies_seen.get(eid, 0) + 1
    asym_id.append(np.full(n, i + 1))
    entity_id.append(np.full(n, eid))
    sym_id.append(np.full(n, copies_seen[eid]))

  inputs['residue_index'] = np.concatenate(residue_index).astype(int)
  inputs['asym_id'] = np.concatenate(asym_id).astype(float)
  inputs['sym_id'] = np.concatenate(sym_id).astype(float)
  inputs['entity_id'] = np.concatenate(entity_id).astype(float)

  if use_msa:
    extra = msa_features(fold_input, L)
    if extra is not None:
      inputs.update(extra)

  return inputs, ''.join(seqs)


def aatype_from_sequence(seq: str) -> np.ndarray:
  '''one-letter sequence -> AF2 aatype indices (unknown residues -> 20).

  AF2's alphabet is `residue_constants.restypes` (ARNDCQEGHILKMFPSTWYV), which
  is NOT AF3's ordering. Anything that maps between the two must go through the
  three-letter code rather than assume the indices line up.
  '''
  order = rc.restype_order
  return np.array([order.get(a, rc.restype_num) for a in seq], dtype=int)


# ----------------------------------------------------------------------- MSA

def parse_a3m(a3m: str, length: int) -> tuple[np.ndarray, np.ndarray]:
  '''an A3M alignment -> (msa, deletion_matrix), both (num_seq, length)

  `msa` is indices into `residue_constants.restypes_with_x_and_gap`, so 20 is X
  and 21 is a gap -- the 22-wide alphabet AF2's msa one-hot uses.

  A3M writes insertions relative to the query in LOWERCASE. They occupy no
  aligned column, so they are dropped from `msa` and counted into
  `deletion_matrix` at the column that follows them, which is what AF2's
  has_deletion / deletion_value channels read.
  '''
  order = rc.restype_order_with_x
  gap = len(rc.restypes_with_x_and_gap) - 1     # 21
  unknown = rc.restype_num                      # 20, i.e. X

  msa, deletions = [], []
  for block in a3m.split('>')[1:]:
    lines = block.split('\n')
    seq = ''.join(lines[1:]).strip()
    if not seq:
      continue
    row = np.full(length, gap, dtype=np.int8)
    dels = np.zeros(length, dtype=np.float32)
    col = pending = 0
    for ch in seq:
      if ch.islower():
        pending += 1
        continue
      if col >= length:
        raise ValueError(
            f'A3M row is longer than the query ({length} aligned columns); '
            'the alignment does not match the sequence it is attached to')
      row[col] = gap if ch == '-' else order.get(ch.upper(), unknown)
      dels[col] = pending
      col, pending = col + 1, 0
    if col != length:
      raise ValueError(
          f'A3M row has {col} aligned columns, query has {length}')
    msa.append(row)
    deletions.append(dels)

  if not msa:
    raise ValueError('no sequences in A3M')
  return np.stack(msa), np.stack(deletions)


def msa_features(fold_input, length: int) -> dict | None:
  '''-> {'msa_extra', 'deletion_matrix_extra'} for the chain's alignment, or None

  The QUERY row is deliberately not included. `AF2Runner` writes row 0 itself
  from `params['seq']`, which is what keeps one code path for prediction and
  design; these are the rows stacked BENEATH it, and they are constants.

  Single chain only. Multi-chain alignments need MSA pairing -- rows matched
  across chains by species, with the unpaired remainder block-diagonal -- and
  guessing at that would produce an alignment that looks right and pairs the
  wrong organisms. Raises instead.
  '''
  chains = protein_chains(fold_input)
  a3ms = [(c, (c.unpaired_msa or '') + (c.paired_msa or '')) for c in chains]
  if not any(text.strip() for _c, text in a3ms):
    return None
  if len(chains) > 1:
    raise NotImplementedError(
        'MSA input is single-chain only for now: a multi-chain alignment needs '
        'MSA pairing (rows matched across chains by species, unpaired remainder '
        'block-diagonal), and an unpaired concatenation would look like an '
        'alignment while pairing the wrong organisms. Fold multi-chain inputs '
        'without an MSA, or use an AF3-family model.')

  chain, text = a3ms[0]
  msa, deletions = parse_a3m(text, length)
  # Row 0 of an A3M is the query itself, which the runner supplies.
  if len(msa) > 1 and ''.join(
      rc.restypes_with_x_and_gap[i] for i in msa[0]) == chain.sequence:
    msa, deletions = msa[1:], deletions[1:]
  return {'msa_extra': msa, 'deletion_matrix_extra': deletions}
