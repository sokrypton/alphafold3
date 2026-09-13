'''AlphaFold 3's featurised batch -> AlphaFold 2 input features.

There is ONE featurisation in this package. AF2 does not run a second pipeline
of its own: `alphafold3.data.featurisation` builds the batch, exactly as it does
for every AF3-family model, and this module re-reads the parts of it AF2 wants.

That is what buys the integration rather than a coexistence. The MSA AF2 folds
with is the MSA the AF3 data pipeline produced, chain pairing and all, so a
complex does not need a second pairing implementation here. The token features
are the ones AF3 assigned. And because `alphafold3.af2.output` sends AF2's
coordinates back through the same batch's atom layouts, the same input produces
comparable outputs on either engine, written by one writer.

SCOPE. AF2 is protein-only: no ligands, no DNA or RNA, no covalent bonds, and
one token per residue. That is a strict subset of what a fold input can express,
so anything else RAISES rather than being silently reduced -- a prediction
quietly missing its ligand looks exactly like a prediction that has one.
'''

from __future__ import annotations

import numpy as np

from .common import residue_constants as rc

NUM_ATOM37 = 37
NUM_ATOM14 = 14
# msa_feat channel layout, from ColabDesign v1 main's prep_input_features:
#   23 one-hot (20 + UNK + GAP + MASK), 1 has_deletion, 1 deletion_value,
#   23 profile, 1 deletion_mean  ->  49
MSA_FEAT_DIM = 49
# AF2's msa one-hot before create_msa_feat pads it: 20 + X + gap.
AF2_MSA_ALPHABET = 22


def _af3_to_af2_aatype() -> np.ndarray:
  '''AF3 residue-type index -> AF2 aatype, derived rather than assumed.

  AF3 orders its 31 polymer types with the twenty amino acids first, then UNK
  and a gap, then the nucleotides. AF2 orders its twenty by one-letter code and
  adds X at 20 and a gap at 21. Those two orderings HAPPEN to agree on all 22
  indices AF2 can represent -- both amino-acid blocks are alphabetical by
  three-letter code -- which is exactly the kind of coincidence that should be
  checked rather than relied on, since a reordering upstream would silently
  rewrite every sequence. So the table is built through the residue NAMES and
  the identity is asserted, not hardcoded.
  '''
  from ..constants import residue_names

  af3 = residue_names.POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP
  gap = len(rc.restypes_with_x_and_gap) - 1          # 21
  table = np.full(len(af3), rc.restype_num, dtype=np.int32)   # default X
  for name, i in af3.items():
    if name == '-':
      table[i] = gap
    elif name == 'UNK':
      table[i] = rc.restype_num
    else:
      one = rc.restype_3to1.get(name)
      # Nucleotides have no AF2 amino-acid code; they stay X and are rejected
      # before they can reach the model anyway (see protein_chains).
      if one is not None:
        table[i] = rc.restype_order[one]

  expected = np.arange(AF2_MSA_ALPHABET, dtype=np.int32)
  if not np.array_equal(table[:AF2_MSA_ALPHABET], expected):
    raise AssertionError(
        'the AlphaFold 3 and AlphaFold 2 residue orderings no longer agree on '
        'the first %d indices; the derived table is %s. This is not a failure '
        'to fix here -- it means every AF2 sequence, MSA row and profile built '
        'from an AF3 batch is being relabelled, so check the conversion.'
        % (AF2_MSA_ALPHABET, table[:AF2_MSA_ALPHABET].tolist()))
  return table


AF3_TO_AF2_AATYPE = _af3_to_af2_aatype()


def _af2_to_hhblits_aatype() -> np.ndarray:
  """our restype id -> HHBLITS id, the alphabet AF2 TEMPLATES are written in.

  `aatype`, `target_feat` and every MSA row use `restypes_with_x_and_gap`
  (ARNDCQEGHILKMFPSTWYVX-). AF2's monomer template pipeline instead writes
  `template_aatype` through `HHBLITS_AA_TO_ID` (ACDEFGHIKLMNPQRSTVWYX-), and
  **17 of the 22 positions differ**. Feeding one where the other is meant
  relabels almost every template residue, and does it silently, because class 0
  is alanine in both -- so a wrong template looks like a plausible one.
  """
  table = np.zeros(len(rc.restypes_with_x_and_gap), dtype=np.int32)
  for i, letter in enumerate(rc.restypes_with_x_and_gap):
    table[i] = rc.HHBLITS_AA_TO_ID[letter]
  if sorted(table.tolist()) != list(range(len(table))):
    raise AssertionError(
        'restype -> HHBLITS is not a permutation of 0..%d: %s'
        % (len(table) - 1, table.tolist()))
  return table


AF2_TO_HHBLITS_AATYPE = _af2_to_hhblits_aatype()

# WHICH ALPHABET EACH CHECKPOINT'S TEMPLATE EMBEDDER WAS TRAINED ON. This is a
# property of the weights, so it is measured, not assumed: fold a target with
# its OWN structure as the template and see which ordering collapses the RMSD
# (dev/oracles/af2_template_check.py). AF2_TEMPLATE_ALPHABET can be overridden
# per run to repeat that A/B.
AF2_TEMPLATE_ALPHABET = {'af2_ptm': 'restype', 'af2_multimer': 'restype'}


def blank_features(L: int, N: int = 1, T: int = 1, eN: int = 1) -> dict:
  '''zero-filled AF2 features, matching v1 main's prep_input_features(L, N, T, eN)

  Placeholders: the runner fills msa_feat / target_feat / aatype from the
  sequence parameters, and `from_af3_batch` overwrites the rest.
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
  '''the fold input's protein chains, or a refusal naming what AF2 cannot take.'''
  from ..common import folding_input

  rejected = []
  chains = []
  for chain in fold_input.chains:
    if isinstance(chain, folding_input.ProteinChain):
      chains.append(chain)
      if getattr(chain, 'ptms', None):
        rejected.append(f'modified residues on chain {chain.id!r}')
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


def num_real_tokens(batch) -> int:
  '''how many of the batch's tokens are real, the rest being bucket padding.

  AF2 has no padding concept -- a masked token would still be folded, and would
  still take up a residue in the output -- so it is given the real tokens only,
  and `alphafold3.af2.output` puts the coordinates back at the padded width.
  '''
  mask = np.asarray(batch.token_features.mask).astype(bool)
  n = int(mask.sum())
  if not mask[:n].all():
    raise ValueError('the batch\'s real tokens are not contiguous; AF2 assumes '
                     'padding is a suffix')
  return n


def template_features(batch, L: int, alphabet: str = 'restype') -> dict | None:
  """AF3 batch templates -> AF2's six `template_*` arrays, or None if there are none.

  Two conversions, both of which are silent when wrong:

  * **the alphabet.** `alphabet='hhblits'` writes `template_aatype` through
    `AF2_TO_HHBLITS_AATYPE`; 'restype' leaves it in the ordering `aatype` and
    the MSA use. Which one the weights expect is a property of the CHECKPOINT,
    not of this code -- see `AF2_TEMPLATE_ALPHABET`.
  * **the atom layout.** AF3 carries a template's coordinates in its own dense
    per-token layout (24 slots, meaning-by-restype); AF2 wants atom37. The
    scatter uses AF3's own `PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37`, and only where
    the mask is set -- that table sends an absent atom to index 0, which is N,
    so scattering unmasked would overwrite every backbone nitrogen.
  """
  from ..model import protein_data_processing as pdp

  t = getattr(batch, 'templates', None)
  if t is None:
    return None
  aat3 = np.asarray(t.aatype)
  if aat3.ndim != 2 or aat3.shape[0] == 0:
    return None
  pos3 = np.asarray(t.atom_positions, np.float32)
  msk3 = np.asarray(t.atom_mask)
  T = aat3.shape[0]

  dense_to_37 = np.asarray(pdp.PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37)
  n_dense = min(msk3.shape[-1], dense_to_37.shape[-1])

  aat = np.zeros((T, L), np.int32)
  pos = np.zeros((T, L, NUM_ATOM37, 3), np.float32)
  msk = np.zeros((T, L, NUM_ATOM37), np.float32)
  present = np.zeros(T, np.float32)

  for ti in range(T):
    a3 = aat3[ti, :L].astype(int)
    af2_aat = AF3_TO_AF2_AATYPE[a3]
    aat[ti] = (AF2_TO_HHBLITS_AATYPE[af2_aat] if alphabet == 'hhblits'
               else af2_aat)
    idx = dense_to_37[a3][:, :n_dense]                      # (L, dense)
    live = msk3[ti, :L, :n_dense].astype(bool)
    rows = np.repeat(np.arange(L)[:, None], n_dense, axis=1)
    pos[ti][rows[live], idx[live]] = pos3[ti, :L, :n_dense][live]
    msk[ti][rows[live], idx[live]] = 1.0
    present[ti] = float(live.any())

  # AF2's pseudo-beta: CB, except glycine which has none and uses CA.
  cb, ca = rc.atom_order['CB'], rc.atom_order['CA']
  is_gly = (AF3_TO_AF2_AATYPE[aat3[:, :L].astype(int)] == rc.restype_order['G'])
  pick = np.where(is_gly, ca, cb)
  gi = pick[..., None, None]
  pb = np.take_along_axis(pos, np.broadcast_to(gi, pos.shape[:2] + (1, 3)),
                          axis=2)[:, :, 0, :]
  pb_mask = np.take_along_axis(msk, pick[..., None], axis=2)[:, :, 0]
  return {
      'template_aatype': aat,
      'template_all_atom_positions': pos,
      'template_all_atom_mask': msk,
      'template_mask': present,
      'template_pseudo_beta': pb,
      'template_pseudo_beta_mask': pb_mask.astype(np.float32),
  }


def from_af3_batch(batch, num_seq: int = 1, num_templates: int = 1,
                   use_msa: bool = True,
                   template_alphabet: str = 'restype') -> tuple[dict, str]:
  '''an AF3 featurised batch -> (AF2 input dict, one-letter sequence)

  The sequence comes back alongside because `AF2Runner.predict` takes it
  separately: the runner writes it into `params['seq']` as a peaked one-hot, so
  one code path serves prediction and design and there is no prediction-only
  featurisation to drift.

  `use_msa` carries the batch's alignment through. On by default, because an MSA
  in the input that the model never sees is the quietest way to report a much
  worse number -- single-sequence AF2 is a different model, not a slightly worse
  one.
  '''
  tok = batch.token_features
  L = num_real_tokens(batch)

  inputs = blank_features(L, N=num_seq, T=num_templates)
  inputs['residue_index'] = np.asarray(tok.residue_index)[:L].astype(int)
  inputs['asym_id'] = np.asarray(tok.asym_id)[:L].astype(float)
  inputs['sym_id'] = np.asarray(tok.sym_id)[:L].astype(float)
  inputs['entity_id'] = np.asarray(tok.entity_id)[:L].astype(float)

  aatype = AF3_TO_AF2_AATYPE[np.asarray(tok.aatype)[:L].astype(int)]
  seq = ''.join(rc.restypes_with_x_and_gap[i] for i in aatype)

  tmpl = template_features(batch, L, alphabet=template_alphabet)
  if tmpl is not None and float(np.sum(tmpl['template_mask'])) > 0:
    # resize the blank slots to however many the batch actually carries
    for k, v in tmpl.items():
      inputs[k] = v

  if use_msa and getattr(batch, 'msa', None) is not None:
    # AF3 pads the MSA to a bucket (16384 rows); num_alignments is how many are
    # real. Everything past it is zeros, and folding those would be folding a
    # column of alanines.
    n = int(np.asarray(batch.msa.num_alignments))
    rows = np.asarray(batch.msa.rows)[:n, :L].astype(int)
    dels = np.asarray(batch.msa.deletion_matrix)[:n, :L].astype(np.float32)

    # Row 0 is the query AF3 inserts, and row 1 is the alignment's OWN query
    # line when it has one, so the query can appear twice. Both are dropped:
    # the runner writes the query itself from params['seq'], which is what keeps
    # prediction and design on one code path.
    #
    # This is NOT a general dedupe. A real alignment legitimately contains
    # further rows identical to the query (1STP has one at index 1403) and they
    # stay, because they are homologs that happen to match, and dropping them
    # would quietly reweight the profile. Only the leading duplicate goes.
    drop = 1
    if len(rows) > 1 and np.array_equal(rows[1], rows[0]):
      drop = 2
    rows, dels = rows[drop:], dels[drop:]

    if len(rows):
      inputs['msa_extra'] = AF3_TO_AF2_AATYPE[rows]
      inputs['deletion_matrix_extra'] = dels

  return inputs, seq


def featurise_input(fold_input, num_seq: int = 1, num_templates: int = 1,
                    use_msa: bool = True, buckets=None, ccd=None,
                    template_alphabet: str = 'restype'):
  '''fold input -> (AF2 input dict, sequence, AF3 batch)

  Convenience for callers that have a fold input rather than a batch. It runs
  the SAME featuriser the AF3-family models run and hands the batch back too,
  because `alphafold3.af2.output` needs it to write the structure.
  '''
  from ..constants import decoded_ccd
  from ..data import featurisation
  from ..model import feat_batch

  protein_chains(fold_input)
  batch_dict = featurisation.featurise_input(
      fold_input=fold_input, ccd=ccd or decoded_ccd.get_ccd(),
      buckets=buckets)[0]
  batch = feat_batch.Batch.from_data_dict(batch_dict)
  inputs, seq = from_af3_batch(batch, num_seq=num_seq,
                               num_templates=num_templates, use_msa=use_msa,
                               template_alphabet=template_alphabet)
  return inputs, seq, batch
