# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Tests the OpenFold3 → AlphaFold3 residue alphabet remapping.

OF3 and AF3 order the residue classes differently, so every weight matrix whose
input rows are indexed by residue type has to be permuted during conversion.
Protein and DNA indices happen to coincide, which makes a missing permutation
easy to miss: the shapes match and only RNA bases and gaps come out wrong.

The tests feed the converter a *marker* state dict — every weight row holds its
own OF3 row index — so the converted parameters read back as "which OF3 row did
AF3 class c end up pulling from". That is checked against AF3's real featurizers
(`msa_features`, `residue_names`), which makes these behavioural tests rather
than a restatement of the converter's internal tables.

The same file also converts the two OF3 checkpoint layouts - preview-2 and
openbind (OpenFold3 >= 0.5.0) - whose diffusion transformers store the pair
bias differently; `CheckpointVariantTest` and `DiffusionTransformerLayoutTest`
cover telling them apart and mapping each one.

Neither the openfold3 package nor an OF3 checkpoint is required. The OF3
alphabet is transcribed from `openfold3/core/data/resources/residues.py`
(`STANDARD_RESIDUES_WITH_GAP_3`). `CheckpointLayoutTest` optionally validates
the hardcoded block offsets against a real checkpoint when one is available.
"""

import inspect
import os
import re

from absl.testing import absltest
from absl.testing import parameterized
from alphafold3.constants import mmcif_names
from alphafold3.constants import residue_names
from alphafold3.data import msa_features
from alphafold3.model import data_constants
from alphafold3.model import of3_weight_converter as converter
import numpy as np


# openfold3.core.data.resources.residues.STANDARD_RESIDUES_WITH_GAP_3, with
# OF3's 'GAP' spelled the AF3 way ('-') so the two lists are comparable.
_OF3_ALPHABET = (
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'UNK',
    'A', 'G', 'C', 'U', 'N',
    'DA', 'DG', 'DC', 'DT', 'DN',
    '-',
)  # pyformat: disable

_AF3_ALPHABET = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP

_NUM_OF3_RESTYPES = len(_OF3_ALPHABET)  # 32
_NUM_AF3_RESTYPES = len(_AF3_ALPHABET)  # 31

# Block layout of OF3's msa_feat / s_input, from
# openfold3.core.model.feature_embedders.input_embedders:
#   s_input  = [atom_cond(384), restype(32), profile(32), deletion_mean(1)]
#   msa_feat = [restype(32), has_deletion(1), deletion_value(1)]
_ATOM_COND_DIM = 384
_S_INPUT_DIM = _ATOM_COND_DIM + 2 * _NUM_OF3_RESTYPES + 1  # 449
_MSA_FEAT_DIM = _NUM_OF3_RESTYPES + 2  # 34
_C_SINGLE = 384

# AF3's equivalent target_feat, from featurization.create_target_feat:
#   [restype(31), profile(31), deletion_mean(1), atom_cond(384)]
_AF3_TARGET_FEAT_DIM = 2 * _NUM_AF3_RESTYPES + 1 + _ATOM_COND_DIM  # 447

_CHECKPOINT_ENV_VAR = 'OF3_CHECKPOINT'
# Resolved relative to the repository root (this file lives in src/alphafold3/model).
_CHECKPOINT_FALLBACKS = ('../af3_of3/weights/of3-p2-155k.pt',)
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
)


def _marker_weight(num_in: int, c_out: int = 3) -> np.ndarray:
  """PyTorch-layout (out, in) weight whose transpose has row i filled with i."""
  return np.tile(np.arange(num_in, dtype=np.float32)[None, :], (c_out, 1))


def _marker_state_dict(c_out: int = 3) -> dict[str, np.ndarray]:
  """Minimal OF3 state dict covering every residue-indexed weight matrix."""
  return {
      # Required unconditionally by map_evoformer_input_embeddings.
      'layer_norm_z.weight': np.zeros(c_out, dtype=np.float32),
      'layer_norm_z.bias': np.zeros(c_out, dtype=np.float32),
      'linear_z.weight': np.zeros((c_out, c_out), dtype=np.float32),
      'layer_norm_s.weight': np.zeros(c_out, dtype=np.float32),
      'layer_norm_s.bias': np.zeros(c_out, dtype=np.float32),
      'linear_s.weight': np.zeros((c_out, c_out), dtype=np.float32),
      # The residue-indexed matrices under test.
      'msa_module_embedder.linear_m.weight': _marker_weight(
          _MSA_FEAT_DIM, c_out
      ),
      'msa_module_embedder.linear_s_input.weight': _marker_weight(
          _S_INPUT_DIM, c_out
      ),
      'input_embedder.linear_s.weight': _marker_weight(_S_INPUT_DIM, c_out),
      'input_embedder.linear_z_i.weight': _marker_weight(_S_INPUT_DIM, c_out),
      'input_embedder.linear_z_j.weight': _marker_weight(_S_INPUT_DIM, c_out),
  }


def _template_aatype_mapping_statement() -> str:
  """The `for idx, attr in [...]` line mapping OF3 aatype linears, plus its body.

  Used by the two template checks: driving map_template_embedder end to end
  would require a fully-populated fake template pair stack.
  """
  lines = inspect.getsource(converter.map_template_embedder).splitlines()
  idx = next(
      i
      for i, line in enumerate(lines)
      if 'aatype_linear_1' in line and line.lstrip().startswith('for ')
  )
  return '\n'.join(lines[idx : idx + 3])


# Toy diffusion transformer dimensions. Small enough to write out by hand,
# but with every axis a different length so a wrong reshape cannot pass.
_TOY_N_BLOCKS = 6
_TOY_N_SUPER = 3
_TOY_NUM_HEAD = 2
_TOY_HEAD_DIM = 3
_TOY_C_Z = 5
_TOY_C_SINGLE = 4
_TOY_ACT = _TOY_NUM_HEAD * _TOY_HEAD_DIM  # 6

# Real key prefix: the toy state dicts and a real checkpoint are read the same
# way, so the variant checks below apply to both.
_DIFF_TRANSFORMER_PREFIX = 'diffusion_module.diffusion_transformer'

# The two tensors that move between the layouts: preview-2 keeps a pair
# LayerNorm inside every block, openbind keeps one on the transformer. The
# per-block pair *projection* stays per block in both.
_PER_BLOCK_PAIR_NORM = 'attention_pair_bias.layer_norm_z.weight'
_PER_BLOCK_PAIR_PROJ = 'attention_pair_bias.linear_z.weight'


def _toy_diff_transformer_state_dict(
    per_block_pair_norm: bool = True,
) -> dict[str, np.ndarray]:
  """Self-attention blocks of a diffusion transformer, in OF3 (out, in) layout.

  Every tensor is drawn from one seeded generator, so each one is distinct and
  a parameter that ends up in the wrong scope is visible.
  """
  rng = np.random.default_rng(0)

  def w(*shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)

  def adaln(prefix: str) -> dict[str, np.ndarray]:
    return {
        f'{prefix}.layer_norm_s.weight': w(_TOY_C_SINGLE),
        f'{prefix}.linear_g.weight': w(_TOY_ACT, _TOY_C_SINGLE),
        f'{prefix}.linear_g.bias': w(_TOY_ACT),
        f'{prefix}.linear_s.weight': w(_TOY_ACT, _TOY_C_SINGLE),
    }

  sd = {}
  for block in range(_TOY_N_BLOCKS):
    pa = f'{_DIFF_TRANSFORMER_PREFIX}.blocks.{block}.attention_pair_bias'
    pt = f'{_DIFF_TRANSFORMER_PREFIX}.blocks.{block}.conditioned_transition'
    sd.update(adaln(f'{pa}.layer_norm_a'))
    if per_block_pair_norm:
      sd[f'{pa}.layer_norm_z.weight'] = w(_TOY_C_Z)
    sd[f'{pa}.linear_z.weight'] = w(_TOY_NUM_HEAD, _TOY_C_Z)
    for name in ('q', 'k', 'v', 'g', 'o'):
      sd[f'{pa}.mha.linear_{name}.weight'] = w(_TOY_ACT, _TOY_ACT)
    sd[f'{pa}.mha.linear_q.bias'] = w(_TOY_ACT)
    sd[f'{pa}.linear_ada_out.weight'] = w(_TOY_ACT, _TOY_ACT)
    sd[f'{pa}.linear_ada_out.bias'] = w(_TOY_ACT)
    sd.update(adaln(f'{pt}.layer_norm'))
    sd[f'{pt}.swiglu.linear_a.weight'] = w(2 * _TOY_ACT, _TOY_ACT)
    sd[f'{pt}.swiglu.linear_b.weight'] = w(2 * _TOY_ACT, _TOY_ACT)
    sd[f'{pt}.linear_out.weight'] = w(_TOY_ACT, 2 * _TOY_ACT)
    sd[f'{pt}.linear_g.weight'] = w(_TOY_ACT, _TOY_ACT)
    sd[f'{pt}.linear_g.bias'] = w(_TOY_ACT)
  if not per_block_pair_norm:
    sd[f'{_DIFF_TRANSFORMER_PREFIX}.layer_norm_z.weight'] = w(_TOY_C_Z)
  return sd


def _convert_toy_block(sd: dict, **kwargs) -> dict[str, np.ndarray]:
  return converter.convert_diff_self_attn_block(
      sd,
      0,
      _DIFF_TRANSFORMER_PREFIX,
      'transformer',
      _TOY_NUM_HEAD,
      _TOY_HEAD_DIM,
      **kwargs,
  )


def _network_source(filename: str) -> str:
  """Source of a file in model/network, read rather than imported.

  Importing those modules pulls in the CCD pickles that `build_data` writes;
  the rest of this file needs neither them nor a checkpoint.
  """
  path = os.path.join(
      os.path.dirname(os.path.abspath(converter.__file__)), 'network', filename
  )
  with open(path) as f:
    return f.read()


def _converted_source_rows(param_scope: str) -> np.ndarray:
  """Runs the converter on marker weights; returns OF3 source row per AF3 row."""
  params = {}
  converter.map_evoformer_input_embeddings(_marker_state_dict(), params)
  return params[param_scope]['weights'][:, 0]


class AlphabetOrderingTest(parameterized.TestCase):
  """The two alphabets differ exactly where we think they do."""

  def test_alphabet_sizes(self):
    self.assertLen(_OF3_ALPHABET, 32)
    self.assertLen(_AF3_ALPHABET, 31)

  def test_nucleic_classes_disagree_between_codebases(self):
    """RNA is offset by one; GAP and N move entirely."""
    af3 = {name: i for i, name in enumerate(_AF3_ALPHABET)}
    of3 = {name: i for i, name in enumerate(_OF3_ALPHABET)}
    for name in ('A', 'G', 'C', 'U', 'N', '-'):
      with self.subTest(residue=name):
        self.assertNotEqual(af3[name], of3[name])

  def test_protein_and_dna_classes_agree_between_codebases(self):
    """Why a missing permutation is silent for protein and DNA."""
    af3 = {name: i for i, name in enumerate(_AF3_ALPHABET)}
    of3 = {name: i for i, name in enumerate(_OF3_ALPHABET)}
    for name in residue_names.PROTEIN_TYPES_WITH_UNKNOWN + (
        'DA',
        'DG',
        'DC',
        'DT',
    ):
      with self.subTest(residue=name):
        self.assertEqual(af3[name], of3[name])


class TargetFeatRemapTest(parameterized.TestCase):
  """restype/profile rows of target_feat must name the same residue in OF3."""

  @parameterized.named_parameters(
      ('single_activations', 'diffuser/evoformer/single_activations'),
      ('left_single', 'diffuser/evoformer/left_single'),
      ('right_single', 'diffuser/evoformer/right_single'),
      ('extra_msa_target_feat', 'diffuser/evoformer/extra_msa_target_feat'),
  )
  def test_restype_and_profile_rows_are_remapped(self, scope):
    rows = _converted_source_rows(scope)
    self.assertLen(rows, _AF3_TARGET_FEAT_DIM)

    restype_block = rows[:_NUM_AF3_RESTYPES]
    profile_block = rows[_NUM_AF3_RESTYPES : 2 * _NUM_AF3_RESTYPES]
    for af3_idx, af3_name in enumerate(_AF3_ALPHABET):
      with self.subTest(residue=af3_name):
        restype_src = int(restype_block[af3_idx]) - _ATOM_COND_DIM
        profile_src = (
            int(profile_block[af3_idx]) - _ATOM_COND_DIM - _NUM_OF3_RESTYPES
        )
        self.assertEqual(_OF3_ALPHABET[restype_src], af3_name)
        self.assertEqual(_OF3_ALPHABET[profile_src], af3_name)

  def test_non_residue_blocks_are_reordered_but_not_permuted(self):
    rows = _converted_source_rows('diffuser/evoformer/single_activations')
    # deletion_mean is the last OF3 row; atom conditioning is OF3 rows 0-383.
    self.assertEqual(rows[2 * _NUM_AF3_RESTYPES], _S_INPUT_DIM - 1)
    np.testing.assert_array_equal(
        rows[2 * _NUM_AF3_RESTYPES + 1 :], np.arange(_ATOM_COND_DIM)
    )


class MsaFeatRemapTest(parameterized.TestCase):
  """AF3-featurized MSA columns must land on the OF3 row for that residue.

  This is the regression the original bug would fail: linear_m was transposed
  but never permuted, so gaps read OF3's RNA-adenine row and RNA bases read one
  row too high (A→G, G→C, C→U, U→N).
  """

  def setUp(self):
    super().setUp()
    self._rows = _converted_source_rows('diffuser/evoformer/msa_activations')

  def _of3_name_for_msa_index(self, af3_msa_index: int) -> str:
    return _OF3_ALPHABET[int(self._rows[af3_msa_index])]

  def _of3_name_for_char(self, char: str, chain_poly_type: str) -> str:
    msa, _ = msa_features.extract_msa_features(
        msa_sequences=[char], chain_poly_type=chain_poly_type
    )
    return self._of3_name_for_msa_index(msa[0, 0])

  def test_msa_feat_width_is_unchanged(self):
    self.assertLen(self._rows, _MSA_FEAT_DIM)

  def test_deletion_features_are_not_permuted(self):
    """has_deletion / deletion_value are positional in both layouts."""
    np.testing.assert_array_equal(
        self._rows[-2:], [_NUM_OF3_RESTYPES, _NUM_OF3_RESTYPES + 1]
    )

  @parameterized.named_parameters(
      ('protein', mmcif_names.PROTEIN_CHAIN),
      ('rna', mmcif_names.RNA_CHAIN),
      ('dna', mmcif_names.DNA_CHAIN),
  )
  def test_gap_maps_to_of3_gap(self, chain_poly_type):
    self.assertEqual(self._of3_name_for_char('-', chain_poly_type), '-')

  def test_msa_padding_value_maps_to_of3_gap(self):
    """AF3 pads MSA rows with the gap class; padded rows must embed as gaps."""
    self.assertEqual(
        self._of3_name_for_msa_index(data_constants.MSA_GAP_IDX), '-'
    )

  @parameterized.parameters(('A', 'A'), ('G', 'G'), ('C', 'C'), ('U', 'U'))
  def test_rna_bases_map_to_of3_rna_bases(self, char, expected):
    self.assertEqual(
        self._of3_name_for_char(char, mmcif_names.RNA_CHAIN), expected
    )

  @parameterized.parameters(('A', 'DA'), ('G', 'DG'), ('C', 'DC'), ('T', 'DT'))
  def test_dna_bases_map_to_of3_dna_bases(self, char, expected):
    self.assertEqual(
        self._of3_name_for_char(char, mmcif_names.DNA_CHAIN), expected
    )

  @parameterized.named_parameters(
      ('rna', mmcif_names.RNA_CHAIN),
      # AF3 has a single unknown-nucleic MSA class, so unknown DNA also lands on
      # OF3's RNA 'N' rather than 'DN'.
      ('dna', mmcif_names.DNA_CHAIN),
  )
  def test_unknown_nucleic_maps_to_of3_unknown_nucleic(self, chain_poly_type):
    self.assertEqual(self._of3_name_for_char('X', chain_poly_type), 'N')

  def test_protein_residues_map_to_of3_protein_residues(self):
    for one_letter, three_letter in (
        residue_names.PROTEIN_COMMON_ONE_TO_THREE.items()
    ):
      with self.subTest(residue=one_letter):
        self.assertEqual(
            self._of3_name_for_char(one_letter, mmcif_names.PROTEIN_CHAIN),
            three_letter,
        )

  def test_unknown_protein_residue_maps_to_of3_unk(self):
    self.assertEqual(
        self._of3_name_for_char('X', mmcif_names.PROTEIN_CHAIN), 'UNK'
    )

  def test_spare_af3_class_does_not_shadow_a_used_of3_row(self):
    """AF3's MSA one-hot has one class its featurizer never emits."""
    used = set()
    for char_map in (
        msa_features._PROTEIN_TO_ID,  # pylint: disable=protected-access
        msa_features._RNA_TO_ID,  # pylint: disable=protected-access
        msa_features._DNA_TO_ID,  # pylint: disable=protected-access
    ):
      used.update(char_map.values())
    spare = [i for i in range(_MSA_FEAT_DIM - 2) if i not in used]
    self.assertEqual(spare, [_NUM_AF3_RESTYPES])
    self.assertNotIn(
        int(self._rows[_NUM_AF3_RESTYPES]),
        [int(self._rows[i]) for i in sorted(used)],
    )


class OtherResidueIndexedWeightsTest(parameterized.TestCase):
  """The remaining residue-indexed matrices, checked the same way."""

  def test_template_aatype_reorder_names_the_same_residue(self):
    """Templates one-hot the aatype directly, so 32 OF3 rows → 31 AF3 rows."""
    rows = converter._reorder_aatype_weights(  # pylint: disable=protected-access
        _marker_weight(_NUM_OF3_RESTYPES).T
    )[:, 0]

    self.assertLen(rows, _NUM_AF3_RESTYPES)
    for af3_idx, af3_name in enumerate(_AF3_ALPHABET):
      with self.subTest(residue=af3_name):
        self.assertEqual(_OF3_ALPHABET[int(rows[af3_idx])], af3_name)

  def test_template_aatype_call_site_uses_the_reorder(self):
    """Call-site guard: exercising map_template_embedder end to end would need
    a full fake template pair stack, so assert the wiring statically instead."""
    statement = _template_aatype_mapping_statement()
    self.assertIn('_reorder_aatype_weights', statement)

  def test_confidence_head_target_feat_weights_are_remapped(self):
    params = {}
    converter.map_confidence_head(
        {
            'aux_heads.pairformer_embedding.linear_i.weight': _marker_weight(
                _S_INPUT_DIM
            ),
            'aux_heads.pairformer_embedding.linear_j.weight': _marker_weight(
                _S_INPUT_DIM
            ),
        },
        params,
    )

    for name in ('left_target_feat_project', 'right_target_feat_project'):
      rows = params[f'diffuser/confidence_head/~_embed_features/{name}'][
          'weights'
      ][:, 0]
      self.assertLen(rows, _AF3_TARGET_FEAT_DIM)
      for af3_idx, af3_name in enumerate(_AF3_ALPHABET):
        with self.subTest(param=name, residue=af3_name):
          src = int(rows[af3_idx]) - _ATOM_COND_DIM
          self.assertEqual(_OF3_ALPHABET[src], af3_name)

  def test_diffusion_conditioning_features_1d_is_remapped(self):
    """features_1d = [single(384), target_feat(449)] in OF3 order."""
    rows = converter._reorder_features_1d(  # pylint: disable=protected-access
        np.arange(_C_SINGLE + _S_INPUT_DIM, dtype=np.float32),
        c_single=_C_SINGLE,
    )
    # 833, not 831: this block feeds a LayerNorm, so the two classes AF3 lacks
    # must be kept rather than dropped (see the next test).
    self.assertLen(rows, _C_SINGLE + _S_INPUT_DIM)
    # The single embedding passes through untouched...
    np.testing.assert_array_equal(rows[:_C_SINGLE], np.arange(_C_SINGLE))
    # ...and the restype block is permuted.
    for af3_idx, af3_name in enumerate(_AF3_ALPHABET):
      with self.subTest(residue=af3_name):
        src = int(rows[_C_SINGLE + af3_idx]) - _C_SINGLE - _ATOM_COND_DIM
        self.assertEqual(_OF3_ALPHABET[src], af3_name)

  def test_features_1d_keeps_the_class_af3_lacks(self):
    """The unknown-DNA restype and profile columns must NOT be dropped here.

    `single_cond_initial_norm` is a LayerNorm, so a zero input column still
    contributes -mean/std through its trained weights, and the normalisation
    statistics depend on the channel count. Dropping the two columns AF3 has no
    slot for cost ~1.6e-3 relative error in the diffusion single conditioning.
    """
    rows = converter._reorder_features_1d(  # pylint: disable=protected-access
        np.arange(_C_SINGLE + _S_INPUT_DIM, dtype=np.float32),
        c_single=_C_SINGLE,
    ).astype(int)

    unk_dna = _OF3_ALPHABET.index('DN')
    restype_base = _C_SINGLE + _ATOM_COND_DIM
    profile_base = restype_base + _NUM_OF3_RESTYPES
    # Immediately after each permuted 31-class block sits the kept extra class.
    self.assertEqual(
        rows[_C_SINGLE + _NUM_AF3_RESTYPES], restype_base + unk_dna
    )
    self.assertEqual(
        rows[_C_SINGLE + 2 * _NUM_AF3_RESTYPES + 1], profile_base + unk_dna
    )
    # Every OF3 restype/profile channel is therefore represented exactly once.
    kept = set(rows.tolist())
    for base in (restype_base, profile_base):
      for offset in range(_NUM_OF3_RESTYPES):
        with self.subTest(channel=base + offset):
          self.assertIn(base + offset, kept)


class PairEmbeddingIndexConventionTest(parameterized.TestCase):
  """Row (i) vs column (j) projections must not be swapped.

  AF3's own naming is inconsistent between its two pair-embedding sites, so
  linear_i/linear_j cannot be mapped to left/right by name:

    evoformer._seq_pair_embedding:    left[:, None] + right[None]
        -> out[i, j] = left[i] + right[j]
    confidence_head._embed_features:  left(tf) + right(tf)[:, None]
        -> out[i, j] = left[j] + right[i]

  OF3 uses the evoformer convention in both places, so the confidence head has
  to cross them. A swap here transposes the confidence pair embedding, which
  shows up in PAE: it is the only asymmetric confidence output (PDE is
  explicitly symmetrized, pLDDT and experimentally-resolved come from the
  single representation).
  """

  # Each row of s is a distinct one-hot, so out[i, j] identifies which token
  # index each projection was applied to.
  _NUM_TOKENS = 3

  def _single_input(self) -> np.ndarray:
    """Token features occupying only the atom-conditioning block.

    Keeps the residue-alphabet permutation out of this test.
    """
    s = np.zeros((self._NUM_TOKENS, _S_INPUT_DIM), dtype=np.float32)
    for token in range(self._NUM_TOKENS):
      s[token, token] = 1.0
    return s

  def _af3_target_feat(self, s: np.ndarray) -> np.ndarray:
    """The same features in AF3's target_feat block order."""
    tf = np.zeros((self._NUM_TOKENS, _AF3_TARGET_FEAT_DIM), dtype=np.float32)
    atom_cond_start = 2 * _NUM_AF3_RESTYPES + 1
    tf[:, atom_cond_start:] = s[:, :_ATOM_COND_DIM]
    return tf

  def _of3_weights(self, c_out: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Distinguishable (out, in) weights for the i and j projections."""
    w_i = np.zeros((c_out, _S_INPUT_DIM), dtype=np.float32)
    w_j = np.zeros((c_out, _S_INPUT_DIM), dtype=np.float32)
    w_i[0, :] = 1.0  # channel 0 reports the token index it saw...
    w_j[1, :] = 1.0  # ...channel 1 likewise, for the other projection.
    w_i[0, :] *= np.arange(_S_INPUT_DIM) + 1
    w_j[1, :] *= np.arange(_S_INPUT_DIM) + 1
    return w_i, w_j

  def test_confidence_head_crosses_i_and_j(self):
    s = self._single_input()
    w_i, w_j = self._of3_weights()

    # OF3: zij = zij + linear_i(s.unsqueeze(-2)) + linear_j(s.unsqueeze(-3))
    of3 = (s @ w_i.T)[:, None, :] + (s @ w_j.T)[None, :, :]

    params = {}
    converter.map_confidence_head({
        'aux_heads.pairformer_embedding.linear_i.weight': w_i,
        'aux_heads.pairformer_embedding.linear_j.weight': w_j,
    }, params)
    scope = 'diffuser/confidence_head/~_embed_features'
    left = params[f'{scope}/left_target_feat_project']['weights']
    right = params[f'{scope}/right_target_feat_project']['weights']

    # AF3: out = left(tf); out += right(tf)[:, None]
    tf = self._af3_target_feat(s)
    af3 = (tf @ left) + (tf @ right)[:, None, :]

    np.testing.assert_allclose(af3, of3)

  def test_evoformer_does_not_cross_i_and_j(self):
    """The trunk uses the opposite naming, so there the direct mapping is right."""
    s = self._single_input()
    w_i, w_j = self._of3_weights()

    # OF3: z = emb_i[..., None, :] + emb_j[..., None, :, :]
    of3 = (s @ w_i.T)[:, None, :] + (s @ w_j.T)[None, :, :]

    params = {}
    sd = _marker_state_dict(c_out=2)
    sd['input_embedder.linear_z_i.weight'] = w_i
    sd['input_embedder.linear_z_j.weight'] = w_j
    converter.map_evoformer_input_embeddings(sd, params)
    left = params['diffuser/evoformer/left_single']['weights']
    right = params['diffuser/evoformer/right_single']['weights']

    # AF3: left_single[:, None] + right_single[None]
    tf = self._af3_target_feat(s)
    af3 = (tf @ left)[:, None, :] + (tf @ right)[None, :, :]

    np.testing.assert_allclose(af3, of3)

  def test_template_aatype_indices_are_crossed(self):
    """AF3 index 2 is the column (j) projection, 3 the row (i) projection."""
    idx_for = {
        attr: int(idx)
        for idx, attr in re.findall(
            r'\((\d+), \'(aatype_linear_\d)\'\)',
            _template_aatype_mapping_statement(),
        )
    }
    self.assertEqual(idx_for, {'aatype_linear_1': 3, 'aatype_linear_2': 2})


class CheckpointVariantTest(parameterized.TestCase):
  """Telling openbind (OpenFold3 >= 0.5.0) and preview-2 checkpoints apart.

  Getting this backwards is caught, but only once the model runs: the two
  layouts put the diffusion transformer's pair bias in different parameter
  scopes, so the run stops on a missing parameter. The signature is therefore
  a single tensor that only one of the two layouts has, rather than a version
  string.
  """

  _SHARED_PAIR_NORM = f'{_DIFF_TRANSFORMER_PREFIX}.layer_norm_z.weight'
  _BLOCK_PAIR_NORM = f'{_DIFF_TRANSFORMER_PREFIX}.blocks.0.{_PER_BLOCK_PAIR_NORM}'

  def test_shared_pair_layer_norm_reads_as_openbind(self):
    sd = {self._SHARED_PAIR_NORM: np.zeros(_TOY_C_Z, dtype=np.float32)}
    self.assertTrue(converter.is_openbind_checkpoint(sd))
    self.assertEqual(converter.checkpoint_variant(sd), 'openbind')

  def test_per_block_pair_layer_norm_reads_as_preview2(self):
    sd = {self._BLOCK_PAIR_NORM: np.zeros(_TOY_C_Z, dtype=np.float32)}
    self.assertFalse(converter.is_openbind_checkpoint(sd))
    self.assertEqual(converter.checkpoint_variant(sd), 'p2')

  @parameterized.named_parameters(
      ('openbind', False, 'openbind'),
      ('preview2', True, 'p2'),
  )
  def test_toy_state_dicts_are_recognised(self, per_block_pair_norm, expected):
    sd = _toy_diff_transformer_state_dict(per_block_pair_norm)
    self.assertEqual(converter.checkpoint_variant(sd), expected)

  def test_marker_round_trips_through_the_params_directory(self):
    """A run given only --model_dir has to recover the variant from disk."""
    params_dir = self.create_tempdir().full_path
    self.assertIsNone(converter.read_variant_marker(params_dir))

    sd = _toy_diff_transformer_state_dict(per_block_pair_norm=False)
    self.assertEqual(converter.write_variant_marker(params_dir, sd), 'openbind')
    self.assertEqual(converter.read_variant_marker(params_dir), 'openbind')


class DiffusionTransformerLayoutTest(parameterized.TestCase):
  """Mapping each layout's diffusion transformer pair bias.

  preview-2 runs a pair LayerNorm inside every block, so its scale and the
  pair-logits projection are stacked with the block parameters. openbind runs
  the LayerNorm once, which is stock AF3's own layout: the scale hangs off the
  transformer and AF3 gives each super block one Linear emitting the pair
  logits of all its blocks at once.
  """

  def test_openbind_block_drops_only_the_pair_bias_parameters(self):
    sd = _toy_diff_transformer_state_dict()

    with_pair = _convert_toy_block(sd)
    without_pair = _convert_toy_block(sd, per_block_pair_bias=False)

    self.assertEqual(
        set(with_pair) - set(without_pair),
        {'pair_input_layer_norm/scale', 'pair_logits_projection/weights'},
    )
    self.assertEmpty(set(without_pair) - set(with_pair))
    for key, value in without_pair.items():
      with self.subTest(param=key):
        np.testing.assert_array_equal(value, with_pair[key])

  def test_preview2_block_carries_the_pair_bias_parameters(self):
    sd = _toy_diff_transformer_state_dict()
    pa = f'{_DIFF_TRANSFORMER_PREFIX}.blocks.0.attention_pair_bias'

    converted = _convert_toy_block(sd)

    np.testing.assert_array_equal(
        converted['pair_input_layer_norm/scale'],
        sd[f'{pa}.layer_norm_z.weight'],
    )
    np.testing.assert_array_equal(
        converted['pair_logits_projection/weights'],
        sd[f'{pa}.linear_z.weight'].T,
    )

  def test_openbind_checkpoint_has_no_per_block_pair_layer_norm_to_read(self):
    """Asking for the preview-2 layout has to fail, not silently drop weights."""
    sd = _toy_diff_transformer_state_dict(per_block_pair_norm=False)

    self.assertIsNotNone(_convert_toy_block(sd, per_block_pair_bias=False))
    with self.assertRaises(KeyError):
      _convert_toy_block(sd)

  def test_pair_logits_are_grouped_by_super_block(self):
    sd = _toy_diff_transformer_state_dict(per_block_pair_norm=False)
    super_size = _TOY_N_BLOCKS // _TOY_N_SUPER

    grouped = converter._pair_logits_super_blocks(  # pylint: disable=protected-access
        sd, _DIFF_TRANSFORMER_PREFIX, _TOY_N_BLOCKS, _TOY_N_SUPER
    )

    self.assertEqual(
        grouped.shape,
        (_TOY_N_SUPER, _TOY_C_Z, super_size, _TOY_NUM_HEAD),
    )
    for super_block in range(_TOY_N_SUPER):
      for block in range(super_size):
        with self.subTest(super_block=super_block, block=block):
          index = super_block * super_size + block
          np.testing.assert_array_equal(
              grouped[super_block, :, block, :],
              sd[f'{_DIFF_TRANSFORMER_PREFIX}.blocks.{index}.{_PER_BLOCK_PAIR_PROJ}'].T,
          )


class Preview2ModelBranchTest(parameterized.TestCase):
  """The two preview-2-only model branches must stay gated on of3_openbind.

  Both compensate for something openbind reverted to AF3's own convention. The
  diffusion transformer's per-block pair LayerNorm changes which parameter
  scopes are read, so leaving it on for openbind weights raises. The column
  attention pair bias axis swap changes no shape at all, so leaving that one on
  corrupts the fold silently - which is why it is gated here rather than left
  to fail somewhere. Every *other* of3_weights branch (model.py,
  atom_cross_attention.py, evoformer.py, diffusion_head.py) applies to both
  checkpoints and must not be gated.

  Driving the modules would need a built model, so check the source: these two
  files hold one of3_weights branch each, which has to name of3_openbind.
  """

  @parameterized.named_parameters(
      ('diffusion_transformer_pair_layer_norm', 'diffusion_transformer.py'),
      ('column_attention_pair_bias', 'modules.py'),
  )
  def test_the_of3_weights_branch_is_gated_on_of3_openbind(self, filename):
    source = _network_source(filename)

    self.assertEqual(source.count('of3_weights'), 1)
    self.assertEqual(source.count('of3_openbind'), 1)


class CheckpointLayoutTest(parameterized.TestCase):
  """Validates the hardcoded block offsets against a real OF3 checkpoint.

  Skipped unless a checkpoint is found. Point ${OF3_CHECKPOINT} at an OF3
  `.pt` file to run it.
  """

  @classmethod
  def _checkpoint_path(cls) -> str | None:
    candidates = [os.environ.get(_CHECKPOINT_ENV_VAR)]
    candidates.extend(
        os.path.join(_REPO_ROOT, p) for p in _CHECKPOINT_FALLBACKS
    )
    return next((p for p in candidates if p and os.path.exists(p)), None)

  def setUp(self):
    super().setUp()
    path = self._checkpoint_path()
    if path is None:
      self.skipTest(
          f'No OF3 checkpoint found; set ${_CHECKPOINT_ENV_VAR} to run.'
      )
    try:
      import torch  # pylint: disable=g-import-not-at-top
    except ImportError:
      self.skipTest('PyTorch not installed.')
    state_dict = torch.load(
        path, map_location='cpu', weights_only=True, mmap=True
    )
    self._sd = state_dict.get('state_dict', state_dict)

  @parameterized.named_parameters(
      ('input_embedder_s', 'input_embedder.linear_s.weight', _S_INPUT_DIM),
      ('input_embedder_z_i', 'input_embedder.linear_z_i.weight', _S_INPUT_DIM),
      ('input_embedder_z_j', 'input_embedder.linear_z_j.weight', _S_INPUT_DIM),
      (
          'msa_s_input',
          'msa_module_embedder.linear_s_input.weight',
          _S_INPUT_DIM,
      ),
      ('msa_m', 'msa_module_embedder.linear_m.weight', _MSA_FEAT_DIM),
      (
          'confidence_i',
          'aux_heads.pairformer_embedding.linear_i.weight',
          _S_INPUT_DIM,
      ),
      (
          'template_aatype_1',
          'template_embedder.template_pair_embedder.aatype_linear_1.weight',
          _NUM_OF3_RESTYPES,
      ),
      (
          'diffusion_conditioning_s',
          'diffusion_module.diffusion_conditioning.linear_s.weight',
          _C_SINGLE + _S_INPUT_DIM,
      ),
  )
  def test_input_dim_matches_assumed_layout(self, key, expected_in_dim):
    self.assertIn(key, self._sd)
    self.assertEqual(self._sd[key].shape[-1], expected_in_dim)

  def test_exactly_one_pair_layer_norm_layout_is_present(self):
    """The variant signature is only unambiguous if the layouts are exclusive."""
    shared = f'{_DIFF_TRANSFORMER_PREFIX}.layer_norm_z.weight'
    per_block = f'{_DIFF_TRANSFORMER_PREFIX}.blocks.0.{_PER_BLOCK_PAIR_NORM}'

    self.assertNotEqual(shared in self._sd, per_block in self._sd)
    self.assertEqual(
        converter.is_openbind_checkpoint(self._sd), shared in self._sd
    )


if __name__ == '__main__':
  absltest.main()
