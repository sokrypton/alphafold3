"""Crosslinks (af3x's input): each crosslinker against its public chemistry, and the expansion into ligands and bonds."""

import json

from absl.testing import absltest
from alphafold3.common import crosslinks
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.cpp import cif_dict
from alphafold3.data import featurisation
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

_SEQ_A = 'MKTAYIAKQRQISFVKSHFSRQ'   # M1, K2, T3, A4, Y5, K8, S13, K16, S17
_SEQ_B = 'GSKELAAKLGKEAYEK'          # G1, S2, K3, K8, G10, K11, Y14, K16


def _input(links, **extra):
  return folding_input.Input.from_json(json.dumps({
      'dialect': 'alphafold3', 'version': 4, 'name': 'xl', 'modelSeeds': [1],
      'sequences': [{'protein': {'id': c, 'sequence': seq, 'modifications': [], 'unpairedMsa': '',
                                 'pairedMsa': '', 'templates': []}}
                    for c, seq in (('A', _SEQ_A), ('B', _SEQ_B))],
      'crosslinks': links, **extra}))


class DefinitionsTest(absltest.TestCase):

  def test_reagent_is_the_suppliers_formula(self):
    for xl in crosslinks.CROSSLINKERS.values():
      with self.subTest(xl.name):
        self.assertEqual(CalcMolFormula(Chem.MolFromSmiles(xl.reagent)), xl.formula)

  def test_bridge_is_what_the_reagent_leaves(self):
    # the reacted bridge, heavy atoms only, sits inside the reagent (the diazirine carbon has lost its N=N)
    for xl in crosslinks.CROSSLINKERS.values():
      with self.subTest(xl.name):
        bridge = Chem.MolFromSmiles(xl.bridge)
        for atom in bridge.GetAtoms():
          atom.SetAtomMapNum(0)
        # heavy atoms, bonds and charges only: an attachment atom has one hydrogen or bond fewer than in the reagent
        query = Chem.MolFromSmarts(Chem.MolToSmarts(bridge))
        self.assertTrue(Chem.MolFromSmiles(xl.reagent).HasSubstructMatch(query))

  def test_components_file_is_current(self):
    blocks, sites = crosslinks.components()
    self.assertEqual(set(blocks), {xl.code for xl in crosslinks.CROSSLINKERS.values()})
    for xl in crosslinks.CROSSLINKERS.values():
      with self.subTest(xl.name):
        block, site = crosslinks._component(xl)
        self.assertEqual(sites[xl.code], (site[1], site.get(2, site[1])))
        # atoms and bonds as generated (coordinates may move with the RDKit version)
        strip = lambda text: [l.split()[:6] for l in text.splitlines() if l.startswith(xl.code + ' ')]
        self.assertEqual(strip(blocks[xl.code]), strip(block))

  def test_aliases(self):
    self.assertEqual(crosslinks.lookup('sulfo-SDA').name, 'SDA')
    self.assertEqual(crosslinks.lookup('azide-A-DSBSO').name, 'AZIDE-A-DSBSO')
    with self.assertRaisesRegex(ValueError, 'unknown crosslinker'):
      crosslinks.lookup('EDC')


class ExpandTest(absltest.TestCase):

  def test_round_trip_unexpanded(self):
    links = [{'name': 'DSSO', 'residue_pairs': [[['A', 2], ['B', 3]], [['A', 8], ['B', 11]]]}]
    fi = _input(links)
    self.assertEqual(json.loads(fi.to_json())['crosslinks'], links)
    self.assertEqual(folding_input.Input.from_json(fi.to_json()), fi)

  def test_each_pair_a_ligand_and_two_bonds(self):
    fi, dropped = _input([{'name': 'DSSO', 'residue_pairs': [[['A', 2], ['B', 3]], [['A', 1], ['B', 14]]]},
                          {'name': 'SDA', 'residue_pairs': [[['A', 13], ['B', 10]]]}]).expand_crosslinks()
    self.assertEqual(dropped, [])
    self.assertEqual(fi.crosslinks, ())
    self.assertEqual([c.id for c in fi.ligands], ['C', 'D', 'E'])
    self.assertEqual([c.ccd_ids for c in fi.ligands], [('XL-DSSO',), ('XL-DSSO',), ('XL-SDA',)])
    _, sites = crosslinks.components()
    s1, s2 = sites['XL-DSSO']
    d1, d2 = sites['XL-SDA']
    self.assertEqual(list(fi.bonded_atom_pairs), [
        (('A', 2, 'NZ'), ('C', 1, s1)), (('B', 3, 'NZ'), ('C', 1, s2)),
        (('A', 1, 'N'), ('D', 1, s1)), (('B', 14, 'OH'), ('D', 1, s2)),   # M1: the N-terminus
        (('A', 13, 'OG'), ('E', 1, d1)), (('B', 10, 'CA'), ('E', 1, d2)),  # the diazirine end on a glycine
    ])
    self.assertEqual(set(cif_dict.parse_multi_data_cif(fi.user_ccd)), {'XL-DSSO', 'XL-SDA'})

  def test_ligand_ids_follow_the_last_chain(self):
    fi = folding_input.Input.from_json(json.dumps({
        'dialect': 'alphafold3', 'version': 4, 'name': 'xl', 'modelSeeds': [1],
        'sequences': [{'protein': {'id': 'A', 'sequence': _SEQ_A}}, {'protein': {'id': 'C', 'sequence': _SEQ_B}}],
        'crosslinks': [{'name': 'DSS', 'residue_pairs': [[['A', 2], ['C', 3]], [['A', 8], ['C', 8]]]}]}))
    self.assertEqual([c.id for c in fi.expand_crosslinks()[0].chains], ['A', 'C', 'D', 'E'])

  def test_diazirine_end_is_the_last_side_chain_atom(self):
    ccd = chemical_components.Ccd()
    for restype, atom in (('LYS', 'NZ'), ('ARG', 'NH2'), ('PHE', 'CZ'), ('TRP', 'CH2'), ('GLY', 'CA'), ('ALA', 'CB')):
      self.assertEqual(crosslinks.residue_atom(crosslinks.DIAZIRINE, restype, 5, ccd), atom)

  def test_nhs_end_refuses_a_residue_it_cannot_acylate(self):
    with self.assertRaisesRegex(ValueError, 'NHS ester reacts with'):
      _input([{'name': 'DSS', 'residue_pairs': [[['A', 4], ['B', 3]]]}]).expand_crosslinks()   # A4 is ALA

  def test_cdi_is_one_carbon_bonded_twice(self):
    fi, _ = _input([{'name': 'CDI', 'residue_pairs': [[['A', 2], ['B', 3]]]}]).expand_crosslinks()
    (_, a), (_, b) = fi.bonded_atom_pairs
    self.assertEqual(a, b)

  def test_overlapping_pairs(self):
    links = [{'name': 'DSS', 'residue_pairs': [[['A', 2], ['B', 3]], [['A', 2], ['B', 8]], [['A', 8], ['B', 8]]]}]
    fi, dropped = _input(links).expand_crosslinks()
    self.assertEqual(dropped, [('DSS', ('A', 2), ('B', 8))])
    self.assertLen(fi.ligands, 2)
    with self.assertRaisesRegex(ValueError, 'carries two crosslinks'):
      _input(links).expand_crosslinks(remove_overlapping=False)

  def test_combinations(self):
    links = [{'name': 'DSS', 'residue_pairs': [[['A', 2], ['B', 3]], [['A', 8], ['B', 8]]]},
             {'name': 'DSSO', 'residue_pairs': [[['A', 17], ['B', 11]]]}]
    combos = list(crosslinks.combinations(links, 2))
    self.assertLen(combos, 3)
    self.assertEqual(combos[0], ('DSS-A2-B3_DSS-A8-B8',
                                 [{'name': 'DSS', 'residue_pairs': [[['A', 2], ['B', 3]], [['A', 8], ['B', 8]]]}]))

  def test_malformed(self):
    for links, message in (([{'name': 'DSS'}], 'expected'),
                           ([{'name': 'DSS', 'residue_pairs': [[['A', 2]]]}], 'expected'),
                           ([{'name': 'DSS', 'residue_pairs': [[['A', 99], ['B', 3]]]}], 'not a protein residue')):
      with self.subTest(message), self.assertRaisesRegex(ValueError, message):
        _input(links).expand_crosslinks()

  def test_featurises(self):
    # AF3's own featurisation takes the crosslinker as a user-CCD ligand and the bonds as declared
    fi, _ = _input([{'name': 'DSSO', 'residue_pairs': [[['A', 2], ['B', 3]]]},
                    {'name': 'SDA', 'residue_pairs': [[['A', 13], ['B', 10]]]}]).expand_crosslinks()
    batch = featurisation.featurise_input(fi, ccd=chemical_components.Ccd(user_ccd=fi.user_ccd), buckets=None)[0]
    n_bridge_atoms = sum(1 for e in cif_dict.parse_multi_data_cif(fi.user_ccd)['XL-DSSO']['_chem_comp_atom.type_symbol']
                         if e != 'H')
    asym = batch['asym_id']
    self.assertEqual(int((asym == 3).sum()), n_bridge_atoms)   # the ligand is atomised: a token per heavy atom


if __name__ == '__main__':
  absltest.main()
