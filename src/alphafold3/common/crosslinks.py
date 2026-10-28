"""Chemical crosslinks as explicit ligands: the `crosslinks` input of af3x.

af3x (KosinskiLab/af3x; Obarska-Kosinska & Kosinski, bioRxiv
10.1101/2024.12.03.626671) models a crosslinking mass-spec restraint by adding
the crosslinker itself as a ligand, covalently bonded to the two residues it
joined, rather than as a distance restraint. This module takes the same input,

    "crosslinks": [{"name": "DSSO", "residue_pairs": [[["A", 53], ["B", 66]], ...]}]

and is an independent implementation of it: the format and the idea are af3x's,
every definition here is written from the crosslinkers' public structures. Each
crosslinker is its REACTED bridge - what is left between the two residues once
the reaction has happened - as a SMILES whose two attachment atoms carry the
atom maps 1 and 2:

  * an NHS ester acylates a nucleophile: LYS NZ, SER OG, THR OG1, TYR OH, or the
    N-terminal N of residue 1. The ester's N-hydroxysuccinimide leaves, so the
    bridge ends in the acyl carbon.
  * a diazirine (the SDA family) photolyses to a carbene that inserts into any
    X-H bond: the bridge ends in that carbon, bonded to the residue's last
    side-chain heavy atom (CA for glycine). The pair's FIRST residue is the NHS
    end, the second the diazirine end.

Every SMILES below reproduces the supplier's molecular formula once its leaving
groups are restored (tests/crosslinks_test.py). The components are generated
once by `python -m alphafold3.common.crosslinks --write` into crosslinkers.cif
beside this file, so every port reads the same atoms, names and ideal geometry.

Each pair becomes one ligand chain (chain ids after the job's last) and two
bondedAtomPairs. A residue can carry one crosslink: with remove_overlapping
(af3x's default) a pair touching a residue an earlier pair already used is
dropped, and without it such a job is refused, since the second bond would
over-bond the same atom.
"""

import dataclasses
import itertools
import os
from typing import Any, Mapping, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
COMPONENTS_CIF = os.path.join(_HERE, 'crosslinkers.cif')

NHS, DIAZIRINE = 'nhs', 'diazirine'

# NHS-ester nucleophiles; residue 1's alpha amine when the residue has none
NHS_ATOMS = {'LYS': 'NZ', 'SER': 'OG', 'THR': 'OG1', 'TYR': 'OH'}
_BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}


@dataclasses.dataclass(frozen=True)
class Crosslinker:
  name: str
  bridge: str          # the reacted bridge; atom maps 1 and 2 are the two attachment atoms (one atom for CDI)
  ends: tuple[str, str]
  formula: str         # the reagent as sold, to check `bridge` against
  reagent: str         # the reagent's SMILES (leaving groups included)
  description: str

  @property
  def code(self) -> str:
    """The component id: prefixed, so it never shadows a real CCD entry."""
    return 'XL-' + self.name.upper()


_NHS_ESTER = 'ON9C(=O)CCC9=O'   # ring label 9: the reagent around it may hold ring 1 open
_SULFO_NHS = 'ON9C(=O)CC(S(=O)(=O)O)C9=O'


def _reagent(core, left=_NHS_ESTER, right=_NHS_ESTER):
  return core.replace('{L}', left).replace('{R}', right)


_DEFS = [
    Crosslinker('DSS', 'O=[C:1]CCCCCC[C:2]=O', (NHS, NHS), 'C16H20N2O8',
                _reagent('O=C({L})CCCCCCC(=O){R}'), 'disuccinimidyl suberate'),
    Crosslinker('BS3', 'O=[C:1]CCCCCC[C:2]=O', (NHS, NHS), 'C16H20N2O14S2',
                _reagent('O=C({L})CCCCCCC(=O){R}', _SULFO_NHS, _SULFO_NHS),
                'bis(sulfosuccinimidyl) suberate (the DSS bridge)'),
    Crosslinker('DSG', 'O=[C:1]CCC[C:2]=O', (NHS, NHS), 'C13H14N2O8',
                _reagent('O=C({L})CCCC(=O){R}'), 'disuccinimidyl glutarate'),
    Crosslinker('BS2G', 'O=[C:1]CCC[C:2]=O', (NHS, NHS), 'C13H14N2O14S2',
                _reagent('O=C({L})CCCC(=O){R}', _SULFO_NHS, _SULFO_NHS),
                'bis(sulfosuccinimidyl) glutarate (the DSG bridge)'),
    Crosslinker('DSSO', 'O=[C:1]CCS(=O)CC[C:2]=O', (NHS, NHS), 'C14H16N2O9S',
                _reagent('O=C({L})CCS(=O)CCC(=O){R}'), 'disuccinimidyl sulfoxide'),
    Crosslinker('DSBU', 'O=[C:1]CCCNC(=O)NCCC[C:2]=O', (NHS, NHS), 'C17H22N4O9',
                _reagent('O=C({L})CCCNC(=O)NCCCC(=O){R}'), 'disuccinimidyl dibutyric urea'),
    Crosslinker('CDI', 'O=[C:1]', (NHS, NHS), 'C7H6N4O',
                'O=C(n1ccnc1)n1ccnc1',
                'carbonyldiimidazole: one carbonyl joins the two nucleophiles'),
    Crosslinker('PHOX', 'O=[C:1]c1cc(cc(c1)P(=O)(O)O)[C:2]=O', (NHS, NHS), 'C16H13N2O11P',
                _reagent('O=C({L})c1cc(cc(c1)P(=O)(O)O)C(=O){R}'),
                'PhoX, disuccinimidyl phenylphosphonic acid'),
    Crosslinker('BSPEG5', 'O=[C:1]CCO' + 'CCO' * 4 + 'CC[C:2]=O', (NHS, NHS), 'C22H32N2O13',
                _reagent('O=C({L})CCO' + 'CCO' * 4 + 'CCC(=O){R}'), 'bis-succinimide ester PEG5'),
    Crosslinker('BSPEG9', 'O=[C:1]CCO' + 'CCO' * 8 + 'CC[C:2]=O', (NHS, NHS), 'C30H48N2O17',
                _reagent('O=C({L})CCO' + 'CCO' * 8 + 'CCC(=O){R}'), 'bis-succinimide ester PEG9'),
    Crosslinker('AZIDE-A-DSBSO',
                'CC1(CCCN=[N+]=[N-])OCC(CS(=O)CC[C:1]=O)(CS(=O)CC[C:2]=O)CO1', (NHS, NHS),
                'C24H33N5O12S2',
                _reagent('CC1(CCCN=[N+]=[N-])OCC(CS(=O)CCC(=O){L})(CS(=O)CCC(=O){R})CO1'),
                'azide-tagged acid-cleavable disuccinimidyl bissulfoxide'),
    Crosslinker('SDA', 'O=[C:1]CC[CH:2]C', (NHS, DIAZIRINE), 'C9H11N3O4',
                _reagent('O=C({L})CCC1(C)N=N1'), 'succinimidyl 4,4\'-azipentanoate (sulfo-SDA: the same bridge)'),
    Crosslinker('LCSDA', 'O=[C:1]CCCCCNC(=O)CC[CH:2]C', (NHS, DIAZIRINE), 'C15H22N4O5',
                _reagent('O=C({L})CCCCCNC(=O)CCC1(C)N=N1'),
                'succinimidyl 6-(4,4\'-azipentanamido)hexanoate (sulfo-LC-SDA: the same bridge)'),
    Crosslinker('SDAD', 'O=[C:1]CCSSCCNC(=O)CC[CH:2]C', (NHS, DIAZIRINE), 'C14H20N4O5S2',
                _reagent('O=C({L})CCSSCCNC(=O)CCC1(C)N=N1'),
                'succinimidyl 2-((4,4\'-azipentanamido)ethyl)-1,3\'-dithiopropionate (sulfo-SDAD: the same bridge)'),
]
CROSSLINKERS = {d.name: d for d in _DEFS}
# names as af3x and the suppliers write them
ALIASES = {'SULFO-SDA': 'SDA', 'SULFO-LC-SDA': 'LCSDA', 'LC-SDA': 'LCSDA', 'SULFO-SDAD': 'SDAD',
           'BS(PEG)5': 'BSPEG5', 'BS(PEG)9': 'BSPEG9'}


def lookup(name: str) -> Crosslinker:
  key = name.upper()
  key = ALIASES.get(key, key)
  if key not in CROSSLINKERS:
    raise ValueError(f'unknown crosslinker {name!r}; known: {", ".join(sorted(CROSSLINKERS))}')
  return CROSSLINKERS[key]


# ---------------------------------------------------------------- components

def _component(xl: Crosslinker, seed: int = 0) -> tuple[str, dict[int, str]]:
  """-> (one mmCIF chem_comp block, {atom map: atom name}). Heavy atoms are named element + running index per
  element in SMILES order, hydrogens H1.. after them; ideal coordinates from one ETKDG embedding."""
  from rdkit import Chem
  from rdkit.Chem import AllChem
  from rdkit.Chem.Descriptors import MolWt
  from rdkit.Chem.rdMolDescriptors import CalcMolFormula

  mol = Chem.MolFromSmiles(xl.bridge)
  maps = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
  for a in mol.GetAtoms():
    a.SetAtomMapNum(0)
  mol = Chem.AddHs(mol)
  params = AllChem.ETKDGv3()
  params.randomSeed = seed
  if AllChem.EmbedMolecule(mol, params) != 0:
    raise RuntimeError(f'{xl.name}: no conformer')
  AllChem.MMFFOptimizeMolecule(mol)
  pos = mol.GetConformer().GetPositions()
  counts: dict[str, int] = {}
  names = []
  for a in mol.GetAtoms():
    e = a.GetSymbol()
    counts[e] = counts.get(e, 0) + 1
    names.append(f'{e.upper()}{counts[e]}')
  Chem.Kekulize(mol, clearAromaticFlags=False)
  order = {Chem.BondType.SINGLE: 'SING', Chem.BondType.DOUBLE: 'DOUB', Chem.BondType.TRIPLE: 'TRIP'}
  heavy = Chem.RemoveHs(mol)
  lines = [
      f'data_{xl.code}', '#',
      f'_chem_comp.id {xl.code}',
      f"_chem_comp.name '{xl.name} crosslink bridge'",
      '_chem_comp.type non-polymer',
      f"_chem_comp.formula '{CalcMolFormula(mol)}'",
      '_chem_comp.mon_nstd_parent_comp_id ?',
      '_chem_comp.pdbx_synonyms ?',
      f'_chem_comp.formula_weight {MolWt(mol):.3f}',
      f"_chem_comp.pdbx_smiles '{Chem.MolToSmiles(heavy)}'",
      '#', 'loop_',
      *['_chem_comp_atom.' + k for k in (
          'comp_id', 'atom_id', 'type_symbol', 'charge', 'pdbx_aromatic_flag', 'pdbx_leaving_atom_flag',
          'pdbx_model_Cartn_x_ideal', 'pdbx_model_Cartn_y_ideal', 'pdbx_model_Cartn_z_ideal')],
  ]
  for a, n, p in zip(mol.GetAtoms(), names, pos):
    lines.append(f'{xl.code} {n} {a.GetSymbol().upper()} {a.GetFormalCharge()} '
                 f'{"Y" if a.GetIsAromatic() else "N"} N {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}')
  lines += ['#', 'loop_', *['_chem_comp_bond.' + k for k in (
      'comp_id', 'atom_id_1', 'atom_id_2', 'value_order', 'pdbx_aromatic_flag')]]
  for b in mol.GetBonds():
    lines.append(f'{xl.code} {names[b.GetBeginAtomIdx()]} {names[b.GetEndAtomIdx()]} '
                 f'{order[b.GetBondType()]} {"Y" if b.GetIsAromatic() else "N"}')
  lines.append('#')
  return '\n'.join(lines) + '\n', {m: names[i] for m, i in maps.items()}


def write_components(path: str = COMPONENTS_CIF) -> None:
  blocks, sites = [], []
  for xl in _DEFS:
    block, site = _component(xl)
    blocks.append(block)
    sites.append(f'# {xl.code} attach {site[1]} {site.get(2, site[1])}')
  header = ('# Crosslinker bridges for alphafold3.common.crosslinks, generated by\n'
            '#   python -m alphafold3.common.crosslinks --write\n'
            '# from the SMILES there. Do not edit by hand.\n') + '\n'.join(sites) + '\n'
  with open(path, 'w') as f:
    f.write(header + ''.join(blocks))


_LOADED: tuple[dict[str, str], dict[str, tuple[str, str]]] | None = None


def components() -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
  """-> ({code: chem_comp block}, {code: (attachment atom 1, attachment atom 2)}) from crosslinkers.cif."""
  global _LOADED
  if _LOADED is None:
    text = open(COMPONENTS_CIF).read()
    sites, blocks = {}, {}
    for line in text.splitlines():
      if line.startswith('# XL-'):
        _, code, _, a1, a2 = line.split()
        sites[code] = (a1, a2)
    for chunk in text.split('data_')[1:]:
      code = chunk.split('\n', 1)[0].strip()
      blocks[code] = 'data_' + chunk
    _LOADED = blocks, sites
  return _LOADED


# ---------------------------------------------------------------- expansion

def residue_atom(end: str, restype: str, resid: int, ccd) -> str:
  """The residue atom a crosslinker end bonds to."""
  if end == NHS:
    if restype in NHS_ATOMS:
      return NHS_ATOMS[restype]
    if resid == 1:
      return 'N'
    raise ValueError(f'an NHS ester reacts with K, S, T, Y or the N-terminus; residue {resid} is {restype}')
  if restype == 'GLY':
    return 'CA'
  atoms = [a for a, e in zip(ccd[restype]['_chem_comp_atom.atom_id'], ccd[restype]['_chem_comp_atom.type_symbol'])
           if e != 'H' and a not in _BACKBONE]
  if not atoms:
    raise ValueError(f'{restype} has no side chain for a diazirine to insert into')
  return atoms[-1]


def flat(crosslinks: Sequence[Mapping[str, Any]]) -> list[tuple[str, tuple, tuple]]:
  out = []
  for k, link in enumerate(crosslinks):
    if set(link) - {'name', 'residue_pairs'} or 'name' not in link or 'residue_pairs' not in link:
      raise ValueError(f'crosslinks[{k}]: expected {{"name", "residue_pairs"}}, got {sorted(link)}')
    lookup(link['name'])
    for p, pair in enumerate(link['residue_pairs']):
      if (len(pair) != 2 or any(len(end) != 2 or not isinstance(end[0], str) or not isinstance(end[1], int)
                                for end in pair)):
        raise ValueError(f'crosslinks[{k}].residue_pairs[{p}]: expected [["A", 12], ["B", 30]], got {pair}')
      out.append((link['name'], tuple(pair[0]), tuple(pair[1])))
  return out


def regroup(links: Sequence[tuple[str, tuple, tuple]]) -> list[dict[str, Any]]:
  grouped: dict[str, list] = {}
  for name, a, b in links:
    grouped.setdefault(name, []).append([list(a), list(b)])
  return [{'name': n, 'residue_pairs': p} for n, p in grouped.items()]


def remove_overlapping(crosslinks: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list]:
  """af3x's remove_overlapping_crosslinks: in input order, drop a pair touching a residue already crosslinked.
  -> (the kept crosslinks, the dropped (name, end, end))."""
  used, kept, dropped = set(), [], []
  for link in flat(crosslinks):
    if link[1] in used or link[2] in used:
      dropped.append(link)
      continue
    used.update((link[1], link[2]))
    kept.append(link)
  return regroup(kept), dropped


def combinations(crosslinks: Sequence[Mapping[str, Any]], k: int):
  """af3x's sample_crosslink_combinations: every k-subset of the pairs, in input order.
  -> [(a name suffix, the crosslinks)]."""
  links = flat(crosslinks)
  if not 0 < k <= len(links):
    raise ValueError(f'cannot take {k} of {len(links)} crosslinks')
  for combo in itertools.combinations(links, k):
    tag = '_'.join(f'{n}-{a[0]}{a[1]}-{b[0]}{b[1]}' for n, a, b in combo)
    yield tag, regroup(combo)


def _chain_ids():
  for n in itertools.count(1):
    for letters in itertools.product('ABCDEFGHIJKLMNOPQRSTUVWXYZ', repeat=n):
      yield ''.join(letters)


def expand_json(raw: Mapping[str, Any], ccd, remove_overlapping_pairs: bool = True) -> tuple[dict[str, Any], list]:
  """An input JSON's `crosslinks` -> the same JSON with a ligand chain per pair, its two bondedAtomPairs and the
  crosslinkers' components in userCCD. -> (the JSON, the dropped pairs)."""
  raw = dict(raw)
  links = raw.pop('crosslinks', None) or []
  if not links:
    return raw, []
  if remove_overlapping_pairs:
    links, dropped = remove_overlapping(links)
  else:
    dropped = []
    seen = set()
    for _, a, b in flat(links):
      for end in (a, b):
        if end in seen:
          raise ValueError(f'residue {end[0]}{end[1]} carries two crosslinks; --remove_overlapping_crosslinks drops'
                           ' the later one')
        seen.add(end)
  # the polymer residue at each (chain, residue): its CCD code, a PTM's when it is modified
  residues = {}
  used_ids = set()
  for entry in raw['sequences']:
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    used_ids.update(ids)
    if kind != 'protein':
      continue
    from alphafold3.constants import residue_names
    seq = [residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(c, 'UNK') for c in e['sequence']]
    for m in e.get('modifications') or ():
      seq[m['ptmPosition'] - 1] = m['ptmType']
    for cid in ids:
      for i, code in enumerate(seq, 1):
        residues[(cid, i)] = code
  blocks, sites = components()
  # after the job's last chain id, not in its first gap: the ligands are appended, and a job of A and C would
  # otherwise name the crosslinker after C "B"
  last = max(((len(c), c) for c in used_ids), default=(0, ''))
  fresh = (c for c in _chain_ids() if (len(c), c) > last)
  sequences = list(raw['sequences'])
  bonds = [list(map(list, b)) for b in raw.get('bondedAtomPairs') or []]
  codes = []
  for name, a, b in flat(links):
    xl = lookup(name)
    for end in (a, b):
      if end not in residues:
        raise ValueError(f'crosslink {name}: {end[0]}{end[1]} is not a protein residue of this job')
    cid = next(fresh)
    sequences.append({'ligand': {'id': cid, 'ccdCodes': [xl.code]}})
    s1, s2 = sites[xl.code]
    for end, site, kind in ((a, s1, xl.ends[0]), (b, s2, xl.ends[1])):
      atom = residue_atom(kind, residues[end], end[1], ccd)
      bonds.append([[end[0], end[1], atom], [cid, 1, site]])
    if xl.code not in codes:
      codes.append(xl.code)
  raw['sequences'] = sequences
  raw['bondedAtomPairs'] = bonds
  raw['userCCD'] = (raw.get('userCCD') or '') + ''.join(blocks[c] for c in codes)
  return raw, dropped


def export(ccd) -> dict[str, Any]:
  """Everything another port needs, as data: the crosslinkers, their names, the components and the residue atoms
  (localfold's shared/chem/crosslinkers.js is generated from this)."""
  from alphafold3.constants import residue_names
  blocks, sites = components()
  standard = sorted(set(residue_names.PROTEIN_COMMON_ONE_TO_THREE.values()) - {'UNK'})
  return {
      'crosslinkers': {xl.name: {'code': xl.code, 'ends': list(xl.ends), 'attach': list(sites[xl.code]),
                                 'bridge': xl.bridge, 'description': xl.description} for xl in _DEFS},
      'aliases': ALIASES,
      'nhsAtoms': NHS_ATOMS,
      'diazirineAtoms': {r: residue_atom(DIAZIRINE, r, 2, ccd) for r in standard},
      'components': blocks,
  }


if __name__ == '__main__':
  import json
  import sys
  if sys.argv[1:] == ['--write']:
    write_components()
    print(f'wrote {COMPONENTS_CIF}')
  elif len(sys.argv) == 3 and sys.argv[1] == '--export':
    from alphafold3.constants import chemical_components
    with open(sys.argv[2], 'w') as f:
      json.dump(export(chemical_components.Ccd()), f, indent=1)
    print(f'wrote {sys.argv[2]}')
  else:
    sys.exit('usage: python -m alphafold3.common.crosslinks --write | --export OUT.json')
