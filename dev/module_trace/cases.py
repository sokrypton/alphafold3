"""Input cases: one entry per problem shape we want traced.

A case owns its BATCH and, where one exists, a reference structure to score
against. The registry is the axis we extend over time -- protein / non-protein /
templates / MSA / multimer -- so the same trace + compare machinery covers all of
them without new code.

6mrr_single_seq (68-residue protein, single sequence, no template, no MSA) is the
smallest case that still runs every trunk/diffusion/confidence module, and the one
every ported model has a published number for. It is the regression workhorse.

The rest are AlphaFold3's OWN canonical inputs, which is the right way to cover the
problem space rather than inventing cases:
  * af3_featurised_example -- DeepMind's featurised test batch (150 tokens, a real
    16384-row MSA, 4 templates, an atomised ligand, 2 chains). Their model test's
    input. It has NO covalent polymer-ligand bond despite the ligand: its 96 bonds
    are all ligand-ligand. It also carries template_atom_mask and no template_mask,
    which is why batch_report has to accept either.
  * the 13 JSONs in alphafold3/examples/ -- monomer, homodimer, protein-protein,
    CCD ligand + covalent bond, SMILES ligand, 4 ions, DNA duplex, methylated DNA,
    RNA hairpin, modified RNA, glycosylation, phosphorylation.
Each declares `uses`: the optional feature channels it must exercise. trace.py checks
the built batch against that declaration, which is how we catch "the extra input
silently did nothing". Only channels that are actually decidable from a featurised
batch are declared -- see batch_report for why smiles/glycan/ptm are not among them.
"""
import json
import os

import numpy as np

PDB_6MRR = os.path.expanduser('~/6MRR.pdb')
AF3_REPO = os.path.expanduser('~/alphafold3')
AF3_EXAMPLES = os.path.join(AF3_REPO, 'examples')
AF3_FEATURISED = os.path.join(
    AF3_REPO, 'src/alphafold3/test_data/featurised_example.pkl')


def _prep_6mrr():
  """native 6MRR (chain A) via v1's pdb parser -> (seq, idx, ref_batch)."""
  import sys
  v1 = os.path.expanduser('~/v1-main')
  if v1 not in sys.path:
    sys.path.insert(0, v1)
  from colabdesign.af.prep import prep_pdb
  p = prep_pdb(PDB_6MRR, chain='A', ignore_missing=True)
  alpha = 'ARNDCQEGHILKMFPSTWYVX'
  seq = ''.join(alpha[min(int(a), 20)] for a in p['batch']['aatype'])
  return seq, p['idx'], p['batch']


# ------------------------------------------------------------------ active case

def case_6mrr_single_seq(**feat):
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  seq, idx, ref = _prep_6mrr()
  spec = parse_contigs('A').resolve(idx=idx)
  batch = f3.featurise_spec(spec, sequences={0: seq}, msa_crop_size=1, **feat)
  return dict(batch=batch, spec=spec, seq=seq,
              native_ca=np.asarray(ref['all_atom_positions'])[:, 1, :],
              num_msa=1)


def case_6mrr_self_template(**feat):
  """same input plus its own structure as a template -- the regime RoseTTAFold3
  needs to fold at all, and the one binder design runs in."""
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  from colabdesign2.af3.features import spec_to_templates
  seq, idx, ref = _prep_6mrr()
  spec = parse_contigs('A').resolve(idx=idx)
  batch = f3.featurise_spec(spec, sequences={0: seq}, msa_crop_size=1,
                            templates=spec_to_templates(spec, ref), **feat)
  return dict(batch=batch, spec=spec, seq=seq,
              native_ca=np.asarray(ref['all_atom_positions'])[:, 1, :],
              num_msa=1)


# ------------------------------------------------------- later: AF3's own inputs

def case_af3_featurised_example(**_feat):
  """DeepMind's featurised batch, loaded verbatim -- no featurisation of ours in
  the path, so a mismatch here is unambiguously a MODEL bug.

  The pickle names its classes under the top-level `alphafold3` package. Importing
  that alongside our vendored copy loads the compiled cpp extension twice and dies
  with 'type "CifDict" is already registered'. Aliasing the vendored package into
  sys.modules under that name makes pickle resolve to the copy already imported --
  submodules then resolve through the vendored __path__.
  """
  import pickle
  import sys
  import colabdesign2.af3.alphafold3 as vendored

  had = sys.modules.get('alphafold3')
  sys.modules['alphafold3'] = vendored
  try:
    with open(AF3_FEATURISED, 'rb') as fh:
      batch = pickle.load(fh)
  finally:
    if had is None:
      sys.modules.pop('alphafold3', None)
    else:
      sys.modules['alphafold3'] = had
  if isinstance(batch, (list, tuple)):
    batch = batch[0]
  return dict(batch=batch, spec=None, seq=None, native_ca=None, num_msa=None)


def _af3_json_case(path, msa_crop_size=1, **_feat):
  """featurise one of alphafold3/examples/*.json through OUR path.

  The JSONs carry no MSA (they expect AF3's data pipeline to fetch one), so every
  polymer chain is pinned to single-sequence mode and given no templates.

  Done on the JSON rather than on the parsed objects: folding_input's ProteinChain
  and RnaChain are __slots__ classes, not dataclasses, so dataclasses.replace raises
  TypeError on them. Empty unpairedMsa/pairedMsa plus an empty templates list is
  AF3's own documented way to ask for single-sequence mode.
  """
  from colabdesign2.af3.alphafold3.common import folding_input
  from colabdesign2.af3 import features as f3

  spec = json.loads(open(path).read())
  for chain in spec.get('sequences', []):
    if 'protein' in chain:
      chain['protein'].update(unpairedMsa='', pairedMsa='', templates=[])
    if 'rna' in chain:
      # RNA takes unpairedMsa only -- pairedMsa is rejected as an unexpected key.
      chain['rna']['unpairedMsa'] = ''
  fold_input = folding_input.Input.from_json(json.dumps(spec))
  batch = f3.featurise(fold_input, msa_crop_size=msa_crop_size)
  return dict(batch=batch, spec=None, seq=None, native_ca=None,
              num_msa=msa_crop_size)


_AF3_JSON = {                                   # file -> feature channels covered
    'ubiquitin_monomer':          (),
    'tetr_homodimer':             ('multimer',),
    'barnase_barstar':            ('multimer',),
    'calmodulin_4calcium':        ('ligand', 'ion', 'multimer'),
    'kras_g12c_sotorasib':        ('ligand', 'bond', 'atomized'),
    'streptavidin_biotin_smiles': ('ligand', 'atomized'),
    'tetr_dimer_tetracycline':    ('ligand', 'multimer', 'atomized'),
    'rnaseb_glycosylated':        ('ligand', 'bond', 'atomized'),
    'erk2_phosphorylated':        ('atomized',),
    'u1a_rna_hairpin':            ('rna', 'multimer'),
    'modified_rna':               ('rna', 'atomized'),
    'methylated_dna':             ('dna', 'multimer', 'atomized'),
    'tetr_dimer_dna':             ('dna', 'multimer'),
}


CASES = {
    '6mrr_single_seq': dict(
        build=case_6mrr_single_seq, uses=(), needs=(PDB_6MRR,), status='active',
        doc='6MRR chain A (68 res), single sequence, no template, no MSA'),
    '6mrr_self_template': dict(
        build=case_6mrr_self_template, uses=('template',), needs=(PDB_6MRR,),
        status='active',
        doc='6MRR chain A + its own structure as a template'),
    'af3_featurised_example': dict(
        build=case_af3_featurised_example,
        uses=('msa', 'template', 'ligand', 'multimer', 'atomized'),
        needs=(AF3_FEATURISED,), status='active',
        doc="AlphaFold3's own featurised test batch (150 tokens, real MSA + templates)"),
}

for _name, _uses in _AF3_JSON.items():
  _path = os.path.join(AF3_EXAMPLES, _name + '.json')
  CASES['af3_' + _name] = dict(
      build=(lambda p=_path, **kw: _af3_json_case(p, **kw)), uses=_uses, needs=(_path,),
      status='active', doc='alphafold3/examples/%s.json' % _name)


# ------------------------------------------- real targets with reference structures
# AF3's example JSONs carry no coordinates, so tests on them can only assert that the
# model runs. These three add a ground truth, which is what lets a test say the
# prediction is RIGHT rather than merely finite.

CIF_DIR = os.path.expanduser('~')


def _cif_chain(pdb_id, chain_id, drop_hetero=True):
  """(sequence, reference representative-atom coords) for one polymer chain.

  Representative atom is CA for protein and C1' for nucleic acid -- the same choice
  the RMSDs elsewhere in this repo use.
  """
  import gemmi
  st = gemmi.read_structure(os.path.join(CIF_DIR, pdb_id + '.cif'))
  st.setup_entities()
  if drop_hetero:
    st.remove_ligands_and_waters()
  ch = st[0][chain_id]
  seq = gemmi.one_letter_code([r.name for r in ch])
  coords = []
  for r in ch:
    a = r.find_atom('CA', '*') or r.find_atom("C1'", '*')
    coords.append([a.pos.x, a.pos.y, a.pos.z] if a else [np.nan] * 3)
  return seq.upper(), np.asarray(coords, np.float32)


def _real_target_case(pdb_id, chain_id, kind='protein', with_template=False, **feat):
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  seq, native = _cif_chain(pdb_id, chain_id)
  # gemmi writes X for any residue outside the standard alphabet -- modified bases in
  # tRNA, for instance. Map them to the unknown token rather than dropping them, so
  # the chain length still matches the reference coordinates.
  seq = seq.replace('X', 'N' if kind == 'rna' else 'A')
  spec = parse_contigs(chain_id).resolve(length=len(seq))
  batch = f3.featurise_spec(spec, sequences={0: seq}, msa_crop_size=1, **feat)
  return dict(batch=batch, spec=spec, seq=seq, native_ca=native, num_msa=1)


CASES['1stp_streptavidin'] = dict(
    build=(lambda **kw: _real_target_case('1STP', 'A', **kw)),
    uses=(), needs=(os.path.join(CIF_DIR, '1STP.cif'),), status='active',
    doc='streptavidin core (121 aa) -- small protein with a real biotin complex')

CASES['1ehz_trna'] = dict(
    build=(lambda **kw: _real_target_case('1EHZ', 'A', kind='rna', **kw)),
    uses=('rna',), needs=(os.path.join(CIF_DIR, '1EHZ.cif'),), status='active',
    doc="yeast tRNA-Phe (76 nt, 1.93 A) -- pure RNA fold, no protein anywhere")

CASES['7bny_emcv_2a'] = dict(
    build=(lambda **kw: _real_target_case('7BNY', 'B', **kw)),
    uses=(), needs=(os.path.join(CIF_DIR, '7BNY.cif'),), status='active',
    doc='EMCV 2A protein (140 aa, released 2021-12-08, after AF3 cutoff)')


LIGAND_BUCKETS = (64,)      # covers BTN(16) / ATP(31) / HEM(43) in one shape


def _ligand_only_case(ccd_code, **feat):
  """A single ligand, no polymer at all -- the cheapest input that exercises the
  ligand path end to end.

  AF3 tokenises a ligand per atom, so this is 16-43 tokens against 150-500 for the
  example complexes: seconds to compile and run, which makes it usable as a routine
  check rather than an occasional one. Ligand length is resolved from the CCD during
  featurisation, so the spec must NOT be .resolve()d first.
  """
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  spec = parse_contigs('lig:' + ccd_code)
  # One shared bucket so every small ligand pads to the SAME token count. Compile cost
  # is set by the graph (48 pairformer blocks + the diffusion transformer), not by the
  # input, so a 16-atom ligand costs as much to compile as a 500-token complex -- and
  # with buckets=None each distinct atom count is a separate compile. Bucketing turns
  # three compiles into one, which is the whole point of a "fast" ligand check.
  feat.setdefault('buckets', LIGAND_BUCKETS)
  # NOT return_spec=True: resolve_ligands counts chains in the batch, and bucket
  # padding reads as an extra chain ("batch has 2 chains but the spec has 1"). The
  # resolved spec is only needed to fill in ligand token counts, which nothing here
  # consumes -- AF3 resolves them itself during featurisation.
  batch = f3.featurise_spec(spec, msa_crop_size=1, **feat)
  return dict(batch=batch, spec=spec, seq=None, native_ca=None, num_msa=1,
              ccd_code=ccd_code)


for _code, _doc in (('BTN', 'biotin, 16 atoms -- rigid fused bicycle + valerate tail'),
                    ('ATP', 'ATP, 31 atoms -- flexible triphosphate, ring + sugar'),
                    ('HEM', 'haem, 43 atoms -- large macrocycle with an Fe centre')):
  CASES['lig_' + _code.lower()] = dict(
      build=(lambda c=_code, **kw: _ligand_only_case(c, **kw)),
      uses=('ligand', 'atomized'), needs=(), status='active',
      doc='ligand only: ' + _doc)


def available(name):
  """(ok, reason) -- whether this case's inputs are on disk."""
  missing = [p for p in CASES[name]['needs'] if not os.path.exists(p)]
  return (not missing), ('missing ' + ', '.join(missing) if missing else '')


def build(name, **feat):
  """feat: extra featurise_spec kwargs a MODEL requires (e.g. opendde=True, which
  attaches the structural-token batch its diffusion runs on). Keeping this on the
  call rather than in the case means one case serves every model."""
  ok, why = available(name)
  if not ok:
    raise FileNotFoundError(f'case {name!r}: {why}')
  case = CASES[name]['build'](**feat)
  case['name'] = name
  case['uses'] = CASES[name]['uses']
  return case
