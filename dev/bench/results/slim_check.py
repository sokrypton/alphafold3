"""Does the SLIM wheel work with no libcifpp components.cif anywhere?

The v3.1.8 fix moved the dictionary check out of module REGISTRATION and into
the `get_dssp` call. This asks the three questions that separates:

  1. does `import alphafold3.cpp` succeed (it did not in 3.1.7),
  2. does featurisation -- which needs cif_dict from that same module -- run,
  3. does get_dssp still raise something a human can act on.

A fold needs jax + weights and is NOT tested here; that is slim.ipynb on Colab.
"""
import os, sys, glob, traceback

os.environ.pop('LIBCIFPP_DATA_DIR', None)
venv = os.path.dirname(os.path.dirname(sys.executable))
found = glob.glob(os.path.join(venv, '**/components.cif'), recursive=True)
print('components.cif inside the venv:', found or 'NONE')

import alphafold3
print('alphafold3', alphafold3.__version__ if hasattr(alphafold3, '__version__') else '?',
      alphafold3.__file__)

print('--- 1. import the cpp extension ---')
from alphafold3.cpp import cif_dict
print('OK  cif_dict', cif_dict)

print('--- 2. CCD from rcsb + featurise ---')
from alphafold3.constants import ccd_fetch
# The package reads the two pickles from fixed paths inside itself
# (constants/converters/), which build_data normally writes. Write them there,
# exactly as the notebook's prefetch_ccd.py does.
conv = os.path.join(os.path.dirname(alphafold3.__file__), 'constants', 'converters')
os.makedirs(conv, exist_ok=True)
codes = ccd_fetch.codes_for_input(extra=('BTN',))
ccd_fetch.write_pickles(codes, os.path.join(conv, 'ccd.pickle'),
                        os.path.join(conv, 'chemical_component_sets.pickle'))
print('OK  wrote', len(codes), 'components')

from alphafold3.common import folding_input
from alphafold3.data import featurisation
from alphafold3.constants import chemical_components
_SEQ = 'GSHMKQLEDKVEELLSKNYHLENEVARLKKLV'
fi = folding_input.Input(
    name='t', chains=[
        folding_input.ProteinChain(id='A', sequence=_SEQ, ptms=[],
                                   unpaired_msa=f'>query\n{_SEQ}\n',
                                   paired_msa='', templates=[]),
        folding_input.Ligand(id='B', ccd_ids=['BTN']),
    ], rng_seeds=[1])
feats = featurisation.featurise_input(fold_input=fi, buckets=(64,),
                                      ccd=chemical_components.Ccd(),
                                      verbose=False)
print('OK  featurised', {k: v.shape for k, v in list(feats[0].items())[:3]})

print('--- 3. get_dssp still says something useful ---')
from alphafold3.cpp import mkdssp
try:
    mkdssp.get_dssp('')
    print('?? get_dssp did not raise')
except Exception as e:
    print('OK  get_dssp ->', type(e).__name__, str(e)[:120])
print('SLIM_CHECK_DONE')
