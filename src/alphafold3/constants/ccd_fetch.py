# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Build the CCD pickles from the components an input actually names.

`build_data` parses libcifpp's whole 518 MB components.cif -- 51,275 components
-- to produce `ccd.pickle` (505 MB) and `chemical_component_sets.pickle`. That
is 48 s of a Colab session for a fold that references perhaps thirty codes.

LocalFold fetches one component at a time from
`https://files.rcsb.org/ligands/download/<CODE>.cif` (kilobytes each), and the
same trick works here because `Ccd` already accepts an arbitrary
`ccd_pickle_path`.

WHY THIS IS SAFE, and where it is not. Both pickles are derived from the SAME
fetched set, so they are self-consistent: a glycan in the set is classified as
one by `chemical_component_sets`, and inter-chain bond detection sees what it
expects. The failure mode is a component the input names and we did not fetch --
and that raises a KeyError from `Ccd.__getitem__`, which is loud. What must
never happen is fetching a SUBSET and pretending it is the dictionary: a glycan
missing from `GLYCAN_LINKING_LIGANDS` is silently treated as a generic ligand,
which changes bonding without erroring. So `codes_for_input` is deliberately
generous -- every standard residue, every code named anywhere in the input.
"""

from __future__ import annotations

import concurrent.futures
import pickle
import urllib.error
import urllib.request

CCD_URL = 'https://files.rcsb.org/ligands/download/{code}.cif'

# The residues any fold needs regardless of input: AF3 reads reference
# conformers for standard residues through the same dictionary as ligands.
_STANDARD = (
    'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP '
    'TYR VAL UNK SEC PYL MSE '                      # protein, incl. the two 22nd
    'A C G U N '                                    # RNA
    'DA DC DG DT DN '                               # DNA
    'HOH'                                           # water, cleaned out but looked up
).split()


def codes_for_input(fold_input=None, extra=()) -> list[str]:
  """Every component code a fold may look up. Generous on purpose."""
  codes = set(_STANDARD) | {c.upper() for c in extra}
  if fold_input is not None:
    for chain in getattr(fold_input, 'chains', ()):
      for attr in ('ccd_codes',):
        codes.update(c.upper() for c in getattr(chain, attr, ()) or ())
      # protein `ptms` and nucleic `modifications` are the same tuple shape
      for attr in ('ptms', 'modifications'):
        for mod in getattr(chain, attr, ()) or ():
          code = getattr(mod, 'ptm_type', None) or getattr(
              mod, 'modification_type', None)
          if code is None and isinstance(mod, (tuple, list)) and mod:
            code = mod[0]
          if code:
            codes.add(str(code).removeprefix('CCD_').upper())
  return sorted(codes)


def fetch_cifs(codes, max_workers: int = 16, log=None) -> dict[str, str]:
  """code -> its CIF text, fetched concurrently. Missing codes are skipped."""
  def one(code):
    try:
      with urllib.request.urlopen(CCD_URL.format(code=code), timeout=30) as r:
        return code, r.read().decode('utf-8', 'replace')
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
      return code, None

  out = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
    for code, text in pool.map(one, codes):
      if text:
        out[code] = text
      elif log:
        log(f'  note: no CCD entry for {code}')
  return out


def write_pickles(codes, ccd_pickle_path, sets_pickle_path, log=print,
                  libcifpp_dir=None):
  """Fetch `codes` and write the two pickles `alphafold3.constants` reads.

  `libcifpp_dir`, if given, also gets a `components.cif` holding the same
  fetched components. libcifpp needs one for DSSP, and without it every fold
  logs `rasa calculation failed` once per sample and `fraction_disordered`
  comes back 0.0 -- a value, not a gap, which then enters the ranking score at
  weight 0.5. The full dictionary is 518 MB; this one is 0.26 MB and is built
  from what we already downloaded. Point it at `<site-packages>/share/libcifpp`
  and `get_dssp` finds it on its own.
  """
  import os

  from alphafold3.constants.converters import chemical_component_sets_gen
  from alphafold3.cpp import cif_dict

  cifs = fetch_cifs(codes, log=log)
  log(f'fetched {len(cifs)} of {len(codes)} components from files.rcsb.org')
  # One multi-data CIF, parsed by the same parser `Ccd(user_ccd=...)` uses, so
  # the dict shape is whatever that produces rather than something rebuilt here.
  merged = '\n'.join(cifs.values())
  ccd = {k: v.to_dict()
         for k, v in cif_dict.parse_multi_data_cif(merged).items()}
  with open(ccd_pickle_path, 'wb') as f:
    pickle.dump(ccd, f)
  sets = chemical_component_sets_gen.find_ions_and_glycans_in_ccd(ccd)
  with open(sets_pickle_path, 'wb') as f:
    pickle.dump(sets, f)
  if libcifpp_dir is not None:
    os.makedirs(libcifpp_dir, exist_ok=True)
    dst = os.path.join(libcifpp_dir, 'components.cif')
    with open(dst, 'w') as f:
      f.write(merged)
    log(f'wrote {dst} ({os.path.getsize(dst) / 1e6:.2f} MB) so DSSP works')
  log(f'wrote {ccd_pickle_path} ({len(ccd)} components) and {sets_pickle_path}')
  return ccd
