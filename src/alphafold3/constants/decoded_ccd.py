"""The chemical component dictionary, with mmCIF quoting decoded.

Wrapping `chemical_components.Ccd` rather than editing it: the quoting is the
cif format's, so decoding belongs at the boundary where this project reads the
dictionary.
"""

from __future__ import annotations

import functools

from alphafold3.constants import chemical_components


# 🔴 mmCIF QUOTES ARE ENCODING, NOT PART OF THE NAME, AND THE CCD KEEPS THEM.
# A value containing a quote character must be delimited in a cif file, so every
# primed nucleic-acid atom is stored as `"O5'"` - five characters, quotes
# included - while no amino acid atom needs quoting and none of them carry any.
# The dictionary hands those back verbatim, so anything reading an atom name out
# of it sees the quotes.
#
# What that costs is not an error anywhere obvious. AF3 packs an atom name into
# a four-character field, and five characters raise "Can not pad to a smaller
# shape. arr.shape=(5,), shape=(4,)" from deep inside the featuriser, naming
# neither the component, nor the atom, nor nucleic acids - so DNA and RNA simply
# do not featurise. Bonds are named the same way, so unquoting one side and not
# the other would silently match no bonds at all.
#
# 🔴 STRIP ONLY A MATCHED SURROUNDING PAIR. Stripping the quote characters
# generally also eats the trailing prime, turning O5' into O5 - and a residue
# whose C5' and C5 have become the same name still featurises, and is wrong.
_CCD_QUOTES = ('"', "'")
_CCD_QUOTED_FIELDS = (
    '_chem_comp_atom.atom_id',
    '_chem_comp_atom.alt_atom_id',
    '_chem_comp_bond.atom_id_1',
    '_chem_comp_bond.atom_id_2',
)


def _ccd_unquote(value):
  if len(value) >= 2 and value[0] == value[-1] and value[0] in _CCD_QUOTES:
    return value[1:-1]
  return value


def _ccd_decoded(entry):
  '''One CCD component with its atom and bond names decoded.'''
  decoded = dict(entry)
  for field in _CCD_QUOTED_FIELDS:
    if field in decoded:
      decoded[field] = [_ccd_unquote(name) for name in decoded[field]]
  return decoded


class _DecodedCcd:
  '''The chemical component dictionary, decoding each component as it is read.

  Wrapping rather than editing the vendored alphafold3 copy: the quoting is the
  cif format's, so decoding belongs at the boundary where this project reads it,
  and the upstream file stays a clean copy of upstream.
  '''

  __slots__ = ('_ccd', '_cache')

  def __init__(self, ccd):
    self._ccd = ccd
    self._cache = {}

  def __getitem__(self, key):
    component = self._cache.get(key)
    if component is None:
      component = _ccd_decoded(self._ccd[key])
      self._cache[key] = component
    return component

  def get(self, key, default=None):
    return self[key] if key in self._ccd else default

  def __contains__(self, key):
    return key in self._ccd

  def __iter__(self):
    return iter(self._ccd)

  def __len__(self):
    return len(self._ccd)

  def __hash__(self):
    return id(self)          # ok: the dictionary is immutable

  def keys(self):
    return self._ccd.keys()

  def items(self):
    return ((key, self[key]) for key in self._ccd)

  def values(self):
    return (self[key] for key in self._ccd)


@functools.lru_cache(maxsize=1)
def get_ccd(user_ccd=None):
  '''the chemical component dictionary, loaded once (it is large and slow)

  Atom and bond names come back decoded; see the note above on what the raw
  dictionary's mmCIF quoting costs.
  '''
  return _DecodedCcd(chemical_components.Ccd(user_ccd=user_ccd))
