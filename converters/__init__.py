"""AF3-lineage PyTorch -> haiku weight converters.

One module per model, named for the model itself (full name + version, matching
the runner's WEIGHTS keys and global_config.model): openfold3, intellifold2,
opendde, boltz2, protenix2, rosettafold3, chai1. Shared primitives live in common.py;
each family is a thin dialect + map on top. The runner dispatches through
CONVERTERS; the per-model `convert_<model>_weights` entry points and the blob io
are re-exported here.
"""

_ENTRY_POINTS = {
    'openfold3': ('openfold3', 'convert_openfold3_weights'),
    # same module: the two OpenFold3 releases share a converter, which
    # refuses a checkpoint that does not match the name asked for.
    'openbind': ('openfold3', 'convert_openbind_weights'),
    'intellifold2': ('intellifold2', 'convert_intellifold2_weights'),
    'opendde': ('opendde', 'convert_opendde_weights'),
    'boltz2': ('boltz2', 'convert_boltz2_weights'),
    'protenix2': ('protenix2', 'convert_protenix2_weights'),
    # same module and the same mapping: Protenix model types differ only in
    # counts and widths, which converters/protenix2.derive_dims reads off the
    # checkpoint. The entry points exist to pin the NAME, and each refuses a
    # checkpoint whose shape does not match it.
    'protenix05': ('protenix2', 'convert_protenix05_weights'),
    'protenix1': ('protenix2', 'convert_protenix1_weights'),
    'protenix1_20250630': ('protenix2', 'convert_protenix1_20250630_weights'),
    'protenix_mini': ('protenix2', 'convert_protenix_mini_weights'),
    'protenix_tiny': ('protenix2', 'convert_protenix_tiny_weights'),
    'rosettafold3': ('rosettafold3', 'convert_rosettafold3_weights'),
    'chai1': ('chai1', 'convert_chai1_weights'),
}


class _Converters(dict):
  """The converter registry, imported on first use.

  Lazily, because importing a converter pulls in alphafold3 (for the blob
  format) and numpy, and not every environment that needs this package has
  them: uploading needs huggingface_hub and nothing else, and a publish-time
  check that silently no-ops where it cannot import is worse than no check.
  """

  def __missing__(self, key):
    import importlib

    if key not in _ENTRY_POINTS:
      raise KeyError(key)
    module, attr = _ENTRY_POINTS[key]
    fn = getattr(importlib.import_module(f'.{module}', __name__), attr)
    self[key] = fn
    return fn

  def __iter__(self):
    return iter(_ENTRY_POINTS)

  def __len__(self):
    return len(_ENTRY_POINTS)

  def keys(self):
    return _ENTRY_POINTS.keys()


CONVERTERS = _Converters()


def __getattr__(name):
  """Module-level lazy access to the per-model entry points and the blob io."""
  import importlib

  for key, (module, attr) in _ENTRY_POINTS.items():
    if name == attr:
      return getattr(importlib.import_module(f'.{module}', __name__), attr)
  if name in ('encode_record', 'read_blob', 'read_records', 'write_params_blob'):
    return getattr(importlib.import_module('.common', __name__), name)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'CONVERTERS',
    'convert_intellifold2_weights', 'convert_openfold3_weights',
    'convert_opendde_weights', 'convert_boltz2_weights',
    'convert_protenix2_weights', 'convert_rosettafold3_weights',
    'convert_chai1_weights',
    'write_params_blob', 'read_blob', 'read_records', 'encode_record',
]
