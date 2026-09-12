"""Diff a VENDOR's own featuriser output against our batch, field by field.

    PYTHONPATH=src:.:dev/oracles python dev/oracles/featurisation_diff.py \
        boltz2 <vendor_dump>.npz [sequence]

Every port bug found on 2026-09-11/12 was an INPUT convention -- the self-MSA
depth, the empty template's restype, the structural atom axis, boltz2's template
visibility -- and L0-L4 are blind to all of them by construction: they feed each
module NATIVE's own features. This compares the features themselves.

The atom axis is matched by NAME before anything is compared, so "atom k on our
side is atom k on theirs" is proven rather than assumed (that check is what
turned protenix's `ref_element` from a 100% mismatch into a documented 1-index
shift).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _dec(v):
  v = np.asarray(v)
  if v.ndim == 2:
    v = v.argmax(-1)
  return ''.join(chr(32 + int(i)) for i in v).strip()


def _cmp(tag, ours, nat, note=''):
  ours, nat = np.asarray(ours, np.float64), np.asarray(nat, np.float64)
  if ours.shape != nat.shape:
    print('  %-26s SHAPE ours %-18s native %-18s %s'
          % (tag, ours.shape, nat.shape, note))
    return False
  md = float(np.abs(ours - nat).max())
  print('  %-26s %-18s max|d| %-12.6g %s'
        % (tag, str(nat.shape), md, note if md < 1e-6 else
           (note + '   <-- DIFFERS').strip()))
  return md < 1e-6


def main(argv=None):
  argv = sys.argv[1:] if argv is None else argv
  model, npz = argv[0], argv[1]
  seq = argv[2] if len(argv) > 2 else None
  import fold_check
  # Third argument: a SEQUENCE, or a fold-input JSON so the LIGAND and
  # MULTI-CHAIN cases can be driven -- a diff run only on a plain monomer
  # certifies a plain monomer, which is how the terminal-atom drop came to eat a
  # phosphoserine's O3P (see HOLES.md).
  chains = None
  if seq and seq.endswith('.json'):
    from alphafold3.common import folding_input
    chains = list(folding_input.Input.from_json(open(seq).read()).chains)
    seq = getattr(chains[0], 'sequence', '')
  if seq is None:
    seq = fold_check.parse_ca(os.path.expanduser('~/6MRR.pdb'))[0]
  d = np.load(npz)
  # Dumps written by the native_*_dump.py hooks prefix batch fields with
  # 'batch_'; accept either naming.
  if any(k.startswith('batch_') for k in d.files):
    _stripped = {k[len('batch_'):] if k.startswith('batch_') else k: d[k]
                 for k in d.files}

    class _Dump:                       # the npz interface the rest of this uses
      files = list(_stripped)

      def __getitem__(self, k):
        return _stripped[k]

    d = _Dump()
  def _sq(k):
    # Drop a LEADING BATCH AXIS only when it is actually one. Dropping the first
    # axis of every multi-dimensional array turned opendde's unbatched
    # `ref_pos` (602, 3) into (3,) and read as "native has 3 atoms".
    v = d[k]
    return v[0] if (v.ndim > 1 and v.shape[0] == 1) else v
  # The vendor pads its atom axis too (boltz2: 576 slots for 574 real atoms),
  # and comparing unfiltered reads as a shape mismatch that looks like a missing
  # atom. Filter by the vendor's own pad mask.
  _pad = None
  for k in ('atom_pad_mask', 'atom_mask', 'ref_mask'):
    if k in d.files:
      _pad = _sq(k).astype(bool)
      break

  # Some vendors pad the TOKEN axis too (intellifold2: 256 slots for 68 real
  # tokens). Trim it, or every per-token comparison reads as a shape mismatch.
  _ntok = None
  for k in ('token_pad_mask', 'token_mask', 'seq_mask'):
    if k in d.files:
      _ntok = int(np.asarray(_sq(k)).astype(bool).sum())
      break

  def sq(k, atoms=False):
    v = _sq(k)
    if atoms and _pad is not None and v.shape[:_pad.ndim] == _pad.shape:
      return v[_pad]
    if (not atoms) and _ntok is not None and v.ndim >= 1 and v.shape[0] > _ntok:
      return v[:_ntok]
    return v
  b, cfg, _ = fold_check._fold_setup(model, seq, None, chains=chains)
  mask = np.asarray(b['pred_dense_atom_mask']) > 0
  flat = mask.reshape(-1)
  print('%s: %d tokens, %d real atoms (native %d)'
        % (model, mask.shape[0], int(mask.sum()), sq('ref_pos', True).shape[0]))

  # ATOM ORDER, by name, before anything else
  onm = np.asarray(b['ref_atom_name_chars'])
  ours_names = [_dec(onm.reshape(-1, *onm.shape[2:])[i]) for i in np.flatnonzero(flat)]
  nat_names = [_dec(sq('ref_atom_name_chars', True)[i])
               for i in range(sq('ref_atom_name_chars', True).shape[0])]
  n = min(len(ours_names), len(nat_names))
  agree = sum(1 for i in range(n) if ours_names[i] == nat_names[i])
  print('  atom order by NAME: %d/%d agree%s'
        % (agree, n, '' if agree == n else
           '   first mismatch at %d: ours %r native %r'
           % (next(i for i in range(n) if ours_names[i] != nat_names[i]),
              ours_names[next(i for i in range(n) if ours_names[i] != nat_names[i])],
              nat_names[next(i for i in range(n) if ours_names[i] != nat_names[i])])))

  def ours_atoms(key):
    a = np.asarray(b[key])
    return a.reshape(-1, *a.shape[2:])[flat]

  print('  --- per-ATOM')
  _cmp('ref_pos', ours_atoms('ref_pos'), sq('ref_pos', True))
  _cmp('ref_charge', ours_atoms('ref_charge'), sq('ref_charge', True))
  _cmp('ref_space_uid', ours_atoms('ref_space_uid'), sq('ref_space_uid', True))
  # The element one-hot's base differs per vendor -- of3 and protenix index from
  # 0 (so ours is argmax + 1), boltz2 from 1 (ours is argmax). Detected rather
  # than assumed: a blanket +1 reported boltz2's elements as 100% wrong, which is
  # the same shape of harness fault as the atom order.
  ne = sq('ref_element', True)
  ne = ne.argmax(-1) if ne.ndim > 1 else ne
  oe = ours_atoms('ref_element')
  base = 0 if np.array_equal(oe, ne) else (1 if np.array_equal(oe, ne + 1) else None)
  _cmp('ref_element', oe, ne + (1 if base == 1 else 0),
       '(native one-hot base %s)' % ('1-indexed' if base == 0 else
                                     '0-indexed' if base == 1 else 'UNKNOWN'))
  print('  --- per-TOKEN')
  for ours_k, nat_k, note in (('residue_index', 'residue_index', ''),
                              ('token_index', 'token_index', ''),
                              ('asym_id', 'asym_id', ''),
                              ('entity_id', 'entity_id', ''),
                              ('sym_id', 'sym_id', '')):
    if ours_k in b and nat_k in d.files:
      o = np.asarray(b[ours_k])
      _cmp(ours_k, o - o.min(), sq(nat_k) - sq(nat_k).min(),
           note + ' (offset-normalised: both sides index from their own base)')
  # Fields a vendor names differently but means identically. A convention hides
  # in any of these, and none is visible to L0-L4.
  ALIAS = {
      'token_bonds': ('token_bonds', lambda v: v[..., 0] if v.ndim == 3 else v),
      'profile': ('profile', None),
      'deletion_mean': ('deletion_mean', None),
      'cyclic_period': ('cyclic_period', None),
      'method_feature': ('method_feature', None),
  }
  for ours_k, (nat_k, fn) in ALIAS.items():
    if ours_k in b and nat_k in d.files:
      o, nt = np.asarray(b[ours_k]), sq(nat_k)
      if fn is not None:
        nt = fn(nt)
      if o.ndim == nt.ndim + 1 and o.shape[-1] == 1:
        o = o[..., 0]
      if o.shape != nt.shape and o.ndim == nt.ndim and o.shape[-1] != nt.shape[-1]:
        # DIFFERENT VOCABULARY WIDTH. Comparing the first k columns of two
        # different vocabularies is meaningless -- what matters is whether one is
        # a fixed RELABELING of the other, because the converter permutes the
        # consuming Linear's columns and a relabeling is therefore free. Checked
        # by asking whether the argmax mapping is a consistent bijection.
        oa, na = o.reshape(-1, o.shape[-1]).argmax(-1), nt.reshape(-1, nt.shape[-1]).argmax(-1)
        pairs = set(zip(oa.tolist(), na.tolist()))
        fwd = len({a for a, _ in pairs}) == len(pairs)
        bwd = len({b for _, b in pairs}) == len(pairs)
        same_mass = np.allclose(o.sum(-1), nt.sum(-1))
        print('  %-26s %-18s width ours %d native %d: %s'
              % (ours_k, str(nt.shape), o.shape[-1], nt.shape[-1],
                 ('a consistent RELABELING over %d classes seen%s -- free, the '
                  'converter permutes the consuming Linear'
                  % (len(pairs), '' if same_mass else ', BUT THE MASS DIFFERS'))
                 if (fwd and bwd) else 'NOT a bijection <-- DIFFERS'))
        continue
      _cmp(ours_k, o, nt)
  if 'restype' in b and 'res_type' in d.files:
    o = np.asarray(b['restype'])
    nt = sq('res_type')
    _cmp('restype (argmax)', o.argmax(-1) if o.ndim > 1 else o, nt.argmax(-1),
         '(ours %d classes, native %d)' % (o.shape[-1] if o.ndim > 1 else -1,
                                           nt.shape[-1]))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
