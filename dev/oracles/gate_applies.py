"""Is this (gate, model) cell a HOLE, or genuinely not applicable?

  python3 dev/oracles/gate_applies.py L1b.msa esmfold2_fast
  -> "n/a: esmfold2_fast has no MSA encoder (msa=0)"   exit 0
  -> ""                                                exit 1  (it applies: a hole)

`run_all_parity.sh` reports one SKIP status for two very different things: a
module the model DOES NOT HAVE, and a module it has with no adapter written.
Mixing them means the summary has no denominator -- 88 SKIPs sounds like 88
holes, and most of them are not. That is the same blindness
[[ungated-modules]] describes at the level of the L0-L6 table.

The knowledge lives in the registry and in model_config, so it is read from
there rather than restated: a variant that gains an MSA encoder tomorrow stops
being n/a without anyone editing this file.
"""
import sys

sys.path.insert(0, 'src')


def reason(gate, model):
  """-> a string saying why the cell is n/a, or '' if it genuinely applies."""
  from alphafold3.model import model_config as mc
  from alphafold3.model import model_registry as mr

  fam = model in mc.ESMFOLD2_FAMILY
  variants = getattr(mr, 'ESMFOLD2_VARIANTS', {})
  v = variants.get(model, {})

  # --- models with no comparable native at all ----------------------------
  # alphafold3 IS the reference implementation. There is no second
  # implementation to gate it against, so every module cell is n/a by
  # construction rather than by omission -- its L5/L6 folds are the real
  # measurement.
  if model == 'alphafold3' and not gate.startswith(('L5', 'L6')):
    return 'alphafold3 IS the reference -- nothing to compare it against'
  # chai-1 ships its modules inside TorchScript archives with no callable
  # submodule forward, so a module gate cannot be built by importing it. The
  # cells that ARE possible are the injection ones, built from verbatim I/O
  # captured during the port (L4.confidence_inject).
  if model == 'chai1' and not gate.startswith(('L0', 'L5', 'L6')) \
     and not gate.endswith('_inject'):
    return ('chai1 modules live in TorchScript archives with no callable '
            'forward; only *_inject cells are possible')

  # --- modules a model does not have at all ------------------------------
  if gate.startswith('L1b.') and fam and not v.get('msa'):
    return '%s has no MSA encoder (ESMFOLD2_VARIANTS msa=0)' % model
  if gate.startswith('L4.') and model in mc.NO_CONFIDENCE_HEAD:
    return '%s ships no confidence head (NO_CONFIDENCE_HEAD)' % model
  if gate == 'L1t.template' and fam:
    return 'the ESMFold2 family has no template embedder'

  # --- covered by a DIFFERENT cell ---------------------------------------
  # The in-process gates need the vendor importable beside jax. ESMFold2's
  # implementation lives in ~/venv_esm only, so the family is gated through
  # npz dumps instead, under the *_ref and *dump names. Reporting the
  # in-process cell as a hole would double-count what is already covered.
  # protenix's MSA module has its own cell. `msa_parity.py`'s own docstring says
  # so: "prot_parity.py gates the MSA module for protenix only, which is what
  # closes CLAMPED_OPM_NORM and NO_MSA_ROW_UPDATE". So L1b.msa is n/a there,
  # covered by L1b.prot.
  if gate.startswith('L1b.msa') and model in mc.PROTENIX_FAMILY:
    return 'protenix\'s MSA module is gated by L1b.prot, not L1b.msa'

  ref_covered = {'L1.trunk': 'L1.trunk_ref',
                 'L3.denoise': 'L3.denoise_ref',
                 'L1d.dgram': 'L1.trunk_ref (the distogram rides on it)',
                 'L2.diffusion': 'L3.denoise_ref'}
  if fam and gate in ref_covered:
    return ('ESMFold2 is gated by dump, not in-process: see %s'
            % ref_covered[gate])
  return ''


def main(argv):
  if len(argv) != 3:
    print(__doc__)
    return 2
  r = reason(argv[1], argv[2])
  if r:
    print('n/a: %s' % r)
    return 0
  return 1


if __name__ == '__main__':
  sys.exit(main(sys.argv))
