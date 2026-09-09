# The 22 non-esmfold2 parity holes, and what each needs

Generated from the adapter map (`NATIVES` in each gate module) crossed against
the last full matrix. The esmfold2 holes are not listed: 44 of them, all
targeted by the converter/reference fixes of 2026-09-09, and the authoritative
run is what says how many closed.

Read `dev/oracles/gate_applies.py` first -- a cell is only a hole if the model
HAS that module and no other cell covers it. 79 cells are n/a for reasons that
are properties of the model, not gaps in the work.

## Adapters that do not exist (the real work)

| gate | models | note |
|---|---|---|
| `L2.conditioning` | boltz2, opendde, openfold3, openbind0, intellifold2 | 5. `conditioning_parity` has protenix1/2, rosettafold3 and all eight esmfold2; these five need a `native_*` that runs the vendor's own DiffusionConditioning on our batch. The protenix adapter is the closest template. |
| `L2.atom_decoder` | boltz2, opendde, openfold3, openbind0, intellifold2, rosettafold3 | 6. `atom_parity` has 14 models for the ENCODER half; `DECODER=1` reaches `_decoder`, which needs the vendor's decoder called on native's own `q`/`c`/`p`. |
| `L2.diffusion` | boltz2, opendde | 2. `diffusion_parity` has if2/of3/openbind0/protenix/rf3. |
| `L3.denoise` | boltz2, opendde | 2. Same two, same shape. |
| `L2.atom_encoder` | boltz2, opendde | 2. Everything else in the panel has one. |
| `L4.confidence` | boltz2 | 1. The converter is done (66 -> 11); what is left is forward branches, and `boltz2-confidence-port` has the oracle recipe. |
| `L1.trunk` | rosettafold3 | 1. `trunk_parity` has the other six in-process vendors. |

## Cells whose adapter EXISTS but which skipped anyway

`L1.trunk` for boltz2, opendde and intellifold2 -- `trunk_parity.NATIVES` has
all three. The 2026-09-08 logs for those three are absent while the summary
carries SKIP rows, which is stale-log archaeology rather than evidence; the
authoritative run settles it. If they still skip, the reason is in the log and
is likely cheap -- the esmfold2 equivalents turned out to be two crashes
(`stack_blocks(0)` and the released-only confidence head), not missing work.

## Priority

boltz2 (7 holes) and opendde (6) are half the remaining work and both vendors
import cleanly beside jax, so neither needs the npz-dump machinery esmfold2
does. Start with `L2.conditioning` for those two: it is the same module for
both, `native_protenix` is a working template, and it unblocks the reasoning for
`L2.diffusion`/`L3.denoise`, which consume its output.
