# What is left for 100% parity

TWO categories, and the second was invisible until `parity_audit.py` existed:

  1. **HOLES** -- cells that never ran. `gate_applies.py` says which of those
     are real (the model has that module and no other cell covers it).
  2. **cells that ran and DISAGREE.** `run_all_parity.sh`'s `classify()` only
     asks whether a matching line exists, never what it says, so a comparison
     at corr 0.9678 is reported OK. `parity_audit.py` grades on corr AND
     max|d|/rms.

On the last full run: 234 OK / 79 N/A / 66 holes, and of the 115 actual
comparisons inside those OK cells, **PARITY 77, CLOSE 16, LOOSE 10, BAD 12**.

## Category 2: cells that ran and are NOT at parity

| grade | gate | model | quantity | corr | max\|d\|/rms |
|---|---|---|---|---|---|
| BAD | L2.atom_encoder | protenix2 | `p_atom_pair` | 0.967768 | 9.64 |
| BAD | L2.atom_encoder | protenix1 | `p_atom_pair` | 0.983858 | 3.19 |
| BAD | L2.atom_encoder | openfold3 | `p_atom_pair` | 0.999816 | 1.39 |
| BAD | L2.atom_encoder | openbind0 | `p_atom_pair` | 0.999860 | 1.26 |
| BAD | L2.atom_* | rosettafold3 | `q_atom` | 0.999582 | 0.84 |
| BAD | L2.atom_* | rosettafold3 | `a_token` | 0.999870 | 0.57 |
| BAD | L1t.template | intellifold2 | `template_embed` | 0.999999 | 0.157 |
| BAD | L3.denoise | rosettafold3 | `x_denoised` | 0.999948 | 0.153 |
| BAD | L3.denoise | intellifold2 | `x_denoised` | 0.999950 | 0.141 |
| BAD | L2.conditioning | rosettafold3 | `single_cond` | 0.999996 | 0.124 |

**`p_atom_pair` is four models of one lineage, so it is one suspect, not four.**
There is a tension to resolve first: `q_atom` is 1.000000 for protenix while
`p_atom_pair` is 9.64 off, and q is computed FROM p -- so either the comparison
is misaligned or p is not what feeds q. The comparison rests on an assumption
stated in its own comment ("the leading windows hold the same atoms in the same
order") while our flat atom axis is padded to `num_tokens * max_atoms` and
native's is not (51 windows against 18). `DIAG=1` now breaks it down per window
and per key position -- run that before reading any vendor source, which is the
lesson the ESMFold2 OXT taught.

Eliminated already: it is NOT the padded-key `ref_space_uid` collision.
`OPENFOLD3_LINEAGE` -- which gates `offsets_valid & keys_mask` -- already
contains all four padded-key families. And it is not the conformer centering:
the numbers are byte-identical centred or not.

`rosettafold3` accounts for four of the twelve on its own and has no
`L2.conditioning`/`L1.trunk` adapter either, so it is the single worst-covered
model in the panel.

# Category 1: the 22 non-esmfold2 holes, and what each needs

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
