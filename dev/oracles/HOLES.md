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

**`p_atom_pair` is four models of one lineage, so it is one suspect, not four --
and the shapes say it is probably the HARNESS.**

    ours   (51, 32, 128, 16)      51 * 32 = 1632 = 68 tokens * 24 max_atoms
    native (18, 32, 128, 16)      18 * 32 =  576, i.e. 574 real atoms rounded up

Our flat queries axis is padded to `num_tokens * max_atoms`; native's stops at
the real atoms. The gate compares `pg[:18]` against `pr[:18]` on the stated
assumption that "the leading windows hold the same atoms in the same order",
and for the QUERY axis that holds -- `c_atom_cond` is exact under the same
`[:574]` slicing, which proves the real atoms are contiguous at the front.

But the KEY axis is 128 wide per window, and for the last real windows those
keys run PAST atom 574. Ours then reads our own padding slots (574..623);
native's axis simply ends at 576. So the two windows hold different keys near
the tail, and a tail artifact would explain a large `max|d|` with `q_atom`
exact -- q is a per-atom quantity over the query axis, p is per key PAIR.

Prediction to check with `DIAG=1` (added for exactly this): the per-window
max|d| should be concentrated at the HIGH window indices, and the per-key-position
max|d| at the END of each window. If it is spread evenly instead, the residual
is real and the pair conditioning genuinely differs.

Do not "fix" this by trimming the comparison until the DIAG says which it is.
Seven of the nineteen BAD cells so far have been the oracle, so the prior is
strong -- but the prior is exactly what makes a wrong trim easy to believe.
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


# The trunk z-INIT had no gate at all, and now it does

`dev/oracles/trunk_init_parity.py` (L1i). `trunk_parity` feeds the pairformer
SYNTHETIC s and z -- the right way to gate 48 blocks of arithmetic, and it means
nothing ever measured the tensor those blocks start from: the relative position
encoding, the bond embeddings, and for boltz2 two terms AF3 has no equivalent
for. `conditioning_parity` gates the DIFFUSION conditioner's copy of the
relative encoding, not the trunk's.

boltz2 is the model that proved the gap was real, and it now reads
**corr 1.000000, max|d|/rms 2.77e-06** with its own convention -- against
0.954268 / 6.49e-01 with AF3's, where the per-pair max|d| is IDENTICAL on all
4624 pairs, one constant vector everywhere.

Only boltz2 has an adapter. The other twelve are one function each and the
recipe is the same: assemble the vendor's own z-init terms from its own
checkpoint and feed them OUR features.

`PASSES=n` turns the same gate into the FULL TRUNK LOOP -- boltz's own
`s/z recycle -> msa_module -> pairformer`, n times, against our own
`Evoformer.__call__` carrying `prev`. CLOSED, and it settled the chain-bucket
question:

  convention     z-init      loop, 1 pass         loop, 4 passes
  same-chain     2.77e-06    s 1.9e-05 z 3.5e-05  s 1.5e-05 z 1.9e-05
  same-entity    6.49e-01    s 1.49    z 9.71     s 1.02    z 3.42

Our recycled trunk is bit-faithful to boltz2 with its own convention, which is
now the shipped default.

THREE oracle bugs came out of this gate before it was trustworthy, and they are
the reason its first reading (s 0.963 / z 0.915) was a lead and not a verdict:

  * boltz2's pairformer needs `v2=True` -- its layers carry `pre_norm_s` where
    the default builds `attention.norm_s`, 128 tensors missing, and the
    checkpoint's hparams do not mention the flag.
  * the MSA one-hot has to be built in BOLTZ's 33-class order, because our
    converter permutes those columns into our 31.
  * `is_paired` is 1 ON THE QUERY ROW for boltz2 (0 everywhere for rf3 -- the
    two vendors disagree about what the flag means). Feeding zeros was the last
    of the three and worth s 0.974 -> 1.000000 on its own.

What remains is DOWNSTREAM of the trunk and is a real question rather than a
hole: with the trunk provably exact, something in the diffusion or the sampler
turns a correct pair representation into a worse structure on 6MRR about 10% of
the time (mean 0.700 against 0.540, four samples of 40 at 0.84-1.5, while at
zero recycles the two conventions are level). The loop gate is the tool for the
next model that shows this.
