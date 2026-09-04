# The ported models: what each one needed

`OF3_AF3_PORTING_NOTES.md` records the first port, OpenFold3, in full detail.
This is the companion for the rest: what kind of change each model needed, and
how much of it is forced.

## How much is forced

A branch that creates or omits a parameter cannot be replaced by AlphaFold 3's
behaviour — the converted weights simply stop loading. A branch that only changes
arithmetic could be, at some cost in accuracy. Deriving each model's parameter
tree twice with identical config shapes, flipping only `global_config.model`,
separates the two:

| `--model` | parameters | forced by its branches | AF3 parameters it does not have |
|---|---|---|---|
| `intellifold2` | 406 | **0** | 0 |
| `openfold3` | 406 | 22 | 22 |
| `chai1` | 397 | 52 | 63 |
| `boltz2` | 439 | 116 | 83 |
| `protenix2` | 403 | 147 | 150 |
| `rosettafold3` | 442 | 186 | 150 |
| `opendde` | 480 | 240 | 166 |

IntelliFold-2 is the limiting case: it runs AlphaFold 3's graph verbatim, and its
entire port is the config widening (pair 512, MSA 256, 8 heads). It is
deliberately absent from `model_config.OPENFOLD3_LINEAGE` for that reason.

## Can a branch be replaced by a weight transform?

Sometimes. Going through every `global_config.model` branch, they fall into
three kinds:

**Genuine divergence (most of them).** Different operations, layouts or heads:
chai-1's merged triangle multiplication and fused two-direction attention, its
`eps=0.1` LayerNorm and `(scale + 1)` gate where AlphaFold 3 uses a sigmoid, the
pair-bias transposes, the OpenFold3-lineage atom masking rules, OpenDDE's
structural expansion, Boltz-2's `a_to_b`, RoseTTAFold3's `kq_norm`, the
per-block pair LayerNorm. Several of these are index permutations of DATA, which
no reshape of a parameter can express.

**Parameter existence (18 sites).** Every `create_offset=` and `use_bias=`
toggle, plus Boltz-2's `_embed_atom_features_bias` and RoseTTAFold3's
`_conformer_embedding_bias`. These *could* all go -- a LayerNorm with a zero
offset is a LayerNorm without one -- by always creating the parameter and having
the converters emit zeros. They are kept deliberately:
`create_offset=model in ('boltz2', 'rosettafold3')` states a fact about those
checkpoints where it matters, and a uniform graph with silent zeros would leave
no way to tell a trained offset from an absent one.

**Exactly foldable arithmetic (5 sites, 4 folded).** These were doing per-step
work to compensate for something fixed, and the converter now does it once:

| was | is | checked against the branch it replaced |
|---|---|---|
| `ref_element - 1` on the batch, for openfold3/opendde/protenix2 | a row gather on `embed_ref_element` | max\|d\| = 0, all 128 element values |
| chai-1 fed the unscaled noise level | `bias += weight * 0.25*log(16)` | max\|d\| 3.7e-7 over 2000 noise levels |
| boltz2's token-level pLDDT, broadcast over atom slots | the weight tiled over the 24 slots | every slot identical; fold bit-identical |
| boltz2's token-level experimentally-resolved, likewise | likewise | likewise |

Both follow from where the feature enters the graph. `one_hot(e) @ W` makes an
index shift a row gather; `cos(2pi(0.25*log(s)*w + b))` makes a constant factor
inside the log a shift of `b`. The third, RoseTTAFold3's all-zero `is_paired`
column, is foldable by zeroing one weight row but is left alone: it is a fact
about its FEATURISATION, and a zeroed weight row is the wrong place to say so.

The element shift is the one worth calling out. It mutated the batch dict inside
the forward pass, and its correctness rested on a hand-kept list of which models
need it -- Boltz-2 and RoseTTAFold3 are 1-indexed and must NOT shift. That is
now a property of each model's weights instead of a condition to get right.

## Input conventions

Some families also need the INPUT built their way; `model_registry._FEATURISE`
declares which, and `model/pipeline/model_features.py` applies them. Every one is
silent when it is wrong — nothing errors, the fold is merely worse — so
`--featurise_off=NAME` exists to measure what each one buys.

| convention | models | what it is |
|---|---|---|
| `padded_keys` | opendde, protenix2 | zero-pad the 128-key atom window; AF3 slides an out-of-range window back in bounds |
| `circular_keys` | chai1 | take that window modulo the atom count instead |
| `drop_atoms` | chai1 | chai numbers its atoms without the C-terminal OXT |
| `std_conformers` | chai1 | chai's own standard-residue conformers. **OFF by default and not published** -- see below |
| `zero_msa_without_alignment` | chai1 | with no MSA, chai feeds an all-gap one, not a depth-1 self-MSA |
| `esm` | chai1 | ESM2 token embeddings, most of chai's token stream |
| `chirals` | rosettafold3 | chirality features |
| `atomized_element_names` | rosettafold3 | atomised atoms carry their ELEMENT symbol |
| `restype_alignment` | rosettafold3 | rf3's own restype ordering on atomised tokens |
| `atomized_unknown_restype` | rosettafold3 | an atomised polymer token is UNKNOWN, not its parent residue type |
| `atomized_backbone_bonds` | rosettafold3 | restore the peptide bond at each atomised residue's two junctions |
| `modified_as_one_token` | boltz2 | a modified residue is ONE token, UNK restype, `is_modified` set — all three together |

These are weight conventions: applying one model's to another is not a feature,
it is a mistake.

## What each input convention is worth

`--featurise_off=NAME` runs without one, so what a convention buys is measured
rather than assumed. Two lessons about the measurement came first:

* **Test it where it can act.** RoseTTAFold3's three conventions only touch
  ATOMISED tokens, so ablating them on a plain protein reads null by
  construction. They need a ligand.
* **Test what it can change.** Stereochemistry is invisible to RMSD -- inverting
  two of biotin's three centres costs 0.81 A, which reads as a nudge. See
  `scripts/chirality_check.py`.
* **Test it where the model's prior is WRONG.** Six targets said RoseTTAFold3's
  chirality machinery bought nothing, but every one of them was a molecule whose
  stereochemistry the learned prior already gets right, so nothing could show.
  `examples/regression/5kx0_cyclic_dl_peptide.json` is a de novo cyclic peptide
  with eleven D-amino acids -- the prior is wrong eleven times -- and it is what
  found the two conventions below. Scored by `scripts/dl_chirality_score.py`.

| convention | measured effect |
|---|---|
| `atomized_unknown_restype` | **decides the enantiomer.** On 5KX0, 11/11 D-residues are built as D with it and **0/11** without. AlphaFold 3 keeps the PARENT residue type on an atomised residue, so D-aspartate reads back as ASP in `aatype`, in `profile` and in every `msa` row -- three channels naming the L residue, which outvote a reference conformer and a chirality feature that are both correct. Mind the confidence: without it the model is 0/11 correct at pTM 0.69, with it 11/11 at 0.62. The wrong answer was the confident one |
| `atomized_backbone_bonds` | **load-bearing.** 5KX0 folds to 1.40 A with it, 2.95 A without. AF3 extracts inter-residue bonds only where one side is a ligand chain, so a residue atomised INSIDE a polymer chain loses the peptide bond to each neighbour -- 14 of native's 97 token-bond pairs on this target |
| `atomized_element_names` | **load-bearing.** Without it biotin comes out as the wrong stereoisomer at two of three centres (3/3 -> 1/3) and the ligand moves 0.81 A; on 5KX0 the D-residues go 11/11 -> 7/11 and the fold 1.40 -> 4.97 A |
| `std_conformers` (chai-1) | **depends which quantity you ask about, and the two disagree.** 6MRR loses 2.2 pLDDT and 0.04 pTM without it -- but the STRUCTURE barely moves: CA-RMSD 1.698 A with, 1.776 A without, i.e. 0.08 A. The model gets less confident in an answer that is essentially the same one. Ligands are unaffected either way, because chai `center_random_augment`s a ligand's reference pose on every run by design |
| `modified_as_one_token` (boltz2) | **load-bearing.** On a phosphoserine: pTM 0.91 / pLDDT 92.8 with, 0.75 / 82.2 without |
| `padded_keys` | live on RNA (moves protenix2's tRNA 0.42 A), inert on protein |
| `drop_atoms` (OXT, chai-1) | small: 0.06 A |
| `circular_keys` (chai-1) | no measured effect on protein |
| `chirals` | **no measured effect on any outcome, including the one target built to break that claim.** 494 stereocentres across five targets -- protein backbones, biotin, N-glycan sugars, tRNA riboses, a phosphoserine -- all correct with it and without; and on 5KX0, where the L prior is wrong eleven times, ablating it gives 11/11 D-residues at 1.41 A against 11/11 at 1.40 A with it. Our centres are row-for-row identical to native's 81 on that input, so this is not an inert feature badly built -- what actually carries the D configuration is the REFERENCE CONFORMER through the atom encoder, once the parent restype stops contradicting it. RF3 trained the chirality term as a weak correction: its projection weight has rms 0.087 against 2.385 for the coordinate projection beside it |
| `restype_alignment` | no measured effect: 0.002 A on protein, 0.01 A on a ligand, 0.08 A on an atomised phosphoserine, stereochemistry unchanged throughout |

The two atomised-token conventions change nothing on an input with no atomised
polymer residue: the featurised batch for 6MRR, 1STP+BTN and 1EHZ is
byte-identical with them on and off, which is a stronger regression argument than
re-folding those targets and is how they were checked. The only other input they
touch is a phospho-ubiquitin, where they are a small gain (pTM 0.90 against 0.88).

`chirals` and `restype_alignment` are kept anyway. They are part of the input
convention RoseTTAFold3 was trained under, the measured cost of keeping them is
nil, and "no effect on six targets" is not the same claim as "no effect". The
5KX0 result is the reason to state that carefully rather than delete them: the
D-amino-acid failure looked exactly like a broken chirality path, and was not.

## What is NOT model-specific

Two things in this package are not AlphaFold 3's behaviour and are not tied to
any model, because they act on parts of the graph or the input that every family
shares:

* **`--cyclic`** wraps the relative-position encoding, so a chain has no N- or
  C-terminus. AlphaFold 3's input JSON cannot express this; the encoding is
  shared by the trunk and the diffusion head of every model here, so all of them
  honour it. A chain left out is byte-identical to before.
* **The decoded CCD** (`alphafold3.constants.decoded_ccd`). The dictionary stores
  every primed nucleic-acid atom name mmCIF-quoted (`"O5'"`), and five characters
  do not fit AlphaFold 3's four-character atom-name field, so DNA and RNA fail to
  featurise with an error naming neither the component nor the atom. Applied to
  every model including stock AlphaFold 3.

## A trap worth repeating

RoseTTAFold3's confidence head normalises its trunk inputs over the WHOLE tensor
rather than along the feature axis. Token padding — which native RF3 never has —
therefore enters the mean and variance. On a 68-residue input padded to 128 that
gave a PAE of ~28 Å everywhere, diagonal included, and pTM 0.04 against 0.89
unpadded. The structure was unaffected, so an RMSD screen reads clean either way.

The invariant to test is that **the bucket cannot change the answer**: run the
same input at two padded lengths and compare. Any parameter-free normalisation
that reduces over more than the feature axis has to be masked.

## Known gaps (audited 2026-09-04)

Two classes of divergence that every gate we had was structurally blind to. Both
were found by reading an independent Protenix port
(ChoongHwanLee, `chlee19990109-cloud/ColabFold`, branch
`colabfold2-protenix-proof`) that runs
the same AF3 graph and had already handled them.

**1. Padded atom windows with AF3's product mask.** AF3's cross-attention biases
with `1e9 * (mask_q - 1) * (mask_k - 1)`, penalising a pair only when the query
*and* the key are invalid. It can afford that because `AtomCrossAtt` never emits
padding: an out-of-bounds atom window is SHIFTED bodily back inside the real atom
count, so all 128 keys are real. rosettafold3, protenix2 and opendde do not
slide, they PAD (`_padded_key_window`), and under padding the product rule leaves
every padded key fully attendable from every real query. Only rosettafold3 has
the corrected OR-mask; **protenix2 and opendde still take the stock branch**.
Both natives mask explicitly for real queries — `attn_bias[..., :n, 0:pad_left]
= -inf` at `protenix/model/modules/primitives.py:497` and
`opendde/model/modules/primitives.py:533`. Note AF3's *token-level*
`self_attention` already uses the key-only form; the product form appears only
where the sliding guarantee was doing the work. Impact is the edge windows: 4 of
51 query blocks for a 1626-atom input, but 4 of 18 for a 574-atom one, so short
chains and ligands feel it most.

**2. Trained tensors the graph has no slot for.** Stock AF3 builds its norms with
`create_offset=False` and its Linears with `use_bias=False`. Where a ported
family added one, the converter never asks for the key and nothing reports it —
the shape manifest audits the other direction. Four families lose a trained
distogram bias this way (protenix2, rosettafold3, boltz2, opendde), plus
rosettafold3's `process_s_trunk`/`process_z` LayerNorm offsets and boltz2's
`single_conditioner.single_embed.bias`. See `converters/README.md`, "Coverage
runs BOTH ways", for the full table and the audit tool.

Neither class can be caught by an RMSD gate: the first only perturbs edge
windows, and the second mostly feeds heads that do not move coordinates.

**Both are now FIXED**, and the first one is worth recording honestly because the
size of the effect did not follow the size of the defect. Measured on the
featurised batch, the share of attended key slots that were padding is 6.3% for a
68-residue protein, 11.6% for 1STP+biotin and **87.5% for a lone biotin** -- 16
real atoms in a 32-query/128-key window. That looked decisive. It is not: folding
a lone biotin through protenix2 both ways moves it by **0.016-0.052 A** over three
seeds, and 1STP+BTN moves by under 0.02 A on both protenix2 and opendde. The
reason is that a masked key is a ZEROED key, so a real query attending to it gains
nothing and merely spreads its softmax thinner. The fix is right because it is
what both natives do, not because it buys accuracy -- do not go looking for a
regression it explains.

A third gap found by the same audit is **still open**: boltz-2 adds a learned
cyclic term to the SINGLE track (`s = s + self.cyclic_conditioning_init(cyclic)`,
`trunkv2.py:204`, weight absmax 0.45), and our cyclic support is only the
relative-position wrap in `create_relative_encoding`. So a boltz2 fold of a cyclic
peptide is missing part of boltz's cyclic conditioning. Closing it needs a forward
branch and the per-token cyclic scalar plumbed to the input embedder, not just a
converter line.


## Why chai-1's conformers are not shipped

`std_conformers.npz` is 22 KB, so this is not about size. It is derived from
chai's own RDKit cache (`conformers_v1.apkl`) rather than from anything we
converted, and chai-1's weights licence is one we have **not** established -- the
registry says so, and `output_terms()` prints it. Redistributing chai's data
alongside our blob is a claim we have not earned, and the cost of not doing it is
0.08 A (above).

So the default is the CCD ideal conformer every other model in this family uses.
The mechanism is intact: `featurise_chai1(std_conformers=True)` with the file
beside the weights restores chai's, and the oracles that exist to REPRODUCE
native chai pass it, because reproducing native is exactly the case where you
want chai's own inputs.

It is an explicit flag rather than "use the file if it happens to be present" on
purpose. Sniffing the filesystem would make a fold depend on which machine ran
it, which is the silent, environment-dependent divergence this codebase has paid
for more than once.
