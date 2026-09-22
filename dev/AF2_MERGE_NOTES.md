# Merging AF2's monomer and multimer code paths

We currently carry `modules.py` and `modules_multimer.py`, selected by
`RunModel(use_multimer=...)`. They are close enough that one file should serve
both. This records what is actually different, measured against the real
checkpoints, so the work can start from facts rather than from reading.

## How close they already are

    103 of 109 monomer scopes exist in multimer
    234 of 239 shared tensors have identical shapes
    35 of 175 config fields differ

The five shape mismatches are alphabet widths, nothing structural:

    left_single / right_single   (22,128) vs (21,128)
    preprocess_1d                (22,256) vs (21,256)
    masked_msa_head/logits       (256,23) vs (256,22)

## The 43 multimer-only modules are really 7

36 of them are `template_embedding/single_template_embedding/...` -- multimer
has a different template embedder. Design runs `use_templates=False`, so on the
design path those 36 do not exist. The remaining 7 are all in the structure
module, and they are a rename plus a fusion, not new mathematics:

    monomer (fused)                 multimer (split)
    q_scalar        (384, 192)  ->  q_scalar_projection  (384, 12, 16)   reshape
    kv_scalar       (384, 384)  ->  k_ + v_scalar_projection             slice+reshape
    q_point_local   (384, 144)  ->  q_point_projection   (384, 12, 12)   reshape
    kv_point_local  (384, 432)  ->  k_ + v_point_projection              slice+reshape
    affine_update   (384, 6)    ->  quat_rigid/rigid     (384, 6)        identical

192 = 12 heads x 16; 144 = 12 heads x 4 points x 3 coords; 432 = 144 (k, 4
points) + 288 (v, 8 points). Every split falls on a clean boundary.

## The one real blocker

**Multimer's scalar projections have no bias; monomer's do, and they are not
zero.**

    q_scalar  bias  |max| 0.904   mean|.| 0.151
    kv_scalar bias  |max| 0.185   mean|.| 0.021

So a converted monomer has nowhere to put them, and dropping them changes every
IPA output. This is the failure mode to avoid: it would not raise, it would
quietly degrade, and with no multimer oracle test nothing would catch it.

Fixable precisely because the code is vendored: give the multimer projections an
optional bias, absent for multimer weights and present for converted monomer
ones.

## Two config differences that matter

- `position_scale` 10.0 (monomer) vs 20.0 (multimer). Benign -- a scalar, set
  per model. This is the "we divide by different values" one.
- `outer_product_mean.first` False vs True. Multimer runs OPM at the start of
  the evoformer block. Monomer weights were trained with it last, so this cannot
  be padded around. It is already a config field read at graph-construction
  time, so one file can serve both orderings -- which is what makes the merge
  possible at all.

## Order of work

1. **A multimer oracle test first.** There is none. Right now nothing would
   catch a regression introduced by this merge, and the IPA bias is exactly the
   kind of change that degrades silently. Give multimer the
   `test_runner_matches_v1_feature_for_feature` treatment that found two real
   bugs on the monomer side.
2. Optional bias on the multimer scalar/point projections.
3. `convert_monomer_params()` -- the reshapes and slices above, plus zero-padding
   the five alphabet-width tensors.
4. Verify: converted monomer weights on the multimer graph reproduce the monomer
   graph's outputs, at `outer_product_mean.first=False` and
   `position_scale=10.0`.
5. Only then delete `modules_multimer.py`.

The compile-time argument for merging is weak -- you recompile per shape either
way. The argument is maintenance: one file, one place for a bug fix.

## Measured 2026-08-07: multimer is not pinned to v1, and the copies feature path diverges

Before merging, step 1 (a multimer oracle) was attempted. It surfaced that our
multimer path does **not** reproduce v1 multimer on the design path -- and the
cause is feature construction, not the structure module.

Setup: v1 `mk_afdesign_model(protocol='hallucination', use_multimer=True)`,
`length=12, copies=2`, dropout off, shared weights and seq params; compared
feature-for-feature and output-for-output against our runner at `copies=2`.

Raw output tensors diverge hugely (CA positions maxabs ~30 Å, distogram maxabs
13). Feature-for-feature localises it:

    residue_index   v1 [0..11, 61..72]   ours [1..12, 1..12]   <-- driver
    msa_feat        maxabs 0.99
    extra_msa       shape (1,24) int  vs  (1,24,23) one-hot     (gamma vs main)

**residue_index**: v1 multimer separates chains with a +50 gap (chain B starts
at 11+50=61) and marks copies with asym_id/sym_id. Ours resets each copy to the
same 1..12, so every cross-chain pair looks like a zero-offset intra-chain pair
to the relative-position encoding. Also off-by-one (ours starts at 1). The gap
convention is the one that matters; a global +1 shift is benign for a relative
encoding.

**extra_msa / msa_feat**: our featurizer uses gamma's MSA pipeline (one-hot
extra_msa, msa_profile/cluster_profile), v1 main here uses integer extra_msa.
This is the same main-vs-gamma contract difference the monomer oracle already
tolerates via DELIBERATE_DIVERGENCE, but for multimer it is unverified.

### Consequence for the plan

The merge cannot proceed on "multimer is close to monomer" alone, because
multimer is not yet numerically tied to any reference. The corrected order:

1. **Fix the copies residue_index convention** in featurize to match v1's +50
   gap (and confirm whether the model's rel-pos encoding uses raw cross-chain
   offset or gates on asym_id -- if it gates, the gap is cosmetic and the real
   drivers are the MSA features).
2. **Pin multimer to v1 with a clean oracle.** The copies=2 homo-oligomer is a
   confound (it also exercises the copies feature path AND the con/pae intra
   mask divergence). A single-chain or hetero-dimer multimer isolates the
   structure module, which is what the merge actually risks.
3. Only then the IPA-bias / convert_monomer_params work below.

The con/pae loss gaps seen alongside this (con Δ1.5) are the separate,
already-documented intra/inter mask divergence, not a forward-pass error.

## Measured 2026-08-07 (cont.): gamma is the right oracle; the gap is in featurize, not the forward

Per decision: pin multimer to **gamma** (our vendoring source), before merging.
Set up a gamma worktree (needed the same three JAX `jnp.clip(a_min=/a_max=)`
fixes our vendored copy already carries) and drove its multimer design model to
dump a reference (inputs + distogram/plddt/CA + losses) for a copies=2 L=12
homo-oligomer, dropout off. Scripts in scratch: gamma_dump.py, ours_cmp.py,
fwd_iso.py.

Findings, confirmed vs open:

**CONFIRMED -- residue_index was a false alarm.** Against gamma our copies
residue_index MATCHES. The earlier `[1..12,1..12]` vs `[0..11,61..72]` gap was
gamma-vs-main, not our bug -- main uses the +50 gap, gamma does not, and we
match gamma. (The off-by-one start is benign for a relative encoding.)

**CONFIRMED -- the divergence is in featurize, not the forward.** Full pipeline
(our featurize + forward) vs gamma: distogram maxabs 13. Feeding gamma's EXACT
input dict through our vendored RunModel: distogram maxabs 2.7, meanabs 0.06 --
and that residual is confounded by precision (ours ran f32 to dodge a bf16 carry
mismatch; gamma ran bf16), so it is consistent with faithful vendoring. The
structure-module code -- the thing the merge risks -- looks fine. The real
output difference comes from the features we build.

**LOCALIZED -- cluster_profile.** The only substantive feature diff (besides the
known-deliberate template_mask) is msa_feat channels 25-47, the cluster_profile
block: gamma's is a hard one-hot of the sequence, ours is the soft pseudo.
Puzzle: gamma's `_update_seq` sets `cluster_profile = where(pssm_hard, hard,
pseudo)`, identical to our update_seq_gamma -- so the difference must be
downstream in make_msa_feats / nearest_neighbor_clusters, or an artifact of the
debug dict capturing a different processing stage on each side. This is the one
thing to resolve before the oracle can be made exact.

### Revised next steps

1. Resolve the cluster_profile stage difference against gamma (make_msa_feats /
   use_cluster_profile path), so our multimer featurize reproduces gamma.
2. Re-run fwd_iso at MATCHED precision (cast gamma inputs to bf16, or run gamma
   at f32) to put a real tolerance on the forward. Expect near-exact.
3. Formalize as tests/test_af2_multimer_oracle.py against a committed gamma
   snapshot. THEN the module merge, with this as the safety net.

Good news for the merge: the structure-module numerics appear faithfully
vendored, so folding modules_multimer into modules is a refactor, not a
numerical rewrite -- exactly the case the IPA-bias worry was about.

## Measured 2026-08-07 (Stage 0 done): multimer is faithful to gamma; earlier "drift" was harness

Correction to the section above. At MATCHED f32 precision:

- **Forward is bit-exact.** Gamma's exact inputs through our RunModel reproduce
  gamma's distogram and plddt to maxabs 0.000000.
- **Featurize matches gamma to <1e-6** on every feature except the
  deliberately-divergent template_mask, ONCE the opt matches. The cluster_profile
  "hard vs soft" was a `pssm_hard` mismatch in the comparison harness: gamma sets
  `pssm_hard=True` during design (af/model.py:120, af/design.py:338), so
  cluster_profile = where(pssm_hard, hard, pseudo) = the hard one-hot. Our
  featurize honours pssm_hard identically; the test had used False.
- `msa.py` is byte-identical to gamma's (only the import path differs).

So the multimer path did NOT drift from gamma -- the code is faithful. The
scary numbers earlier were bf16-vs-f32 plus the pssm_hard harness mismatch. This
de-risks the merge further: modules_multimer is a faithful vendoring, so folding
it into modules is a pure refactor.

Oracle committed: tests/data/multimer_golden.npz + tests/test_af2_multimer_oracle.py
(hetero de-novo 12:12, f32, deterministic; pins the current -- gamma-faithful --
output so the merge cannot change numerics silently).

## Measured 2026-08-07 (Stage 2 groundwork): exact monomer->multimer param map

Diffed model_1_ptm vs model_1_multimer_v3 (both loaded, fuse=True). 135 vs 146
modules; 106 shared, 29 monomer-only, 40 multimer-only.

### Structure-module IPA fusion (the convertible core, unambiguous)

Scope: `.../structure_module/fold_iteration/`

```
monomer (fused)                          multimer (split)
q_scalar      w(384,192) b(192)   ->  q_scalar_projection      w(384,12,16) b(12,16)   reshape 192=12*16
kv_scalar     w(384,384) b(384)   ->  k_scalar_projection      w(384,12,16) b(12,16)   [:, :192]
                                        v_scalar_projection      w(384,12,16) b(12,16)   [:, 192:]
q_point_local w(384,144) b(144)   ->  q_point_projection/point_projection w(384,12,12) b(12,12)  144=12*12
kv_point_local w(384,432) b(432)  ->  k_point_projection/point_projection w(384,12,12) b(12,12)  [:, :144]
                                        v_point_projection/point_projection w(384,12,24) b(12,24)  [:, 144:] (288=12*24)
affine_update w(384,6) b(6)       ->  quat_rigid/rigid         w(384,6) b(6)            rename (verify quaternion semantics in Stage 3)
```

The split scalar biases land in the Stage 1 always-present bias slots. Native
multimer had zeros there; a converted monomer has these real values.

### Alphabet-width tensors (clean REDUCTION, not padding)

Monomer is WIDER (has a trailing gap symbol multimer's target_feat omits), so
convert DROPS the extra row/col, it does not pad:

```
left_single   (22,128) -> (21,128)   drop row 21 (gap)
right_single  (22,128) -> (21,128)
preprocess_1d (22,256) -> (21,256)
masked_msa_head/logits w(256,23)->(256,22) b(23,)->(22,)   drop col 22
```

Assumes the first 21 restypes align (they do: ARNDCQEGHILKMFPSTWYV + X, then
gap). Stage 3 confirms.

### Two scope boundaries (findings)

1. **Template embedder is architecturally different, NOT convertible.** Monomer
   uses `template_embedding/single_template_embedding/template_pair_stack/
   __layer_stack_no_state/...`; multimer uses `template_embedding_iteration/...`
   plus `template_pair_embedding_0..8`, `output_linear`, `query_embedding_norm`,
   and `~_relative_encoding/position_activations`. No reshape maps one to the
   other. => convert_monomer_params targets the **no-template path**
   (hallucination / fixbb, the monomer use cases). Templates on the multimer
   graph use NATIVE multimer weights (which is what binder design already does).
2. `template_single_embedding` (57,256) vs (34,256) is part of the same template
   fork; out of scope for the same reason.

### convert_monomer_params scope

Trunk (106 shared modules pass through) + structure module (IPA fusion +
affine_update->quat_rigid) + alphabet reduction. Template modules excluded.
Load-time only, never written to disk. Graph config for the run:
`position_scale=10`, `outer_product_mean.first=False` (Stage 3 / traced for
no-recompile).

### Relative position encoding (last non-template/non-SM difference)

Confirmed the map is CLOSED: outside templates and the structure module, the
only monomer<->multimer difference is the rel-pos encoding.

```
monomer  evoformer/pair_activiations                   w(65,128) b(128)
multimer evoformer/~_relative_encoding/position_activations  w(73,128) b(128)
```

73 = 65 rel-pos bins + 8 chain features (rel_chain/entity/sym). Convert by
ZERO-PADDING the 8 chain columns (a single-chain monomer contributes nothing
there, so the multimer graph reproduces monomer's rel-pos contribution) and
renaming. Assumes the 65 rel-pos bins come first in the multimer feature order;
Stage 3 confirms.

### Complete conversion map (closed)

1. IPA fusion (structure module) — reshape/split as tabulated above.
2. affine_update -> quat_rigid/rigid — rename.
3. Alphabet reduction — drop the gap row/col (left/right_single, preprocess_1d,
   masked_msa_head/logits).
4. rel-pos — pair_activiations -> ~_relative_encoding/position_activations,
   zero-pad 65->73, rename.
5. scalar-projection biases — from the monomer fused biases (Stage 1 slots).
6. ~105 shared modules — passthrough unchanged.
7. template embedder — excluded (no-template path; native multimer for templates).

Everything is a value/shape transform with no free parameters, so
convert_monomer_params is mechanical. Stage 3 verifies it against the monomer
graph's own output at position_scale=10, outer_product_mean.first=False.

## Monomer comes in four variants (ptm x template)

The unified graph is the multimer graph, which is ALWAYS ptm and
template-capable. Monomer weights come in four flavors:

```
model            ptm head   template weights   modules
model_1_ptm         yes         yes              135
model_1             no          yes              134
model_3_ptm         yes         no               109
model_3             no          no               108
model_1_multimer_v3 yes         yes              146
```

Consequences for convert (all consistent with "one graph, differences are
values, no recompile"):

- **pTM head is identical shape** (predicted_aligned_error_head/logits (128,64)
  + (64,) on monomer and multimer). Passthrough for a ptm source; ZERO-FILL for
  a non-ptm source (same pattern as the scalar bias). A zeroed pTM head makes
  pae/iptm meaningless, so objectives that read them need a ptm source.
- **Template-capable graph, masked off.** The graph always has the (multimer)
  template embedder; a converted monomer has no compatible template weights, so
  it runs with template_mask=0 and zero template-embedder weights -- the
  no-template path. Templates on the multimer graph use NATIVE multimer weights.
- **Cleanest converts: model_3/4/5_ptm** (ptm head present, no template weights
  -- nothing incompatible to drop). model_1/2_ptm drop their template weights.

So the load-time normalizer must, for ALL sources: zero-fill the scalar bias
(Stage 1), zero-fill the pTM head if absent, zero-fill/skip the template
embedder, and (monomer only) apply the IPA fusion + alphabet reduction + rel-pos
zero-pad. Native multimer needs only the scalar-bias zero-fill.

## Measured 2026-08-07 (Stage 3): conversion NOT yet correct -- forward diverges

convert.py (the mechanical map) is written, and Stage 3 run. Result: converted
model_3_ptm on the multimer graph (position_scale=10, outer_product_mean.first=
False, template disabled) does NOT reproduce the monomer graph:

    distogram maxabs 17.9  meanabs 0.67
    CA        maxabs 27.0
    plddt     maxabs 8.5

Isolation done -- **inputs are ruled out**: the monomer-graph and multimer-graph
featurize produce IDENTICAL model inputs for this single sequence (all 38 shared
features < 1e-5, no graph-only features). So the divergence is entirely in the
FORWARD with converted weights. Either:

  (a) a conversion bug -- prime suspects, in order:
      - IPA point-coordinate ordering: reshape (D,144)->(D,12,12) assumes the
        monomer packs points head-major, coords contiguous; if it interleaves
        differently the point projection is wrong.
      - affine_update -> quat_rigid quaternion convention (MERGE_NOTES earlier
        called it "identical"; unverified). Monomer's QuatAffine update vs
        multimer's QuatRigid may normalize/order the quaternion differently.
      - rel-pos binning: multimer's ~_relative_encoding builds the 65-bin
        feature internally; if its binning/clip differs from monomer's, the
        zero-padded 65->73 weight maps onto the wrong bins.
  (b) a genuine monomer-vs-multimer forward-code difference (the "rename +
      fusion, not new math" claim is then wrong somewhere).

Next: per-module isolation. Compare intermediate reps on identical inputs --
first the evoformer pair init (isolates rel-pos), then the evoformer output
(isolates trunk), then the structure module per layer (isolates IPA / quat).
The distogram already diverges, so the trunk differs -> start at the pair init
and the first evoformer block, not the structure module.

convert.py is committed as WIP: the mapping is mechanically complete but
UNVERIFIED until Stage 3 passes.

## Stage 3 localization (cont.): divergence is in the embedding wrapper, not the trunk

Narrowed the forward divergence further:

- **The evoformer iterations are ALREADY shared.** modules_multimer.py:416 uses
  `modules.EvoformerIteration` -- the monomer class -- and it reads
  `c.outer_product_mean.first` (modules.py:1340/1376), so the OPM ordering is
  honored by config (not a dead flag) and the per-block trunk math is one
  implementation for both. Encouraging for the merge: the iterations need no
  reconciliation.
- So the distogram (trunk) divergence must be in the multimer-specific pair/msa
  INITIALIZATION -- `modules_multimer.EmbeddingsAndEvoformer.__call__` (the
  embedding wrapper that builds the initial pair/msa reps before the shared
  iterations). That is exactly where the converted left_single/right_single/
  preprocess_1d (alphabet reduction) and position_activations (rel-pos) feed.

Remaining suspects, to isolate by instrumenting the pre-iteration pair/msa reps:
  1. alphabet reduction direction (dropped the wrong row?) -- affects
     left/right_single and preprocess_1d, which build the initial pair/single.
  2. a genuine structural difference between the monomer and multimer embedding
     wrappers (then it is not a weight bug and the wrapper must be reconciled,
     not just the weights).

Next debugging step: capture pair_activations and msa_activations right before
the first EvoformerIteration on identical inputs, monomer graph vs converted-
multimer graph, and diff. That splits (1) from (2).

## Stage 3 localization (proven): the pre-evoformer PAIR INIT diverges

Threaded the pre-evoformer pair rep out through the jit boundary (temporary
edit, reverted) and compared monomer graph vs converted-multimer graph on
identical inputs:

    pre-evoformer pair init:  maxabs 3.27  meanabs 0.20   (shape 12,12,128)

So the divergence is in `EmbeddingsAndEvoformer.__call__`'s pair CONSTRUCTION,
before any evoformer iteration -- the exact spot fed by the converted
left_single/right_single/preprocess_1d (alphabet reduction) and
pair_activiations->position_activations (rel-pos). Combined with earlier: inputs
identical, iterations shared code, OPM ordering honored -> the whole divergence
originates in the embedding wrapper's pair init.

Still (1) conversion detail vs (2) structural wrapper difference, unresolved.
Note this capture point is AFTER the extra_msa stack (shared code), so it folds
in extra_msa amplification of any earlier error.

Remaining isolation (next session):
  - capture the pair rep BEFORE the extra_msa stack (pure embedding: left/right_
    single + rel-pos), to separate the embedding from extra_msa.
  - verify monomer's relpos is exactly one_hot(clip(offset+32,0,64), 65) so the
    65->73 zero-pad maps bin-for-bin onto multimer's first 65 of its 66-dim
    relpos (dim 65 = cross-chain, zero for one chain; dims 66..72 chain feats,
    zero). If monomer's binning differs, that is the bug.
  - if both check out, it is (2): the monomer and multimer embedding wrappers
    build the pair init with different math, and the wrapper must be reconciled
    under a flag (like OPM.first), not fixed by weight conversion alone.

Status: convert.py remains WIP/UNVERIFIED. Stages 0-1 are done and committed.

## Stage 3 (breakthrough): trunk is bit-exact; bug was the alphabet pad direction

Progressive capture (pre-evoformer -> pre-extra-msa -> pre-relpos base) localized
the trunk divergence to `left+right_single`, and the fix was the ALPHABET
REDUCTION DIRECTION. The embedding wrapper pads the target features by type:

    monomer  jnp.pad(avg_target, [[0,0],[1,1]])   real restypes at indices 1..20
    multimer jnp.pad(avg_target, [[0,0],[0,1]])   real restypes at indices 0..19

So monomer's real-restype weight rows are 1..20; convert must map them onto
multimer's 0..19 by dropping the LEADING row (monomer[1:]), not the trailing one
(monomer[:21]). With that fix:

    pair_base / pair_preextra   maxabs 0.000000   (bit-exact)
    distogram                    maxabs 0.000000   (trunk bit-exact)

Confirms hypothesis (1): a conversion bug, not a structural wrapper difference --
the trunk math IS shared (EvoformerIteration), and converted monomer reproduces
it exactly once the pad convention is respected. The feature-creation unification
stays as-is; the unified graph uses multimer's pad convention and converted
monomer weights are aligned to it (drop-leading).

REMAINING: the structure module still diverges (CA maxabs 26.5, plddt 7.2). The
trunk feeding it is bit-exact, so the bug is in the IPA fusion (point-coordinate
ordering in the (D,144)->(D,12,12) reshape) or the affine_update->quat_rigid
quaternion convention. Next: isolate within fold_iteration.

## Stage 3 PASSES: converted monomer == monomer graph, bit-exact

Two more conversion bugs found in the IPA, both layout mismatches:

1. **IPA points are coord-major in monomer, head-major in multimer.** Monomer
   q_point_local (144) = split-3 into [x(48),y(48),z(48)], each (head12,point4).
   Multimer wants (head, [x4,y4,z4]). Convert must reshape (D,3,head,npts) and
   transpose coord<->head, not flat-reshape.
2. **kv is split PER HEAD, not by flat offset.** Monomer kv_scalar (384) is
   (head12, 32) split at 16 per head; kv_point (432) is (coord,head,qk4+v8) split
   at 4. My flat [:192] / [:144] slices took whole heads -- wrong.

With the alphabet drop-leading fix plus these, converted model_3_ptm on the
multimer graph (position_scale=10, outer_product_mean.first=False, templates
off) reproduces the monomer graph to f32 precision:

    distogram maxabs 0.000000
    CA        maxabs 0.000006
    plddt     maxabs 0.000004

So the affine_update->quat_rigid rename is confirmed correct too (composition
math is equivalent: q (+) q(x)update, rotated translation). The conversion is a
lossless value/shape transform, verified end to end. Stage 3 is done; convert.py
is no longer WIP. tests/test_af2_convert.py pins it.

Remaining for the full merge (Stage 4): thread convert into load_params (monomer
model_type on the multimer runtime -> auto-convert at load), make position_scale
a traced input and outer_product_mean.first a lax.cond per-model flag for the
no-recompile single graph, then collapse modules_multimer.py into modules.py.

## Stage 4 in progress: enabler landed, physical retirement remains

Done:
- `AF2Runner(on_multimer_graph=True)` routes a monomer model through the multimer
  graph (convert at load + monomer regime). Bit-exact vs native monomer (CA
  6e-6). This is the pivot: monomer no longer NEEDS its own forward code.
- Fixed make_config leaking nested config state across calls (deep-copy now) --
  a real latent bug the regime mutation would have hit in production.

Remaining, mechanical, gated by the oracles after each step:
1. Make `on_multimer_graph` the DEFAULT for monomer model types (or drop the
   monomer graph path entirely). This makes the monomer forward code dead.
2. Migrate the two monomer-oracle tests in test_af2_runner.py: they compare the
   native monomer forward to v1 feature-for-feature; that path is going away, so
   its guarantee moves to test_af2_convert (converted monomer == the retired
   monomer graph, already committed). Keep a v1-feature-for-feature check by
   pinning it to a committed golden if desired.
3. Fold the three files into their base names and delete them:
   all_atom_multimer.py -> all_atom.py, folding_multimer.py -> folding.py,
   modules_multimer.py -> modules.py. Delete the now-dead monomer
   EmbeddingsAndEvoformer / StructureModule.
4. model.py: one AlphaFold, no use_multimer branch. runner.py: use_multimer
   becomes internal.
5. For the no-recompile single graph: position_scale -> traced input,
   outer_product_mean.first -> lax.cond per-model bool.
6. Follow-up (user): profile the unified code for speedups.

Definition of done: `ls colabdesign2/af2/alphafold/model/*multimer*` is empty and
all oracles + full suite pass.

## Stage 4 file-merge: concrete order + collision data (measured)

Prereqs DONE: monomer golden committed (tests/data/monomer_golden.npz),
test_af2_convert repointed to it (survives deleting the native monomer graph),
on_multimer_graph enabler verified, make_config deep-copied.

The three files have SAME-NAME / DIFFERENT-BODY collisions (measured), so the
merge is replace-not-append, and needs the monomer forward dead first:

    all_atom vs all_atom_multimer:  13 colliding names (atom37_to_frames,
      frame_aligned_point_error, torsion_angles_to_frames, ... all DIFFERENT)
    folding vs folding_multimer:     6 (FoldIteration, InvariantPointAttention,
      MultiRigidSidechain, StructureModule, l2_normalize, squared_difference)
    modules vs modules_multimer:     5 (AlphaFold, AlphaFoldIteration,
      EmbeddingsAndEvoformer, SingleTemplateEmbedding, TemplateEmbedding)

The multimer path depends on all_atom_multimer's OWN versions of the colliding
names (folding_multimer/modules_multimer call 18 all_atom_multimer functions).
monomer folding.py also calls 3 (atom14_to_atom37, get_atom14_mask,
get_atom37_mask) -- those go dead once the monomer forward is deleted.

Execution order (verify oracles after EACH; revert to last green on any drift):
1. Route ALL monomer types through the multimer graph by default (runner:
   on_multimer_graph default for monomer; monomer+templates -> use native
   multimer weights). model.py: drop the use_multimer branch, one AlphaFold.
   -> monomer forward (modules.AlphaFold/AlphaFoldIteration/EmbeddingsAndEvoformer,
   folding StructureModule/FoldIteration/IPA/MultiRigidSidechain) is now DEAD.
2. Migrate test_af2_runner's two monomer feature-oracle tests: their forward path
   is gone; keep a featurize-vs-v1 check if wanted, move the numeric guarantee to
   the golden (already done).
3. Bottom-up file merge, deleting the dead monomer versions and keeping the
   multimer ones (rename references all_atom_multimer->all_atom,
   folding_multimer->folding, modules_multimer->modules):
   a. all_atom_multimer -> all_atom.py
   b. folding_multimer  -> folding.py
   c. modules_multimer  -> modules.py
   Delete the three *_multimer files.
4. Full suite + both oracles green. `ls *multimer*` empty.
5. no-recompile: position_scale traced, outer_product_mean.first via lax.cond.
6. speedup profiling pass.
