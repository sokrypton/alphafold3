# AF3 dropout: SI vs OpenFold3 (verified agree) — spec for re-adding during design

Source of truth: AF3 SI (41586_2024_7487_MOESM1_ESM.pdf) Algorithms 8/16/17 + §5.5.
Cross-checked against OpenFold3 (github aqlaboratory/openfold-3). THEY AGREE.

AF3 has ONE PairformerStack block (Alg 17) reused in 3 places, each with the same
per-block pair dropout:
  - trunk Pairformer (Alg 17, Nblock 48)
  - template embedder (Alg 16 -> PairformerStack)      [binders use this]
  - MSA module pair updates (Alg 8 lines 9-12)

Per pair-stack block (rate 0.25):
  z += DropoutRowwise0.25   ( TriangleMultiplicationOutgoing )
  z += DropoutRowwise0.25   ( TriangleMultiplicationIncoming )
  z += DropoutRowwise0.25   ( TriangleAttentionStartingNode )
  z += DropoutColumnwise0.25( TriangleAttentionEndingNode )     # COLUMNWISE
  (pair transition: NO dropout — SwiGLU)
MSA module also: DropoutRowwise0.15( MSAPairWeightedAveraging ).

Row vs column (OF3 primitives/dropout.py): mask shared along an axis of the pair
tensor z (…, i, j, c). Rowwise = share over i (batch_dim -3). Columnwise = share
over j (batch_dim -2). OF3 implements the columnwise end-node op as
transpose(-2,-3) + rowwise, which equals columnwise on the original — matches SI.

Our vendored DeepMind AF3 STRIPPED all of this (inference-only). To re-add for
design, follow AF2's no-recompile pattern (af2/.../modules.py apply_dropout +
dropout_wrapper): dropout ALWAYS in the graph, rate = jnp.where(use_dropout,
0.25, 0). rate=0 => bernoulli(1.0)=all-ones => exact identity, so toggling never
recompiles and prediction stays bit-identical. use_dropout is a TRACED opt flag;
key threaded per op.

## AF3 vs OF3 divergence check (per user: gate on global_config.model only if they differ)
Verified OF3's TRAINING config (openfold-3 of3_all_atom/config/model_config.py):
  trunk pairformer pair_dropout=0.25 ; template dropout_rate=0.25 ;
  msa_module msa_dropout=0.15, pair_dropout=0.25.
These MATCH the AF3 SI exactly (rowwise 0.25 out/in/start, columnwise 0.25 end;
msa rowwise 0.15). So the DESIGN-RELEVANT dropout (pair stack feeding the
distogram, incl. template reuse + msa-module pair updates) AGREES between AF3 and
OF3 -> the PairFormerIteration dropout is MODE-AGNOSTIC, no per-model gate needed.

ONE divergence, OUTSIDE the design gradient path: OF3's diffusion
StructureModuleTransition has dropout 0.25; the AF3 SI (sec 5.5) lists dropout
only in MsaModule/Pairformer/template, NOT the diffusion module. Irrelevant to
distogram design (sampler off) and to diffusion='forward' i_pae (sampler
detached). IF diffusion-path dropout is ever added, gate it on global_config.model:
the OPENFOLD3_LINEAGE names -> 0.25 in StructureModuleTransition, 'alphafold3' -> none.
