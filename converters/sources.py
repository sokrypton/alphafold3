"""Where each model's ORIGINAL (unconverted) checkpoint comes from.

This table is converter-side only. Nothing under `src/alphafold3` reads it: the
runtime never downloads a PyTorch checkpoint and never imports torch. Conversion
is an offline step whose product -- a single `<model>.bin.zst` of haiku params --
is what the runtime loads (see `alphafold3.model.model_registry`).

`files` rather than `url`/`file` means the model is published as several archives
(chai-1's five TorchScript modules plus a distogram head), so the converter is
handed the directory and finds its own pieces.
"""

from __future__ import annotations

SOURCES = {
    # OpenFold3 (AlQuraishi Lab). Public, no sign-request needed.
    'openfold3': dict(
        url='https://openfold.s3.amazonaws.com/staging/of3-p2-155k.pt',
        file='of3-p2-155k.pt'),
    # OpenBind, OpenFold3 v0.5.0. A different bucket from preview-2 -- the one
    # their own setup_openfold.py uses (S3_BUCKET = "openfold3-data").
    # Upstream deprecates preview-2 as of this release.
    'openbind': dict(
        url='https://openfold3-data.s3.amazonaws.com/openfold3-parameters/'
            'of3-ob-2025-06-30-174k.pt',
        file='of3-ob-2025-06-30-174k.pt'),
    # Protenix v0.5.0 base, templateless.
    'protenix05': dict(
        url='https://protenix.tos-cn-beijing.volces.com/checkpoint/'
            'protenix_base_default_v0.5.0.pt',
        file='protenix_base_default_v0.5.0.pt'),
    # Protenix-v1, the 368 M release that preceded v2.
    'protenix1': dict(
        url='https://protenix.tos-cn-beijing.volces.com/checkpoint/'
            'protenix_base_default_v1.0.0.pt',
        file='protenix_base_default_v1.0.0.pt'),
    # A later training run of the same graph as protenix1.
    'protenix1_20250630': dict(
        url='https://protenix.tos-cn-beijing.volces.com/checkpoint/'
            'protenix_base_20250630_v1.0.0.pt',
        file='protenix_base_20250630_v1.0.0.pt'),
    # Protenix's small model types, from the same publisher as protenix-v2.
    'protenix_mini': dict(
        url='https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_mini_default_v0.5.0.pt',
        file='protenix_mini_default_v0.5.0.pt'),
    'protenix_tiny': dict(
        url='https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_tiny_default_v0.5.0.pt',
        file='protenix_tiny_default_v0.5.0.pt'),
    # IntelliFold-v2 (IntelligenAI). Stock-AF3 module tree at widened channels.
    'intellifold2': dict(
        url='https://huggingface.co/intelligenAI/intellifold/resolve/main/intellifold_v2.pt',
        file='intellifold_v2.pt'),
    # OpenDDE (Aureka Research). Upstream publishes no version number, hence the
    # bare name. Pinned to the commit we converted.
    'opendde': dict(
        url='https://huggingface.co/aurekaresearch/OpenDDE/resolve/'
            'eddd563ce96571f784012edd8f045181c8f8627d/opendde.pt',
        file='opendde.pt'),
    # Boltz-2 (jwohlwend/boltz, MIT).
    'boltz2': dict(
        url='https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt',
        file='boltz2_conf.ckpt'),
    # Protenix-v2 (ByteDance, Apache-2.0). The official CDN 403s; this is the
    # community HF mirror (SHA256 verified against the CDN copy when it was
    # still reachable).
    'protenix2': dict(
        url='https://huggingface.co/TMF001/protenix-v2-weights/resolve/main/protenix-v2.pt',
        file='protenix-v2.pt'),
    # RoseTTAFold3 (RosettaCommons foundry). We use the EMA shadow weights.
    'rosettafold3': dict(
        url='https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt',
        file='rf3_foundry_01_24_latest_remapped.ckpt'),
    # chai-1 (chaidiscovery, Apache-2.0). Five frozen TorchScript archives.
    # distogram_head.pt is NOT part of stock chai -- upstream has no distogram
    # head at all (its trunk returns only (single, pair)) -- it comes from
    # sokrypton/chai-lab@dgram, trained on the trunk pair representation.
    # The "depencencies" in the asset URLs is chai's own typo; the correctly
    # spelled path 404s.
    'chai1': dict(files=[
        ('https://chaiassets.com/chai1-inference-depencencies/models_v2/feature_embedding.pt',
         'models_v2/feature_embedding.pt'),
        ('https://chaiassets.com/chai1-inference-depencencies/models_v2/token_embedder.pt',
         'models_v2/token_embedder.pt'),
        ('https://chaiassets.com/chai1-inference-depencencies/models_v2/trunk.pt',
         'models_v2/trunk.pt'),
        ('https://chaiassets.com/chai1-inference-depencencies/models_v2/diffusion_module.pt',
         'models_v2/diffusion_module.pt'),
        ('https://chaiassets.com/chai1-inference-depencencies/models_v2/confidence_head.pt',
         'models_v2/confidence_head.pt'),
        ('https://raw.githubusercontent.com/sokrypton/chai-lab/dgram/chai_lab/model/distogram_head.pt',
         'models_v2/distogram_head.pt'),
    ]),
    # AlphaFold3 itself is not converted: DeepMind's release is already a haiku
    # blob, and its terms require you to request it rather than fetch a URL.
}
