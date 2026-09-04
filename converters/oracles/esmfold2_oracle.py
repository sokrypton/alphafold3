"""ESMFold2 native oracle: fold 6MRR and dump every seam we will need to gate against."""
import os, sys, math, numpy as np, torch
torch.manual_seed(0)
SEQ = "GWSTELEKHREELKKEFLKKEGITNVEIRIDNGRLEVRVEGGTEERLKRFLEELRQKLEKKGYTVDDIKIE"
OUT = os.path.dirname(os.path.abspath(__file__))

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import (
    OUTPUT_TO_PDB_FEATURE_KEYS, prepare_protein_features, output_to_pdb)

print("loading (folding model + ESMC-6B bf16)...", flush=True)
model = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16").cuda().eval()
print("loaded", flush=True)

feats = {k: v.cuda() for k, v in prepare_protein_features(SEQ).items()}
for k, v in sorted(feats.items()):
    print("  feat %-24s %-18s %s" % (k, tuple(v.shape), v.dtype))

dump = {}
def cap(name):
    def hook(mod, inp, out):
        def rec(tag, t):
            if torch.is_tensor(t):
                dump[tag] = t.detach().float().cpu().numpy()
        if isinstance(out, tuple):
            for i, t in enumerate(out): rec("%s.out%d" % (name, i), t)
        else:
            rec(name + ".out", out)
        for i, t in enumerate(inp): rec("%s.in%d" % (name, i), t)
    return hook

hooks = []
for name in ["inputs_embedder", "z_init_1", "z_init_2", "rel_pos", "token_bonds",
             "language_model", "lm_encoder", "folding_trunk", "parcae_input_norm",
             "parcae_readout", "parcae_coda", "distogram_head", "msa_encoder"]:
    m = getattr(model, name, None)
    if m is not None:
        hooks.append(m.register_forward_hook(cap(name)))

# freeze the random initial pair state so we can inject the identical one in JAX
orig_init = model._init_pair_state
def seeded_init(ref):
    g = torch.Generator(device="cpu").manual_seed(1234)
    std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
    s = torch.empty(ref.shape, dtype=torch.float32)
    torch.nn.init.trunc_normal_(s, 0.0, std, -3*std, 3*std, generator=g)
    dump["parcae.init_state"] = s.numpy()
    return s.to(device=ref.device, dtype=ref.dtype)
model._init_pair_state = seeded_init

# the parcae dynamics are static -> fold to plain arrays
a, b = model._discretized_dynamics()
dump["parcae.a"] = a.detach().float().cpu().numpy()
dump["parcae.b"] = b.detach().float().cpu().numpy()

with torch.no_grad():
    out = model(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=200)

for h in hooks: h.remove()
for k in ("distogram_logits", "sample_atom_coords", "plddt", "ptm", "complex_plddt"):
    if k in out and torch.is_tensor(out[k]):
        dump["out." + k] = out[k].detach().float().cpu().numpy()

for k in OUTPUT_TO_PDB_FEATURE_KEYS: out[k] = feats[k]
open(os.path.join(OUT, "esmfold2_6mrr.pdb"), "w").write(output_to_pdb(out))
for k, v in feats.items(): dump["feat." + k] = v.detach().float().cpu().numpy()
np.savez_compressed(os.path.join(OUT, "esmfold2_dump.npz"), **dump)

print("\n=== DUMPED %d tensors ===" % len(dump))
for k in sorted(dump): print("  %-46s %s" % (k, dump[k].shape))
