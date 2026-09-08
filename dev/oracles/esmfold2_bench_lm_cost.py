"""How much does ESM-C cost at inference?  Time + peak VRAM, with and without."""
import time, numpy as np, torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features
rng = np.random.RandomState(0)
AA = "ACDEFGHIKLMNPQRSTVWY"

def bench(m, seq, use_lm, steps=200, loops=3, reps=2):
    feats = {k: v.cuda() for k, v in prepare_protein_features(seq).items()}
    if not use_lm:
        feats.pop('input_ids')
    ts = []
    peak = 0.0
    for r in range(reps + 1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            m(**feats, num_loops=loops, num_diffusion_samples=1, num_sampling_steps=steps)
        torch.cuda.synchronize()
        if r:
            ts.append(time.time() - t0)
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    return float(np.median(ts)), peak

print("loading with ESM-C (bf16)...", flush=True)
m = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16").cuda().eval()
print("resident after load: %.2f GiB" % (torch.cuda.memory_allocated() / 2**30), flush=True)
print("%-6s  %-21s  %-21s  %s" % ("len", "with ESM-C", "no ESM-C", "LM share"), flush=True)
for L in (64, 128, 256, 384, 512):
    seq = ''.join(rng.choice(list(AA), L))
    t1, m1 = bench(m, seq, True)
    t2, m2 = bench(m, seq, False)
    print("%-6d  %6.2f s %6.2f GiB   %6.2f s %6.2f GiB   %4.1f%% of runtime" % (
        L, t1, m1, t2, m2, 100 * (t1 - t2) / t1), flush=True)
