"""Run native PairUpdateBlock 0 in pure fp32 (no autocast) on a fixed input."""
import numpy as np, torch, glob
from safetensors import safe_open
from transformers.models.esmfold2.modeling_esmfold2_common import PairUpdateBlock

torch.manual_seed(0)
p = glob.glob('/home/ubuntu/.cache/huggingface/hub/models--biohub--ESMFold2/snapshots/*/model.safetensors')[0]
f = safe_open(p, 'pt')
sd = {k[len('folding_trunk.blocks.0.'):]: f.get_tensor(k).float()
      for k in f.keys() if k.startswith('folding_trunk.blocks.0.')}

blk = PairUpdateBlock(d_pair=256, expansion_ratio=4).float().eval()
missing, unexpected = blk.load_state_dict(sd, strict=False)
print("missing:", [m for m in missing if '_extra' not in m], "unexpected:", unexpected)
blk.set_chunk_size(None)

L = 24
z = torch.randn(1, L, L, 256) * 0.5
out = {}
with torch.no_grad():
    out['z_in'] = z.numpy()
    out['block_out'] = blk(z).numpy()
    out['trimul_out'] = blk.tri_mul_out(z).numpy()
    out['trimul_in'] = blk.tri_mul_in(z).numpy()
    out['trans_out'] = blk.pair_transition(z).numpy()
np.savez('tblock.npz', **out)
print("dumped", {k: v.shape for k, v in out.items()})
