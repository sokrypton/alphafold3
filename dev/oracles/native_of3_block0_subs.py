"""Native block-0 sub-module deltas, on the REAL dumped pairformer input.

The pairformer block adds each sub-module's output to z, so each hook here
captures a DELTA and the z it was computed from -- which is what a per-module
comparison needs ([[esmfold2-trunk-per-block-method]]: compare the UPDATE, not
the running total, or every later module inherits the earlier one's error).

TF32 is OFF: this is the reference, and its own floor on a fixed input is
max|d| ~2 over 48 blocks, so one block is far below that.
"""
import os, sys
import numpy as np
import torch

TREE = os.environ.get('TREE', '/home/ubuntu/openfold-3-v050')
CKPT = os.environ.get('CKPT', '/home/ubuntu/of3-ob-174k.pt')
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.expanduser('~/of3_stub'))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
from openfold3.core.model.latent.pairformer import PairFormerStack

d = np.load(sys.argv[1])
out_npz = sys.argv[2]
s0 = torch.tensor(d['pf_in_s'], dtype=torch.float32).cuda()
z0 = torch.tensor(d['pf_in_z'], dtype=torch.float32).cuda()
tok = torch.tensor(d['batch_token_mask'], dtype=torch.float32).cuda()
pair_mask = tok[..., None] * tok[..., None, :]

raw = torch.load(CKPT, map_location='cpu', weights_only=False)
raw = raw.get('state_dict', raw)
sub = {k[len('pairformer_stack.'):]: v for k, v in raw.items()
       if k.startswith('pairformer_stack.')}
c_z = sub['blocks.0.pair_stack.pair_transition.layer_norm.weight'].shape[0]
c_s = sub['blocks.0.attn_pair_bias.layer_norm_a.weight'].shape[0]
heads_bias = sub['blocks.0.attn_pair_bias.linear_z.weight'].shape[0]
net = PairFormerStack(c_s=c_s, c_z=c_z, c_hidden_pair_bias=24,
                      no_heads_pair_bias=heads_bias, c_hidden_mul=c_z,
                      c_hidden_pair_att=32, no_heads_pair=4,
                      no_blocks=48, transition_type='swiglu', transition_n=4,
                      pair_dropout=0.0, fuse_projection_weights=False,
                      blocks_per_ckpt=None, inf=1e9)
missing, _ = net.load_state_dict(sub, strict=False)
assert not missing, list(missing)[:3]
net.blocks = net.blocks[:1]
net = net.cuda().eval()

caught = {}
blk = net.blocks[0]
targets = {
    'tri_mul_out': blk.pair_stack.tri_mul_out,
    'tri_mul_in': blk.pair_stack.tri_mul_in,
    'tri_att_start': blk.pair_stack.tri_att_start,
    'tri_att_end': blk.pair_stack.tri_att_end,
    'pair_transition': blk.pair_stack.pair_transition,
    'attn_pair_bias': blk.attn_pair_bias,
    'single_transition': blk.single_transition,
}
for name, mod in targets.items():
  def mk(name, mod):
    orig = mod.forward

    def f(*a, **kw):
      ins = [x for x in list(a) + list(kw.values()) if torch.is_tensor(x)]
      if ins:
        caught[name + '_in'] = ins[0].detach().float().cpu().numpy()
      out = orig(*a, **kw)
      t = out[0] if isinstance(out, (tuple, list)) else out
      caught[name + '_out'] = t.detach().float().cpu().numpy()
      return out
    mod.forward = f
  mk(name, mod)

with torch.no_grad():
  s1, z1 = net(s=s0.clone(), z=z0.clone(), single_mask=tok, pair_mask=pair_mask)
caught['block0_s'] = s1.float().cpu().numpy()
caught['block0_z'] = z1.float().cpu().numpy()
np.savez(out_npz, **caught)
print('dumped', out_npz)
for k in sorted(caught):
  v = caught[k]
  print('  %-22s %-18s rms %.4f' % (k, v.shape, float(np.sqrt((v.astype(np.float64)**2).mean()))))
