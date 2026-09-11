"""Native's OWN floor for one 48-block pairformer pass on a FIXED input.

The dumped pf_in is fed to native's PairFormerStack twice -- TF32 on (what of3
sets itself) and TF32 off -- and the two outputs are compared. That is the
resolution this cell has; our disagreement with native only means something
above it.
"""
import os, sys
import numpy as np
import torch

TREE = os.environ.get('TREE', '/home/ubuntu/openfold-3-v050')
CKPT = os.environ.get('CKPT', '/home/ubuntu/of3-ob-174k.pt')
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.expanduser('~/of3_stub'))
from openfold3.core.model.latent.pairformer import PairFormerStack

d = np.load(sys.argv[1])
s0 = torch.tensor(d['pf_in_s'], dtype=torch.float32).cuda()
z0 = torch.tensor(d['pf_in_z'], dtype=torch.float32).cuda()
tok = torch.tensor(d['batch_token_mask'], dtype=torch.float32).cuda()
pair_mask = tok[..., None] * tok[..., None, :]

raw = torch.load(CKPT, map_location='cpu', weights_only=False)
raw = raw.get('state_dict', raw)
sub = {k[len('pairformer_stack.'):]: v for k, v in raw.items()
       if k.startswith('pairformer_stack.')}
n_blocks = 1 + max(int(k.split('.')[1]) for k in sub if k.startswith('blocks.'))
c_z = sub['blocks.0.pair_stack.pair_transition.layer_norm.weight'].shape[0]
c_s = sub['blocks.0.attn_pair_bias.layer_norm_a.weight'].shape[0]
heads_bias = sub['blocks.0.attn_pair_bias.linear_z.weight'].shape[0]
print('ckpt: %d blocks c_s %d c_z %d heads %d' % (n_blocks, c_s, c_z, heads_bias))

out = {}
for tag, tf32 in (('tf32', True), ('fp32', False)):
  torch.backends.cuda.matmul.allow_tf32 = tf32
  torch.backends.cudnn.allow_tf32 = tf32
  torch.set_float32_matmul_precision('high' if tf32 else 'highest')
  stack = PairFormerStack(c_s=c_s, c_z=c_z, c_hidden_pair_bias=24,
                          no_heads_pair_bias=heads_bias, c_hidden_mul=c_z,
                          c_hidden_pair_att=32, no_heads_pair=4,
                          no_blocks=n_blocks, transition_type='swiglu',
                          transition_n=4, pair_dropout=0.0,
                          fuse_projection_weights=False, blocks_per_ckpt=None,
                          inf=1e9).cuda().eval()
  missing, unexpected = stack.load_state_dict(sub, strict=False)
  assert not missing, 'native stack partly at random init: %s' % list(missing)[:3]
  with torch.no_grad():
    s, z = stack(s=s0.clone(), z=z0.clone(),
                 single_mask=tok, pair_mask=pair_mask)
  out[tag] = (s.float().cpu().numpy(), z.float().cpu().numpy())

for i, nm in ((0, 's'), (1, 'z')):
  x = out['tf32'][i].astype(np.float64).ravel()
  y = out['fp32'][i].astype(np.float64).ravel()
  print('  NATIVE FLOOR %s corr %.9f  max|d| %.4f  rms %.3f'
        % (nm, np.corrcoef(x, y)[0, 1], np.abs(x - y).max(),
           np.sqrt((y ** 2).mean())))
np.savez(sys.argv[2], s_tf32=out['tf32'][0], z_tf32=out['tf32'][1],
         s_fp32=out['fp32'][0], z_fp32=out['fp32'][1])
