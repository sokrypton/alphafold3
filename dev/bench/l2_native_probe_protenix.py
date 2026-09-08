"""Can protenix's token DiffusionTransformer be built from the checkpoint and run
on synthetic inputs? That is the risky half of an L2 gate."""
import sys, types, numpy as np, torch

# the fused-LayerNorm stub, same contract as the L1 harness needs
ext = types.ModuleType('fast_layer_norm_cuda_v2')
def _ln(x, shape, w=None, b=None, eps=1e-5):
    dims = tuple(range(x.dim() - len(shape), x.dim()))
    mean = x.mean(dim=dims, keepdim=True); var = x.var(dim=dims, unbiased=False, keepdim=True)
    inv = torch.rsqrt(var + eps); out = (x - mean) * inv
    if w is not None: out = out * w
    if b is not None: out = out + b
    return out, mean.squeeze(-1), inv.squeeze(-1)
# Dispatch on ARITY rather than enumerating entry points. The trunk needed
# three (forward, forward_with_weight_affine, forward_with_both_affine); the
# diffusion transformer also calls forward_none_affine, and enumerating means
# discovering each one by AttributeError.
class _Ext(types.ModuleType):
  def __getattr__(self, name):
    if not name.startswith('forward'):
      raise AttributeError(name)
    def fn(x, shape, *rest):
      eps = rest[-1] if rest and isinstance(rest[-1], float) else 1e-5
      tens = [r for r in rest if torch.is_tensor(r)]
      w = tens[0] if len(tens) > 0 else None
      b = tens[1] if len(tens) > 1 else None
      return _ln(x, shape, w, b, eps)
    return fn
sys.modules.setdefault('fast_layer_norm_cuda_v2', _Ext('fast_layer_norm_cuda_v2'))

from protenix.model.modules.transformer import DiffusionTransformer

sd = torch.load('/home/ubuntu/protenix_weights/protenix-v2.pt', map_location='cpu', weights_only=False)
sd = sd.get('model', sd)
P = 'module.diffusion_module.diffusion_transformer.'
sub = {k[len(P):]: v for k, v in sd.items() if k.startswith(P)}
print('tensors under %r: %d' % (P, len(sub)))
nb = 1 + max(int(k.split('.')[1]) for k in sub if k.startswith('blocks.'))
# a is the token single acting dim; s the conditioning; z the pair bias source
# read off the checkpoint, printed rather than guessed (four wrong guesses)
c_a = sub['blocks.0.attention_pair_bias.attention.linear_q.weight'].shape[0]
c_s = sub['blocks.0.attention_pair_bias.layernorm_a.layernorm_s.weight'].shape[0]
c_z = sub['blocks.0.attention_pair_bias.layernorm_z.weight'].shape[0]
heads = sub['blocks.0.attention_pair_bias.linear_nobias_z.weight'].shape[0]
print('derived: n_blocks %d  c_a %s  c_s %s  c_z %s  heads %s' % (nb, c_a, c_s, c_z, heads))
print('sample keys:', sorted(k for k in sub if k.startswith('blocks.0.'))[:6])

net = DiffusionTransformer(c_a=c_a, c_s=c_s, c_z=c_z, n_blocks=nb, n_heads=heads)
miss, unexp = net.load_state_dict(sub, strict=False)
print('load: %d missing, %d unexpected %s' % (len(miss), len(unexp), list(miss)[:2]))
n = 68
rng = np.random.default_rng(0)
a = torch.tensor((rng.normal(size=(1, n, c_a)) * 0.5).astype(np.float32))
s_ = torch.tensor((rng.normal(size=(1, n, c_s)) * 0.5).astype(np.float32))
z_ = torch.tensor((rng.normal(size=(1, n, n, c_z)) * 0.5).astype(np.float32))
net.eval()
with torch.no_grad():
    out = net(a, s_, z_)
print('L2NATIVE ok: output', tuple(out.shape), 'rms %.4f' % float(out.pow(2).mean().sqrt()))
