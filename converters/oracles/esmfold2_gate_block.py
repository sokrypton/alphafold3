import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'/home/ubuntu/alphafold3')
from converters import esmfold2 as E

def layer_norm(x, p, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + eps) * p['scale'] + p['offset']

def tri_mul(z, p, mask, outgoing):
    zn = layer_norm(z, {'scale': p['left_norm_input/scale'], 'offset': p['left_norm_input/offset']})
    proj = zn @ p['projection/weights']          # (..., 2h) interleaved a/b
    gate = jax.nn.sigmoid(zn @ p['gate/weights'])
    routed = proj * gate * mask[..., None]
    a = routed[..., 0::2]                        # interleave convention
    b = routed[..., 1::2]
    if outgoing:
        c = jnp.einsum('ikd,jkd->ijd', a, b)
    else:
        c = jnp.einsum('kjd,kid->ijd', a, b)   # AF3's incoming equation (weights pre-swapped)
    c = layer_norm(c, {'scale': p['center_norm/scale'], 'offset': p['center_norm/offset']})
    out = c @ p['output_projection/weights']
    return out * jax.nn.sigmoid(zn @ p['gating_linear/weights'])

def transition(x, p):
    h = layer_norm(x, {'scale': p['input_layer_norm/scale'], 'offset': p['input_layer_norm/offset']})
    h = h @ p['transition1/weights']
    n = h.shape[-1] // 2
    return (jax.nn.silu(h[..., :n]) * h[..., n:]) @ p['transition2/weights']

def pair_block(z, p, mask):
    z = z + tri_mul(z, {k[len('triangle_multiplication_outgoing/'):]: v
                        for k, v in p.items() if k.startswith('triangle_multiplication_outgoing/')}, mask, True)
    z = z + tri_mul(z, {k[len('triangle_multiplication_incoming/'):]: v
                        for k, v in p.items() if k.startswith('triangle_multiplication_incoming/')}, mask, False)
    z = z + transition(z, {k[len('pair_transition/'):]: v
                           for k, v in p.items() if k.startswith('pair_transition/')})
    return z


sd = dict(np.load('esmfold2_sd.npz'))
r  = dict(np.load('tblock.npz'))
z  = jnp.asarray(r['z_in'][0])
mask = jnp.ones(z.shape[:2])
p = E.pair_only_block(sd, 'folding_trunk.blocks.0')

def rep(tag, ours, native):
    ours = np.asarray(ours, np.float64); native = np.asarray(native, np.float64)
    print("%-28s relerr %.3e  corr %.10f" % (
        tag, np.abs(ours-native).max()/np.abs(native).max(),
        np.corrcoef(ours.ravel(), native.ravel())[0,1]))

sub = lambda pre: {k[len(pre):]: v for k, v in p.items() if k.startswith(pre)}
rep('tri_mul_out',  tri_mul(z, sub('triangle_multiplication_outgoing/'), mask, True),  r['trimul_out'][0])
rep('tri_mul_in',   tri_mul(z, sub('triangle_multiplication_incoming/'), mask, False), r['trimul_in'][0])
rep('pair_transition (+res)', z + transition(z, sub('pair_transition/')), r['trans_out'][0])
rep('full block',   pair_block(z, p, mask), r['block_out'][0])
