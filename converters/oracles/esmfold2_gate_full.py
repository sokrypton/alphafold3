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


def run_stack(z, p, pre, n, mask):
    for i in range(n):
        z = pair_block(z, {k[len(pre):]: v[i] for k, v in p.items() if k.startswith(pre)}, mask)
    return z

def lm_shim(hidden, p):
    """[L, 81, 2560] ESM-C hidden states -> [L, L, 256] pair."""
    x = layer_norm(hidden, {'scale': p['language_model/lm_norm/scale'],
                            'offset': p['language_model/lm_norm/offset']})
    x = x @ p['language_model/lm_projection/weights']                 # [L,81,256]
    x = jnp.einsum('k,lkc->lc', p['language_model/combine'], x)       # learned layer mix
    x = x @ p['language_model/downproject/weights'] + p['language_model/downproject/bias']
    pair = jnp.concatenate([x[:, None] * x[None, :], x[:, None] - x[None, :]], -1)
    pair = pair @ p['language_model/pair_mlp_1/weights'] + p['language_model/pair_mlp_1/bias']
    pair = jax.nn.gelu(pair, approximate=False)                       # torch nn.GELU is exact
    pair = pair @ p['language_model/pair_mlp_2/weights'] + p['language_model/pair_mlp_2/bias']
    return layer_norm(pair, {'scale': p['language_model/pair_norm/scale'],
                             'offset': p['language_model/pair_norm/offset']})


sd = dict(np.load('esmfold2_sd.npz'))
r  = dict(np.load('esmfold2_det.npz'))
dims = E.derive_dims(sd)
p = E.map_trunk(sd, dims)

def rep(tag, ours, native, warn=3e-2):
    ours = np.asarray(ours, np.float64); native = np.asarray(np.squeeze(native), np.float64)
    e = np.abs(ours-native).max()/max(np.abs(native).max(), 1e-9)
    c = np.corrcoef(ours.ravel(), native.ravel())[0,1]
    print("%-36s relerr %.3e  corr %.8f %s" % (tag, e, c, '' if e < warn else ' <-- CHECK'))
    return c

x_in = jnp.asarray(r['x_inputs.0'][0]); L = x_in.shape[0]
mask = jnp.ones((L, L))
lm_z = lm_shim(jnp.asarray(r['lm_hidden'][0]), p)
rep('language_model shim', lm_z, r['lm_z.0'])
z_init = ((x_in @ p['z_init_1/weights'])[:, None] + (x_in @ p['z_init_2/weights'])[None, :]
          + jnp.asarray(r['rel_pos.0'][0])
          + jnp.asarray(r['token_bonds.0'][0]))
lm_ref = run_stack(lm_z, p, 'lm_encoder/', dims['n_lm_encoder'], mask)
rep('lm_encoder (chained)', lm_ref, r['lmenc.0'])

z = jnp.asarray(r['parcae.init_state'][0])
a = jnp.asarray(p['parcae_a']); bT = jnp.asarray(p['parcae_b/weights'])
print("\n--- 4 parcae loops, fully chained ---")
for i in range(4):
    inject = layer_norm(z_init + lm_ref, {'scale': p['parcae_input_norm/scale'],
                                          'offset': p['parcae_input_norm/offset']})
    rep('  loop%d parcae_input_norm' % i, inject, r['pnorm.%d' % i])
    z = a * z + inject @ bT
    rep('  loop%d recurrence -> trunk in' % i, z, r['pre_trunk_in.%d' % i])
    z = run_stack(z, p, 'folding_trunk/', dims['n_trunk'], mask)
    rep('  loop%d trunk out (48 blk)' % i, z, r['trunk.%d' % i], warn=6e-2)

print()
z = z @ p['parcae_readout/weights']
rep('parcae_readout', z, r['readout.0'])
z = run_stack(z, p, 'parcae_coda/', dims['n_coda'], mask)
rep('parcae_coda', z, r['coda.0'], warn=6e-2)
logits = (z + z.transpose(1,0,2)) @ p['distogram/weights'] + p['distogram/bias']
c = rep('distogram (END TO END)', logits, r['out.distogram_logits'], warn=1e-1)
print("\nEND-TO-END TRUNK CORR: %.6f" % c)
