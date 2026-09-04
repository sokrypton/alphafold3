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
r  = dict(np.load('esmfold2_msa.npz'))
dims = E.derive_dims(sd)
p = E.map_trunk(sd, dims)
def sub(pre, q=None):
    q = q if q is not None else p
    return {k[len(pre):]: v for k, v in q.items() if k.startswith(pre)}

def rep(tag, ours, native, warn=3e-2):
    ours = np.asarray(ours, np.float64); native = np.asarray(np.squeeze(native), np.float64)
    e = np.abs(ours-native).max()/max(np.abs(native).max(), 1e-9)
    c = np.corrcoef(ours.ravel(), native.ravel())[0,1]
    print("%-34s relerr %.3e  corr %.8f %s" % (tag, e, c, '' if e < warn else ' <-- CHECK'))
    return c

def opm(m, mmask, q):
    """m [L,M,c], mmask [L,M] -> pair.  NOTE: Wout(outer)/n_valid, bias scaled too."""
    mn = layer_norm(m, {'scale': q['layer_norm/scale'], 'offset': q['layer_norm/offset']})
    a = (mn @ q['left_projection/weights']) * mmask[..., None]
    b = (mn @ q['right_projection/weights']) * mmask[..., None]
    n_valid = jnp.maximum(mmask @ mmask.T, 1.0)[..., None]
    outer = jnp.einsum('imc,jmd->ijcd', a, b).reshape(a.shape[0], b.shape[0], -1)
    return (outer @ q['output/weights'] + q['output/bias']) / n_valid

def mpwa(m, z, pair_mask, q):
    mn = layer_norm(m, {'scale': q['msa_norm/scale'], 'offset': q['msa_norm/offset']})
    zn = layer_norm(z, {'scale': q['pair_norm/scale'], 'offset': q['pair_norm/offset']})
    bias = zn @ q['bias/weights']                                  # [L,L,h]
    bias = jnp.where(pair_mask[..., None] > 0, bias, -1e5)
    attn = jax.nn.softmax(bias, axis=-2)                           # over j
    L, M = m.shape[0], m.shape[1]
    h = bias.shape[-1]; dh = q['value/weights'].shape[1] // h
    v = (mn @ q['value/weights']).reshape(L, M, h, dh)
    g = jax.nn.sigmoid(mn @ q['gate/weights']).reshape(L, M, h, dh)
    o = jnp.einsum('ijh,jmhd,imhd->imhd', attn, v, g)
    return o.reshape(L, M, h * dh) @ q['output/weights']

def msa_encoder(z, x_inputs, msa_oh, hd, dv, mmask, p, dims):
    q = sub('msa_encoder/')
    feat = jnp.concatenate([msa_oh, hd[..., None], dv[..., None]], -1)
    m = feat @ q['embed/weights'] + (x_inputs @ q['project_inputs/weights'])[:, None]
    tok = mmask[:, 0]
    pair_mask = tok[:, None] * tok[None, :]
    n = dims['n_msa']
    for i in range(n):
        blk = ({k[len('blocks/'):]: v[i] for k, v in q.items() if k.startswith('blocks/')}
               if i < n - 1 else
               {k[len('final_block/'):]: v for k, v in q.items() if k.startswith('final_block/')})
        z = z + opm(m, mmask, {k[len('outer_product_mean/'):]: v
                               for k, v in blk.items() if k.startswith('outer_product_mean/')})
        if i < n - 1:
            m = m + mpwa(m, z, pair_mask,
                         {k[len('msa_pair_weighted_averaging/'):]: v
                          for k, v in blk.items() if k.startswith('msa_pair_weighted_averaging/')})
            m = m + transition(m, {k[len('msa_transition/'):]: v
                                   for k, v in blk.items() if k.startswith('msa_transition/')})
        z = pair_block(z, blk, pair_mask)
    return z

L = r['in_x_pair.0'].shape[1]
for i in range(4):
    out = msa_encoder(jnp.asarray(r['in_x_pair.%d' % i][0]),
                      jnp.asarray(r['in_x_inputs.%d' % i][0]),
                      jnp.asarray(r['in_msa_oh.%d' % i][0]),
                      jnp.asarray(r['in_has_deletion.%d' % i][0]),
                      jnp.asarray(r['in_deletion_value.%d' % i][0]),
                      jnp.asarray(r['in_msa_attention_mask.%d' % i][0]),
                      p, dims)
    rep('msa_encoder loop%d (injected)' % i, out, r['out.%d' % i])
