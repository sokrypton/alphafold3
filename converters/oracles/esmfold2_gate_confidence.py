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


sd = dict(np.load('esmfold2_sd.npz'))
r  = dict(np.load('esmfold2_conf.npz'))
f  = dict(np.load('esmfold2_dump.npz'))
dims = E.derive_dims(sd)
c = E.confidence_head(sd, dims)

def rep(tag, ours, native, warn=3e-2):
    ours=np.asarray(ours,np.float64); native=np.asarray(np.squeeze(native),np.float64)
    e=np.abs(ours-native).max()/max(np.abs(native).max(),1e-9)
    cc=np.corrcoef(ours.ravel(),native.ravel())[0,1]
    print("%-32s relerr %.3e  corr %.8f %s"%(tag,e,cc,'' if e<warn else ' <-- CHECK')); return cc

s_in = jnp.asarray(r['in_s_inputs'][0]); L = s_in.shape[0]
sn = layer_norm(s_in, {'scale': c['s_inputs_norm/scale'], 'offset': c['s_inputs_norm/offset']})
z  = layer_norm(jnp.asarray(r['in_z'][0]), {'scale': c['z_norm/scale'], 'offset': c['z_norm/offset']})
z  = z + jnp.asarray(r['in_relative_position_encoding'][0]) + jnp.asarray(r['in_token_bonds_encoding'][0])
z  = z + (sn @ c['s_to_z/weights'])[:, None] + (sn @ c['s_to_z_transpose/weights'])[None, :]
z  = z + ((sn @ c['s_to_z_prod_in1/weights'])[:, None] *
          (sn @ c['s_to_z_prod_in2/weights'])[None, :]) @ c['s_to_z_prod_out/weights']

rep_idx = r['in_distogram_atom_idx'][0].astype(int)
xp = r['in_x_pred'][0][rep_idx]
dist = np.linalg.norm(xp[:, None] - xp[None, :], axis=-1)
bins = (dist[..., None] > c['boundaries']).sum(-1)
z = z + jnp.asarray(c['dist_bin_embed/weights'])[bins]
rep('confidence pair (pre-trunk)', z, r['inner_trunk_in'])

mask = jnp.asarray(r['in_token_attention_mask'][0]); pm = mask[:, None] * mask[None, :]
delta = run_stack(z, c, 'folding_trunk/', dims['n_conf'], pm)
rep('confidence trunk (4 blk)', delta, r['inner_trunk_out'])
pair = z + delta                                    # <-- stack-level residual

scores = (pair @ c['row_pool_attn/weights'])[..., 0]
scores = jnp.where(mask[None, :] > 0, scores, -1e9)
single = jnp.einsum('nm,nmd->nd', jax.nn.softmax(scores, -1), pair) @ c['row_pool_out/weights']

rep('pae_logits', layer_norm(pair, {'scale': c['pae_norm/scale'], 'offset': c['pae_norm/offset']}) @ c['pae/weights'], r['out_pae_logits'])
rep('pde_logits', layer_norm(pair, {'scale': c['pde_norm/scale'], 'offset': c['pde_norm/offset']}) @ c['pde/weights'], r['out_pde_logits'])

a2t = f['feat.atom_to_token'][0].astype(int) * f['feat.atom_attention_mask'][0].astype(int)
s_at = single[jnp.asarray(a2t)]
# intra-token atom index: running count that resets at each token boundary
intra = np.zeros(len(a2t), int)
run = 0
for i in range(1, len(a2t)):
    run = run + 1 if a2t[i] == a2t[i-1] else 0
    intra[i] = run
intra = np.clip(intra, 0, c['plddt_weight'].shape[0]-1)
pl = layer_norm(s_at, {'scale': c['plddt_norm/scale'], 'offset': c['plddt_norm/offset']})
rep('plddt_logits', jnp.einsum('ac,acb->ab', pl, jnp.asarray(c['plddt_weight'])[intra]), r['out_plddt_logits'])
rs = layer_norm(s_at, {'scale': c['resolved_norm/scale'], 'offset': c['resolved_norm/offset']})
rep('resolved_logits', jnp.einsum('ac,acb->ab', rs, jnp.asarray(c['resolved_weight'])[intra]), r['out_resolved_logits'])
