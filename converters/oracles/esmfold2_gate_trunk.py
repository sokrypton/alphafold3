"""GATE: converted ESMFold2 pair-only trunk + parcae recurrence vs native activations.

Injects the oracle's own z_init / lm_z / msa output and reproduces
folding_trunk.out, parcae_readout.out, parcae_coda.out and distogram_head.out.
"""
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/home/ubuntu/alphafold3')
from converters import esmfold2 as E
from converters import common

sd = dict(np.load('esmfold2_sd.npz'))
ref = dict(np.load('esmfold2_dump.npz'))
dims = E.derive_dims(sd)
print("DIMS:", {k: dims[k] for k in sorted(dims)})

# ---- forward primitives (mirror the AF3 graph ops, pure jnp) ---------------
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

def run_stack(z, stacked, n, mask):
    for i in range(n):
        z = pair_block(z, {k: v[i] for k, v in stacked.items()}, mask)
    return z

# ---- convert ---------------------------------------------------------------
trunk = E.pair_only_stack(sd, 'folding_trunk', dims['n_trunk'])
coda  = E.pair_only_stack(sd, 'parcae_coda',   dims['n_coda'])
a_vec, b_T = E.parcae_dynamics(sd)

print("\nparcae a: native %s  ours %s  max|d| %.3e" % (
    ref['parcae.a'][:3], a_vec[:3], np.abs(ref['parcae.a'] - a_vec).max()))
print("parcae b: max|d| %.3e" % np.abs(ref['parcae.b'].T - b_T).max())

L = ref['folding_trunk.in0'].shape[1]
mask = jnp.ones((L, L))

def relerr(ours, native):
    ours = np.asarray(ours, np.float64); native = np.asarray(native, np.float64)
    return np.abs(ours - native).max() / max(np.abs(native).max(), 1e-9), \
           np.corrcoef(ours.ravel(), native.ravel())[0, 1]

# GATE 1: the 48-block folding trunk, injected with native's own input
z_in = jnp.asarray(ref['folding_trunk.in0'][0])
z_out = run_stack(z_in, trunk, dims['n_trunk'], mask)
e, c = relerr(z_out, ref['folding_trunk.out'][0])
print("\nGATE 1  folding_trunk (%d blocks, injected)   relerr %.3e  corr %.8f" % (dims['n_trunk'], e, c))

# GATE 2: parcae readout + coda, injected
z_r = jnp.asarray(ref['parcae_readout.in0'][0]) @ common.t(sd['parcae_readout.weight'])
e, c = relerr(z_r, ref['parcae_readout.out'][0])
print("GATE 2a parcae_readout                        relerr %.3e  corr %.8f" % (e, c))
z_c = run_stack(jnp.asarray(ref['parcae_coda.in0'][0]), coda, dims['n_coda'], mask)
e, c = relerr(z_c, ref['parcae_coda.out'][0])
print("GATE 2b parcae_coda (%d blocks, injected)      relerr %.3e  corr %.8f" % (dims['n_coda'], e, c))

# GATE 3: distogram head -- symmetrise BEFORE, bias kept
zz = jnp.asarray(ref['distogram_head.in0'][0])
logits = zz @ common.t(sd['distogram_head.weight']) + sd['distogram_head.bias']
e, c = relerr(logits, ref['distogram_head.out'][0])
print("GATE 3  distogram_head                        relerr %.3e  corr %.8f" % (e, c))

# GATE 4: parcae_input_norm
zn = layer_norm(jnp.asarray(ref['parcae_input_norm.in0'][0]),
                {'scale': sd['parcae_input_norm.weight'], 'offset': sd['parcae_input_norm.bias']})
e, c = relerr(zn, ref['parcae_input_norm.out'][0])
print("GATE 4  parcae_input_norm                     relerr %.3e  corr %.8f" % (e, c))
