import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/home/ubuntu/alphafold3')
from converters import esmfold2 as E

sd = dict(np.load('esmfold2_sd.npz'))
r  = dict(np.load('esmfold2_cond.npz'))
p  = E.map_diffusion(sd)
c  = {k[len('conditioning/'):]: v for k, v in p.items() if k.startswith('conditioning/')}

def layer_norm(x, s, o, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + eps) * s + o

def transition(x, q, i):
    h = layer_norm(x, q['input_layer_norm/scale'][i], q['input_layer_norm/offset'][i])
    h = h @ q['transition1/weights'][i]
    n = h.shape[-1] // 2
    return (jax.nn.silu(h[..., :n]) * h[..., n:]) @ q['transition2/weights'][i]

def rep(tag, ours, native):
    ours = np.asarray(ours, np.float64); native = np.asarray(np.squeeze(native), np.float64)
    e = np.abs(ours-native).max()/np.abs(native).max()
    cc = np.corrcoef(ours.ravel(), native.ravel())[0,1]
    print("%-30s relerr %.3e  corr %.8f" % (tag, e, cc))

# --- z conditioning ---------------------------------------------------------
z = jnp.concatenate([jnp.asarray(r['in_z_trunk'][0]), jnp.asarray(r['in_relative_position_encoding'][0])], -1)
z = layer_norm(z, c['z_input_norm/scale'], c['z_input_norm/offset']) @ c['z_projection/weights']
zt = {k[len('z_transitions/'):]: v for k, v in c.items() if k.startswith('z_transitions/')}
for i in range(zt['transition1/weights'].shape[0]):
    z = z + transition(z, zt, i)
rep('diffusion conditioning z', z, r['out_z'])

# --- s conditioning ---------------------------------------------------------
s = layer_norm(jnp.asarray(r['in_s_inputs'][0]), c['s_input_norm/scale'], c['s_input_norm/offset'])
s = s @ c['s_projection/weights']
t_hat = float(r['in_t_hat'][0]); sigma = float(r['in_sigma_data'][0]) if 'in_sigma_data' in r else 16.0
t_noise = 0.25 * np.log(max(t_hat / sigma, 1e-20))
n = jnp.cos(2*jnp.pi * (t_noise * c['fourier_w'] + c['fourier_b']))
n = layer_norm(n, c['noise_norm/scale'], c['noise_norm/offset']) @ c['noise_projection/weights']
s = s + n[None]
st = {k[len('s_transitions/'):]: v for k, v in c.items() if k.startswith('s_transitions/')}
for i in range(st['transition1/weights'].shape[0]):
    s = s + transition(s, st, i)
rep('diffusion conditioning s', s, r['out_s'])
print("   (t_hat=%.6g sigma_data=%.6g -> t_noise=%.6f)" % (t_hat, sigma, t_noise))
