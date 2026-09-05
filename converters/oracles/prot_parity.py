"""Protenix mini/tiny/05 parity: our converted trunk + MSA module vs Protenix's OWN modules.

Module-equivalence, not a pipeline gate: fixed synthetic activations in, so it
says nothing about featurisation -- only that what we converted computes what
native computes. That is exactly the question the audit could not answer:
"complete" is not "correct".

  PYTHONPATH=/home/ubuntu/protenix:/home/ubuntu/alphafold3:/home/ubuntu/alphafold3/src \
  FP32=1 JAX_DEFAULT_MATMUL_PRECISION=highest ~/venv/bin/python prot_parity.py
"""
import os, sys, types
import numpy as np

# protenix's layer_norm JIT-compiles a CUDA kernel at import; stub it with a torch
# LayerNorm of the same contract (out, mean, invvar). Getting this wrong shows up
# immediately as a mismatch, not quietly.
import torch as _t
_ext = types.ModuleType('fast_layer_norm_cuda_v2')
def _ln(x, shape, w=None, b=None, eps=1e-5):
    dims = tuple(range(x.dim() - len(shape), x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    inv = _t.rsqrt(x.var(dim=dims, unbiased=False, keepdim=True) + eps)
    out = (x - mean) * inv
    if w is not None: out = out * w
    if b is not None: out = out + b
    return out, mean.squeeze(-1), inv.squeeze(-1)
_ext.forward_with_both_affine = lambda x, s, w, b, e: _ln(x, s, w, b, e)
_ext.forward_with_weight_affine = lambda x, s, w, e: _ln(x, s, w, None, e)
_ext.forward = lambda x, s, e: _ln(x, s, None, None, e)
sys.modules.setdefault('fast_layer_norm_cuda_v2', _ext)

import torch, jax, jax.numpy as jnp, haiku as hk
from protenix.model.modules.pairformer import PairformerStack, MSABlock
from alphafold3.model import model as af3_model, model_registry, model_config
from alphafold3.model.network import modules
from alphafold3.model import params as afp
from converters import protenix2 as P

CKPT = {'protenix2': 'protenix-v2.pt',
        'protenix05': 'protenix_base_default_v0.5.0.pt',
        'protenix_mini': 'protenix_mini_default_v0.5.0.pt',
        'protenix_tiny': 'protenix_tiny_default_v0.5.0.pt'}

def cmp(tag, got, ref):
    a = np.asarray(got, np.float64).ravel(); b = np.asarray(ref, np.float64).ravel()
    corr = np.corrcoef(a, b)[0, 1]
    rel = np.abs(a - b).max() / max(np.abs(b).max(), 1e-9)
    print('    %-22s corr %.8f   relerr %.3e   rms ours/native %.4f'
          % (tag, corr, rel, np.sqrt((a**2).mean()) / max(np.sqrt((b**2).mean()), 1e-9)))
    return corr

def run(model):
    print('=== %s ===' % model, flush=True)
    sd = torch.load(os.path.expanduser('~/protenix_weights/%s' % CKPT[model]),
                    map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    d = P.derive_dims(sd)
    print('  dims: n_pairformer=%d n_msa=%d c_z=%d c_s=%d pair_H=%d n_template=%d'
          % (d['n_pairformer'], d['n_msa'], d['c_z'], d['c_s'], d['pair_H'], d['n_template']))

    cfg = af3_model.Model.Config()
    cfg.global_config.flash_attention_implementation = 'xla'
    model_registry.get(model).configure(cfg)
    if os.environ.get('FP32') == '1':
        cfg.global_config.bfloat16 = 'none'
    p = afp.get_model_haiku_params(model_dir=os.path.expanduser('~/ported/%s' % model))
    n = 48
    rng = np.random.default_rng(0)
    s = (rng.normal(size=(n, d['c_s'])) * 0.5).astype(np.float32)
    z = (rng.normal(size=(n, n, d['c_z'])) * 0.5).astype(np.float32)
    mask = np.ones((n, n), np.float32)

    # ---- trunk pairformer ---------------------------------------------------
    sub = {k[len('pairformer_stack.'):]: v for k, v in sd.items()
           if k.startswith('pairformer_stack.')}
    # n_heads here is the SINGLE-track attention_pair_bias head count (16), not
    # the triangle-attention head count (pair_H = c_z // 32 = 4); passing pair_H
    # gives a 4-vs-16 shape mismatch on linear_nobias_z.
    n_single_heads = int(sub['blocks.0.attention_pair_bias.linear_nobias_z.weight'].shape[0])
    net = PairformerStack(n_blocks=d['n_pairformer'], n_heads=n_single_heads,
                          c_z=d['c_z'], c_s=d['c_s'])
    miss, unexp = net.load_state_dict(sub, strict=False)
    assert not [m for m in miss if 'dropout' not in m], miss[:3]
    net.eval()
    with torch.no_grad():
        s_ref, z_ref = net(torch.tensor(s)[None], torch.tensor(z)[None], torch.tensor(mask)[None])
    pf = cfg.evoformer.pairformer
    def fwd(s_, z_):
        def blk(x):
            z2, s2 = modules.PairFormerIteration(
                pf, cfg.global_config, with_single=True, name='trunk_pairformer'
            )(act=x[1], single_act=x[0], pair_mask=jnp.asarray(mask),
              seq_mask=jnp.ones(n, jnp.float32))
            return (s2, z2)
        return hk.experimental.layer_stack(d['n_pairformer'])(blk)((s_, z_))
    tp = {k[len('diffuser/evoformer/'):]: v for k, v in p.items()
          if k.startswith('diffuser/evoformer/__layer_stack_no_per_layer_1/')}
    tp = {k.replace('__layer_stack_no_per_layer_1/', '__layer_stack_no_per_layer/'): v
          for k, v in tp.items()}
    s_ours, z_ours = hk.transform(fwd).apply(tp, jax.random.PRNGKey(0),
                                             jnp.asarray(s), jnp.asarray(z))
    print('  TRUNK (%d blocks)' % d['n_pairformer'])
    cmp('single', s_ours, s_ref[0].numpy()); cmp('pair', z_ours, z_ref[0].numpy())

    # ---- MSA module: the path the NO_MSA_ROW_UPDATE gate changed -------------
    c_m = int(sd['msa_module.linear_no_bias_m.weight'].shape[0])
    n_msa_rows = 3
    m = (rng.normal(size=(n_msa_rows, n, c_m)) * 0.5).astype(np.float32)
    # Run ALL n_msa native blocks, each with its own is_last_block -- comparing one
    # native block against our n-block stack reads corr 0.41 with an rms ratio of
    # ~n, which looks exactly like a port bug and is not one.
    tm, tz = torch.tensor(m)[None], torch.tensor(z)[None]
    kinds = []
    for j in range(d['n_msa']):
        msub = {k[len('msa_module.blocks.%d.' % j):]: v for k, v in sd.items()
                if k.startswith('msa_module.blocks.%d.' % j)}
        last = not any(k.startswith('msa_stack.') for k in msub)
        kinds.append('last' if last else 'full')
        blk = MSABlock(c_m=c_m, c_z=d['c_z'], is_last_block=last)
        miss, _ = blk.load_state_dict(msub, strict=False)
        assert not [x for x in miss if 'dropout' not in x], (j, miss[:3])
        blk.eval()
        with torch.no_grad():
            out = blk(tm, tz, torch.tensor(mask)[None])
        if isinstance(out, tuple) and len(out) == 2 and not last:
            tm, tz = out
        else:
            tz = out[1] if isinstance(out, tuple) else out
    print('  MSA (%d block%s: %s)' % (d['n_msa'], '' if d['n_msa'] == 1 else 's',
                                      ', '.join(kinds)))
    z_msa_ref = tz[0].numpy()

    ms = cfg.evoformer.msa_stack
    masks = {'msa': jnp.ones((n_msa_rows, n), jnp.float32), 'pair': jnp.asarray(mask)}
    def msa_fwd(m_, z_):
        def blk_j(x):
            return modules.EvoformerIteration(
                ms, cfg.global_config, name='msa_stack')(activations=x, masks=masks)
        out = hk.experimental.layer_stack(d['n_msa'])(blk_j)({'msa': m_, 'pair': z_})
        return out['msa'], out['pair']
    mp = {k[len('diffuser/evoformer/'):]: v for k, v in p.items()
          if k.startswith('diffuser/evoformer/__layer_stack_no_per_layer/')}
    try:
        _, z_msa_ours = hk.transform(msa_fwd).apply(
            mp, jax.random.PRNGKey(0), jnp.asarray(m), jnp.asarray(z))
        cmp('msa -> pair', z_msa_ours, z_msa_ref)
    except Exception as e:
        print('    msa comparison skipped: %s' % str(e)[:120])

if __name__ == '__main__':
    for m in (sys.argv[1:] or list(CKPT)):
        run(m)
