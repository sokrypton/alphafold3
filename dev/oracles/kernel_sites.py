"""Where could a fused kernel go, model by model?

Counts every call a fused arm sees, at the three kinds of site:

  served     the pallas arm took it
  refused    the arm was offered it and said no (shape outside the kernel)
  bypass     the site never reaches components/attention.py at all -- the
             diffusion transformer and the MSA stack carry their own
             einsum+softmax, so no dispatcher choice reaches them

Run:  FLASH=pallas PYTHONPATH=src:. python dev/oracles/kernel_sites.py <model>
"""
import collections
import runpy
import sys

HITS = collections.Counter()


def install(model):
  from alphafold3.model.components import attention as af3_attn
  from alphafold3.model.components import pallas_attn
  from alphafold3.model.network import diffusion_transformer as dt

  real_attn = pallas_attn.attention
  real_gdp = pallas_attn.gated_dual_proj
  real_dispatch = af3_attn.dot_product_attention
  # NOT `dt.attention` -- there is no such name, and patching it silently
  # counted nothing and reported "no bypass sites". They are called
  # self_attention (the token transformer) and cross_attention (the atom
  # encoder/decoder), and neither goes through components/attention.py.
  real_self = dt.self_attention
  real_cross = dt.cross_attention

  def attn(q, k, v, *, mask, bias, scale):
    out = real_attn(q, k, v, mask=mask, bias=bias, scale=scale)
    key = ('attention', 'served' if out is not None else 'refused',
           tuple(q.shape), None if bias is None else tuple(bias.shape))
    HITS[key] += 1
    return out

  def gdp(x, w_proj, w_gate, mask):
    out = real_gdp(x, w_proj, w_gate, mask)
    HITS[('glu', 'served' if out is not None else 'refused',
          tuple(x.shape), tuple(w_proj.shape))] += 1
    return out

  def dispatch(q, k, v, **kw):
    HITS[('dispatcher', 'reached', tuple(q.shape),
          None if kw.get('bias') is None else tuple(kw['bias'].shape))] += 1
    return real_dispatch(q, k, v, **kw)

  pallas_attn.attention, pallas_attn.gated_dual_proj = attn, gdp
  af3_attn.dot_product_attention = dispatch

  # The diffusion transformer's own attentions: the bypassing sites.
  # Both are called with keyword arguments at some sites, so the wrappers must
  # not name any parameter of their own.
  def _first_shape(a, kw):
    for v in list(a) + [kw.get('x'), kw.get('act')]:
      if hasattr(v, 'shape'):
        return tuple(v.shape)
    return ()

  def self_attn(*a, **kw):
    pl = kw.get('pair_logits')
    HITS[('dit self_attention', 'bypass', _first_shape(a, kw),
          tuple(pl.shape) if hasattr(pl, 'shape') else None)] += 1
    return real_self(*a, **kw)

  def cross_attn(*a, **kw):
    pl = kw.get('pair_logits')
    HITS[('dit cross_attention', 'bypass', _first_shape(a, kw),
          tuple(pl.shape) if hasattr(pl, 'shape') else None)] += 1
    return real_cross(*a, **kw)

  dt.self_attention, dt.cross_attention = self_attn, cross_attn


if __name__ == '__main__':
  model = sys.argv[1]
  sys.argv = ['fold_check.py', model, '/home/ubuntu/6MRR.pdb']
  install(model)
  try:
    runpy.run_path('dev/oracles/fold_check.py', run_name='__main__')
  except SystemExit:
    pass
  print(f'\n=== {model}')
  for (kind, status, a, b), n in sorted(HITS.items(), key=lambda kv: (kv[0][0], -kv[1])):
    print(f'  {n:6d}x  {kind:22s} {status:8s} {a} {b if b else ""}')
