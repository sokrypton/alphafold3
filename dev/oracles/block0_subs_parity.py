"""Our block-0 pair sub-modules, each on NATIVE's own input, vs native's delta.

Localises the openbind0 trunk fault to a sub-module. Each of our modules is
built alone with the blob's BLOCK-0 slice of the layer_stack parameters and fed
the tensor native's corresponding module was fed, so no module inherits an
earlier one's error.
"""
import os, sys
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp

sys.path.insert(0, 'dev/oracles')
from alphafold3.model import model as af3_model, model_registry
from alphafold3.model import params as afp
from alphafold3.model import model_config
from alphafold3.model.network import modules

# NO_TCPB=1 drops this model from TRANSPOSED_COLUMN_PAIR_BIAS in-process, so the
# convention can be tested without editing src/ while the matrix runs.
if os.environ.get('NO_TCPB'):
  model_config.TRANSPOSED_COLUMN_PAIR_BIAS = tuple(
      m for m in model_config.TRANSPOSED_COLUMN_PAIR_BIAS
      if m != sys.argv[1])
  print('  TRANSPOSED_COLUMN_PAIR_BIAS: %s removed' % sys.argv[1])

model, npz = sys.argv[1], sys.argv[2]
d = np.load(npz)
tok = None

cfg = af3_model.Model.Config()
cfg.global_config.bfloat16 = 'none'
cfg.global_config.flash_attention_implementation = 'xla'
model_registry.get(model).configure(cfg)
pf = cfg.evoformer.pairformer
full = afp.get_model_haiku_params(
    model_dir=os.path.expanduser('~/ported/%s' % model))
PRE = [k for k in full if 'trunk_pairformer/' in k]
root = PRE[0][:PRE[0].index('trunk_pairformer/')]
print('%s: trunk scope %r, %d pairformer scopes' % (model, root, len(PRE)))


def block0(name):
  """The blob's block-0 slice for one sub-module scope."""
  out = {}
  want = root + 'trunk_pairformer/' + name
  for k, v in full.items():
    if k == want or k.startswith(want + '/'):
      tail = k[len(root + 'trunk_pairformer/'):]
      out[tail] = {kk: np.asarray(vv)[0] for kk, vv in v.items()}
  return out


def run(name, builder, x, mask):
  params = block0(name)
  if not params:
    print('  %-34s NO PARAMS in the blob' % name)
    return None

  def fwd(x_, m_):
    return builder()(x_, m_)

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0), jnp.asarray(x), jnp.asarray(mask))
  ps = {}
  for sc in init:
    # A standalone module's haiku scope IS the sub-path the blob stores under
    # ('triangle_multiplication_outgoing/left_norm_input'), so it is used as the
    # key directly -- taking the last segment instead is what made every scope
    # read as unmapped on the first run.
    src = params.get(sc)
    if src is None:
      print('  %-34s unmapped scope %s (have %s)' % (name, sc, list(params)[:3]))
      return None
    ps[sc] = {k: src[k] for k in init[sc]}
  return np.asarray(f.apply(ps, jax.random.PRNGKey(0),
                            jnp.asarray(x), jnp.asarray(mask)))


def cmp(tag, ours, nat):
  a = np.asarray(ours, np.float64).ravel(); b = np.asarray(nat, np.float64).ravel()
  rb = np.sqrt((b ** 2).mean())
  print('  %-22s corr %.9f  rms ours/nat %.5f  max|d| %.5f  rms(nat) %.4f'
        % (tag, np.corrcoef(a, b)[0, 1], np.sqrt((a ** 2).mean()) / max(rb, 1e-12),
           np.abs(a - b).max(), rb))


tokm = np.load(sys.argv[3])['batch_token_mask'][0].astype(np.float32)
mask = tokm[:, None] * tokm[None, :]
gc = cfg.global_config
jobs = [
    ('triangle_multiplication_outgoing', 'tri_mul_out',
     lambda: modules.TriangleMultiplication(
         pf.triangle_multiplication_outgoing, gc,
         name='triangle_multiplication_outgoing')),
    ('triangle_multiplication_incoming', 'tri_mul_in',
     lambda: modules.TriangleMultiplication(
         pf.triangle_multiplication_incoming, gc,
         name='triangle_multiplication_incoming')),
    ('pair_attention1', 'tri_att_start',
     lambda: modules.GridSelfAttention(pf.pair_attention, gc,
                                       transpose=False, name='pair_attention1')),
    ('pair_attention2', 'tri_att_end',
     lambda: modules.GridSelfAttention(pf.pair_attention, gc,
                                       transpose=True, name='pair_attention2')),
]
for our_name, nat_name, builder in jobs:
  x = d[nat_name + '_in'][0]
  nat = d[nat_name + '_out'][0]
  # of3 runs tri_att_end on a TRANSPOSED z (`z.transpose(-2, -3)` around the
  # call, mask transposed, transpose_bias=True), so its captured input and
  # output live in the transposed space. Ours does the transposing internally
  # (GridSelfAttention(transpose=True)), so both have to be swapped back --
  # comparing them as captured reads corr 0.43 for BOTH models, which is the
  # harness, not the port.
  if nat_name == 'tri_att_end':
    x = np.swapaxes(x, 0, 1)
    nat = np.swapaxes(nat, 0, 1)
  got = run(our_name, builder, x, mask)
  if got is not None:
    cmp(our_name, got, nat)

# the transition takes no mask
params = block0('pair_transition')
if params:
  def fwd(x_):
    return modules.TransitionBlock(pf.pair_transition, gc,
                                   name='pair_transition')(x_)
  f = hk.transform(fwd)
  x = d['pair_transition_in'][0]
  init = f.init(jax.random.PRNGKey(0), jnp.asarray(x))
  ps = {}
  for sc in init:
    ps[sc] = {k: params[sc][k] for k in init[sc]}
  got = np.asarray(f.apply(ps, jax.random.PRNGKey(0), jnp.asarray(x)))
  cmp('pair_transition', got, d['pair_transition_out'][0])
