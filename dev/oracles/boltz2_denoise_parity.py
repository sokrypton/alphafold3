"""L3 for boltz2 -- the last model missing a whole level -- by INJECTION.

Every other L3 runs the vendor's DiffusionModule live. boltz2 cannot: its atom
path wants boltz's own feature layout (flat atoms + `atom_to_token`, its own
`ref_*` names), which is why the port deferred this gate and why the coverage
table carried a `.` for boltz2 at L3 while every other port had a number.

So this one uses the tensors captured from a real boltz run
(`~/boltz2_6mrr/diff_dump.npz`, `score.in.*` -> `score.out.r_update`) and feeds
OUR diffusion head the same inputs. The EDM algebra is boltz's own, and it is
identical to AF3's (sigma_data 16, same c_skip/c_out/c_in), which is what makes
the two comparable at all:

    times    = c_noise(sigma) = log(sigma / 16) * 0.25   ->  sigma = 16 e^(4t)
    r_noisy  = c_in(sigma) * noised                      ->  noised = r_noisy * sqrt(sigma^2 + 256)
    denoised = c_skip(sigma) * noised + c_out(sigma) * r_update

`score.out.r_update` is the RAW network output, pre-EDM, where our head returns
the denoised coordinates -- so the reference is reconstructed with the line
above rather than compared directly. Comparing r_update against x_denoised
would be a units mismatch that still correlates.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:. \
    python dev/oracles/boltz2_denoise_parity.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atom_parity import flat_atom_features                 # noqa: E402
from confidence_parity import _cmp                          # noqa: E402

SIGMA_DATA = 16.0


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('--dump', default=os.path.expanduser(
      '~/boltz2_6mrr/diff_dump.npz'))
  ap.add_argument('--pdb', default=os.path.expanduser('~/6MRR.pdb'))
  ap.add_argument('--model_dir', default=None)
  args = ap.parse_args(argv)
  sys.argv = sys.argv[:1]

  if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
    raise SystemExit('set JAX_DEFAULT_MATMUL_PRECISION=highest')
  d = np.load(args.dump)
  for k in ('score.in.r_noisy', 'score.in.times', 'score.in.s_trunk',
            'score.in.s_inputs', 'score.out.r_update', 'dc.in.z_trunk'):
    if k not in d.files:
      raise SystemExit('dump lacks %r; it has %s' % (k, sorted(d.files)[:6]))

  import fold_check
  from alphafold3.model import feat_batch

  seq, _ = fold_check.parse_ca(args.pdb)
  batch, cfg, model_dir = fold_check._fold_setup('boltz2', seq, args.model_dir)
  fb = feat_batch.Batch.from_data_dict(batch)
  feats = flat_atom_features(fb)
  mask = feats['mask']
  n_atom_ours = int(mask.sum())
  r_noisy = d['score.in.r_noisy'][0]
  print('boltz2 L3 by injection: %d atoms ours, %d in the dump'
        % (n_atom_ours, r_noisy.shape[0]))
  # ALIGN THE TWO ATOM LISTS, per token, and do not assume they match.
  #
  # For 6MRR ours has 574 real atoms and boltz 573, and the difference is a
  # real featurisation divergence rather than padding: **we give the C-terminal
  # residue an OXT and boltz does not** (token 67, GLU: ours 10 atoms, boltz 9).
  # boltz's flat axis is 576 = 573 real + 3 padding rows, whose all-zero
  # `atom_to_token` rows make `argmax` attribute them to token 0 -- which is why
  # a naive per-token count reads 4 vs 7 there and means nothing.
  #
  # So: take the first min(ours, boltz) atoms of each token on both sides. Atom
  # order within a residue is CCD order on both, and OXT sorts last in GLU, so
  # this drops exactly the atom boltz never sees. Anything else that failed to
  # line up would show as a large residual rather than being hidden.
  a2t = np.asarray(d['feat.atom_to_token'][0])
  pad = np.asarray(d['feat.atom_pad_mask'][0]) > 0
  b_tok = a2t.argmax(-1)
  n_tok = mask.shape[0]
  o_counts = mask.sum(1).astype(int)
  b_counts = np.array([int(((b_tok == i) & pad).sum()) for i in range(n_tok)])
  extra = [(i, int(o_counts[i]), int(b_counts[i])) for i in range(n_tok)
           if o_counts[i] != b_counts[i]]
  print('  boltz real atoms %d (of %d rows); per-token count differs at %s'
        % (int(pad.sum()), len(pad), extra or 'nothing'))
  o_idx, b_idx = [], []
  o_off = np.concatenate([[0], np.cumsum(o_counts)])
  b_flat = np.where(pad)[0]
  b_off = np.concatenate([[0], np.cumsum(b_counts)])
  for i in range(n_tok):
    k = min(o_counts[i], b_counts[i])
    o_idx.extend(range(o_off[i], o_off[i] + k))
    b_idx.extend(b_flat[b_off[i]:b_off[i] + k])
  o_idx, b_idx = np.array(o_idx), np.array(b_idx)
  print('  comparing %d atoms (ours %d, boltz %d)'
        % (len(o_idx), n_atom_ours, int(pad.sum())))

  times = float(np.asarray(d['score.in.times']).reshape(-1)[0])
  sigma = SIGMA_DATA * np.exp(4.0 * times)
  c_in = 1.0 / np.sqrt(sigma ** 2 + SIGMA_DATA ** 2)
  c_skip = SIGMA_DATA ** 2 / (sigma ** 2 + SIGMA_DATA ** 2)
  c_out = sigma * SIGMA_DATA / np.sqrt(SIGMA_DATA ** 2 + sigma ** 2)
  noised_all = r_noisy / c_in
  ref_all = c_skip * noised_all + c_out * d['score.out.r_update'][0]
  noised = noised_all[b_idx]
  ref = ref_all[b_idx]
  print('  times %.6f -> sigma %.4f  (c_in %.6f, c_skip %.6f, c_out %.6f)'
        % (times, sigma, c_in, c_skip, c_out))

  # Our dense (token, slot) array, filled at the atoms that correspond.
  pos_flat = np.zeros((n_atom_ours, 3), np.float32)
  pos_flat[o_idx] = noised
  pos_dense = np.zeros(mask.shape + (3,), np.float32)
  pos_dense[mask] = pos_flat
  s = np.asarray(d['score.in.s_trunk'][0], np.float32)
  z = np.asarray(d['dc.in.z_trunk'][0], np.float32)
  s_inputs = np.asarray(d['score.in.s_inputs'][0], np.float32)
  print('  injected: s_trunk %s, z_trunk %s, s_inputs %s'
        % (s.shape, z.shape, s_inputs.shape))

  got = ours(cfg, model_dir, fb, pos_dense, sigma, s_inputs, s, z)
  got_flat = np.asarray(got)[mask][o_idx]
  _cmp('x_denoised', got_flat, ref)
  dist = np.sqrt(((got_flat - ref) ** 2).sum(-1))
  print('  per-atom distance: mean %.4f A, max %.4f A, rms(native) %.2f'
        % (dist.mean(), dist.max(), np.sqrt((ref ** 2).sum(-1).mean())))
  return 0


def ours(cfg, model_dir, fb, pos_dense, noise, s_inputs, s, z):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import diffusion_head

  cfg.global_config.bfloat16 = 'none'
  full = afp.get_model_haiku_params(model_dir=model_dir)
  emb = {'single': jnp.asarray(s), 'pair': jnp.asarray(z),
         'target_feat': jnp.asarray(s_inputs)}

  def fwd():
    return diffusion_head.DiffusionHead(
        cfg.heads.diffusion, cfg.global_config)(
            positions_noisy=jnp.asarray(pos_dense),
            noise_level=jnp.asarray(noise, jnp.float32),
            batch=fb, embeddings=emb, use_conditioning=True)

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      tail = sc[len('diffusion_head/'):] if sc.startswith('diffusion_head/') \
          else sc
      key = ('diffuser/~/diffusion_head' if tail in ('~', 'diffusion_head')
             else 'diffuser/~/diffusion_head/' + tail)
      src = full.get(key)
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = np.asarray(v, np.float32)
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:3]))
  assert not unmapped, 'our head is partly at init'
  return f.apply(params, jax.random.PRNGKey(0))


if __name__ == '__main__':
  raise SystemExit(main())
