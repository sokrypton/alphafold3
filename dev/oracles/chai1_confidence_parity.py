"""L4 for chai-1, by INJECTION from chai's own captured ConfidenceHead I/O.

chai ships TorchScript and its confidence head IS callable -- `forward` is
undefined, but the BUCKETED entry points `forward_256` .. `forward_1024` are
all there. It was captured verbatim during the port -- nine input tensors and
three output LOGIT tensors -- so the DEFAULT path here needs no torch, no
trunk, no diffusion and no featurisation agreement: feed our converted head
chai's inputs and compare the logits it produces.

FLOOR=1 re-runs the archive instead of reading the capture, which is the only
way to know what this cell can resolve. Do it before reading the two numbers
below as a port difference -- the head runs in BFLOAT16.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:. \
    python dev/oracles/chai1_confidence_parity.py

LOGITS, not the derived pLDDT/PAE, so that no assumption about chai's bin
centres enters the gate -- the trap recorded in memory
`confidence-heads-status` (a head that emits logits under a score key reads
plausible and is wrong).

Only the PAE and PDE heads are compared. They are per TOKEN PAIR, so the
comparison needs no atom-layout agreement at all; pLDDT is per ATOM and would,
which is a separate problem. What the head still needs positionally is the
representative atom per token, and the capture names it
(`token_reference_atom_index`), so those coordinates are placed exactly rather
than assumed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confidence_parity import _cmp                          # noqa: E402

CAP = os.path.expanduser('~/chai_6mrr/conf_seam.npz')


def _chai_floor(z, ref_pae, ref_pde, L):
  """NATIVE against ITSELF: reproduce the capture, then shift the inputs.

  Two numbers, and the first is the surprising one:

    * REPRODUCTION -- the same archive, the same captured inputs, run here on
      the CPU against outputs captured on a GPU. That is not zero, because the
      head runs in BFLOAT16: the TorchScript does `torch.to(x, 15)` on the
      trunk representations and casts every Linear weight to bf16. Coordinates
      stay float32 -- proved by the archive itself, since `torch.cdist` has no
      bf16 kernel and the forward calls it.
    * a half-ulp input shift (2^-9 relative), which is the same order.

  Both bound what this cell can resolve. `EXACT_CDIST` was also tried here and
  changes NOTHING (0.000e+00): chai bins its distances with `searchsorted` and
  has no unbinned distance term, so the float32 cdist error that accounts for
  the whole protenix residual cannot reach these logits.
  """
  import torch

  path = os.path.expanduser('~/chai1_weights/models_v2/confidence_head.pt')
  if not os.path.exists(path):
    print('  FLOOR UNAVAILABLE: no %s' % path)
    return
  m = torch.jit.load(path, map_location='cpu')
  m.eval()
  fwd = m._c._get_method('forward_256')
  order = ['token_single_input_repr', 'token_single_trunk_repr',
           'token_pair_trunk_repr', 'token_single_mask', 'atom_single_mask',
           'atom_coords', 'token_reference_atom_index', 'atom_token_index',
           'atom_within_token_index']
  base = {k: np.asarray(z['in|' + k]) for k in order}

  def run(f):
    args = []
    for k in order:
      v = f[k]
      if k in ('token_single_mask', 'atom_single_mask'):
        args.append(torch.from_numpy(v).bool())
      elif k.endswith('_index'):
        args.append(torch.from_numpy(v).long())
      elif k == 'atom_coords':
        args.append(torch.from_numpy(v.astype(np.float32)))
      else:
        args.append(torch.from_numpy(v.astype(np.float32)).to(torch.bfloat16))
    with torch.no_grad():
      return [np.asarray(o.float()) for o in fwd(*args)]

  out = run(base)
  print('  FLOOR, native re-run (bf16, as the archive declares):')
  _cmp('  repro_pae', out[0][0, :L, :L], ref_pae)
  _cmp('  repro_pde', out[1][0, :L, :L], ref_pde)
  rng = np.random.default_rng(1234)
  eps = float(os.environ.get('EPS', 2 ** -9))
  pert = dict(base)
  for k in ('token_single_input_repr', 'token_single_trunk_repr',
            'token_pair_trunk_repr'):
    v = base[k].astype(np.float32)
    pert[k] = (v * (1 + eps * rng.normal(size=v.shape))).astype(np.float32)
  outp = run(pert)
  print('  FLOOR, native after a %.3g relative input shift:' % eps)
  _cmp('  floor_pae', outp[0][0, :L, :L], out[0][0, :L, :L])
  _cmp('  floor_pde', outp[1][0, :L, :L], out[1][0, :L, :L])


def main(argv=None):
  if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
    raise SystemExit('set JAX_DEFAULT_MATMUL_PRECISION=highest')
  if not os.path.exists(CAP):
    raise SystemExit('no capture at %s' % CAP)
  sys.argv = sys.argv[:1]

  import haiku as hk
  import jax
  import jax.numpy as jnp

  import fold_check
  from alphafold3.model import feat_batch, params as afp
  from alphafold3.model.components import haiku_modules as hm
  from alphafold3.model.network import confidence_head as ch

  z = np.load(CAP)
  L = int(np.asarray(z['in|token_single_mask']).sum())
  seq, _ = fold_check.parse_ca(os.path.expanduser('~/6MRR.pdb'))
  batch, cfg, model_dir = fold_check._fold_setup('chai1', seq)
  cfg.global_config.bfloat16 = os.environ.get('BF16', 'none')
  fb = feat_batch.Batch.from_data_dict(batch)
  n_tok = np.asarray(fb.token_features.mask).shape[0]
  if n_tok != L:
    raise SystemExit('capture has %d real tokens, our batch %d' % (L, n_tok))
  print('chai1 confidence head by injection: %d tokens' % L)

  # chai's REPRESENTATIVE atom per token -> our dense (token, slot) layout, at
  # the slot our own pseudo-beta gather reads. Everything else stays zero: the
  # PAE/PDE heads see positions only through that gather.
  rep = np.asarray(z['in|token_reference_atom_index'])[0].astype(int)[:L]
  coords = np.asarray(z['in|atom_coords'])[0]
  pb = fb.pseudo_beta_info.token_atoms_to_pseudo_beta
  idxs = np.asarray(pb.gather_idxs).reshape(-1)[:L]
  dense = np.zeros(np.asarray(fb.ref_structure.mask).shape + (3,), np.float32)
  dense.reshape(-1, 3)[idxs] = coords[rep]

  emb = {'pair': jnp.asarray(np.asarray(z['in|token_pair_trunk_repr'])[0, :L, :L]),
         'single': jnp.asarray(np.asarray(z['in|token_single_trunk_repr'])[0, :L]),
         'target_feat': jnp.asarray(
             np.asarray(z['in|token_single_input_repr'])[0, :L])}

  # Tap the logit projections by module name: the head returns derived
  # pLDDT/PAE and comparing those would fold our bin centres into the gate.
  taps = {}
  want = ('pae_logits', 'left_half_distance_logits')
  orig = hm.Linear.__call__

  def tapped(self, x, *a, **kw):
    out = orig(self, x, *a, **kw)
    if self.name in want:
      taps[self.name] = out
    return out

  hm.Linear.__call__ = tapped
  try:
    def fwd():
      return ch.ConfidenceHead(cfg.heads.confidence, cfg.global_config)(
          dense_atom_positions=jnp.asarray(dense),
          embeddings=emb,
          seq_mask=fb.token_features.mask,
          token_atoms_to_pseudo_beta=pb,
          asym_id=fb.token_features.asym_id,
          token_features=fb.token_features,
          # REQUIRED for chai1: `confidence_head.py:521` keys its 37-slot pLDDT
          # gather on atom_name_chars being present, and without it the generic
          # path builds a 24-slot projection the blob cannot fill
          # ('plddt_logits/weights' (384, 37, 50) against (384, 24, 50)).
          atom_name_chars=fb.ref_structure.atom_name_chars)

    f = hk.transform(fwd)
    init = f.init(jax.random.PRNGKey(0))
    full = afp.get_model_haiku_params(model_dir=model_dir)
    p, unmapped = {}, []
    for sc in init:
      p[sc] = {}
      for k in init[sc]:
        src = None
        for cand in ('diffuser/' + sc, sc):
          if cand in full:
            src = full[cand]
            break
        v = None if src is None else src.get(k)
        if v is None:
          unmapped.append('%s/%s' % (sc, k))
          p[sc][k] = init[sc][k]
        else:
          p[sc][k] = np.asarray(v, np.float32)
    print('  ours: %d scopes, %d unmapped %s'
          % (len(init), len(unmapped), unmapped[:3]))
    assert not unmapped, 'our confidence head is partly at init'
    f.apply(p, jax.random.PRNGKey(0))
  finally:
    hm.Linear.__call__ = orig

  ref_pae = np.asarray(z['out|0'])[0, :L, :L]
  ref_pde = np.asarray(z['out|1'])[0, :L, :L]
  got_pae = np.asarray(taps['pae_logits'])
  # AF3 emits HALF the distance logits and symmetrises; chai's out|1 is the
  # full matrix, so compare the half we produce against the same half.
  # AF3 emits the LEFT HALF and symmetrises: `pde = left + swapaxes(left)`,
  # where chai's out|1 is `pde_projection(LN(z) + LN(z)^T)` -- the same function
  # (a Linear commutes with the token-axis transpose). So the comparison has to
  # symmetrise ours, and the tell that it does not is a magnitude ratio of
  # exactly one half: unsymmetrised this read rms ours/native 0.5019.
  got_pde = np.asarray(taps['left_half_distance_logits'])
  got_pde = got_pde + np.swapaxes(got_pde, 0, 1)
  print('  shapes: pae ours %s native %s | pde ours %s native %s'
        % (got_pae.shape, ref_pae.shape, got_pde.shape, ref_pde.shape))
  # FLOOR=1 RE-RUNS NATIVE. This file's docstring used to say chai's modules
  # have "no callable forward", and `forward` really is undefined -- but the
  # BUCKETED entry points are not: `forward_256` .. `forward_1024` are all
  # present on confidence_head.pt. So the cell does have a floor after all, and
  # without one its two rows were being read as a port difference with nothing
  # to compare them to.
  #
  # Needs torch, which the default path deliberately does not, so it is opt-in.
  if os.environ.get('FLOOR'):
    _chai_floor(z, ref_pae, ref_pde, L)

  _cmp('pae_logits', got_pae, ref_pae)
  if got_pde.shape == ref_pde.shape:
    _cmp('pde_logits', got_pde, ref_pde)
  else:
    print('  pde: shapes differ (%s vs %s) -- AF3 emits the left half and '
          'symmetrises; not compared' % (got_pde.shape, ref_pde.shape))
  return 0


if __name__ == '__main__':
  sys.exit(main())
