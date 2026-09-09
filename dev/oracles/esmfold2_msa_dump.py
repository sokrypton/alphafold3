"""NATIVE side of L1b for ESMFold2: its own `MSAEncoder`, dumped to an npz.

`msa_parity.py` runs its other six adapters in-process, because torch and the
vendor package are both importable from the JAX venv. ESMFold2 is the exception:
its implementation ships inside `transformers` (`models/esmfold2`), which lives
only in `~/venv_esm`, and installing it beside JAX is not an option here. So the
native side runs there and writes the comparison inputs AND its output to an
npz that `msa_parity.py` reads with numpy alone.

  ~/venv_esm/bin/python dev/oracles/esmfold2_msa_dump.py esmfold2_exp
  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:. \
    ~/venv/bin/python dev/oracles/msa_parity.py esmfold2_exp

The npz carries the EMBEDDED msa (embed(m_feat) + project_inputs(x_inputs)),
not the raw rows, for the same reason every other adapter here returns it: both
sides then enter the stack on an identical activation, and a random activation
on our side reads corr 0.468 and means nothing.

Only three of the eight ESMFold2 releases have an MSA encoder at all --
`msa=4` in `model_registry.ESMFOLD2_VARIANTS` (esmfold2, esmfold2_exp,
esmfold2_exp_cutoff2025). The five `*_fast`/`lm*` rows set `msa_encoder.enabled`
false and carry no such weights, so they are n/a rather than ungated.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

# ESMFold2 ships TWO MSAEncoders, one per line, and they are NOT the same
# module: the released block runs the outer product FIRST and skips the msa
# update in its last block, while the experimental block updates the msa first
# and runs every block. Importing the released one for all three variants is
# what made this gate read 1.000000 while `model_config.MSA_UPDATE_BEFORE_OPM`
# had the experimental releases in the wrong branch -- it was comparing us
# against the wrong native module and agreeing with it. `_encoder_class` picks
# by variant.
from transformers.models.esmfold2 import modeling_esmfold2 as _released
from transformers.models.esmfold2 import modeling_esmfold2_experimental as _exp

_HUB = {'esmfold2': 'ESMFold2',
        'esmfold2_exp': 'ESMFold2-Experimental',
        'esmfold2_exp_cutoff2025': 'ESMFold2-Experimental-Cutoff2025'}

# Which module the release actually runs. Keyed on the same split
# model_config.ESMFOLD2_EXPERIMENTAL uses.
_EXPERIMENTAL = ('esmfold2_exp', 'esmfold2_exp_cutoff2025')


def _encoder_class(model):
  mod = _exp if model in _EXPERIMENTAL else _released
  return mod.MSAEncoder, mod.MSAEncoderBlock, mod is _exp


def _state_dict(model):
  """The variant's flat state dict, from the local snapshot if there is one."""
  from safetensors.torch import load_file
  d = os.path.expanduser('~/esmfold2_variants/%s' % _HUB[model])
  if not os.path.isdir(d):
    from huggingface_hub import snapshot_download
    d = snapshot_download('biohub/%s' % _HUB[model])
  shards = sorted(glob.glob(os.path.join(d, '*.safetensors')))
  if not shards:
    raise SystemExit('no safetensors under %s' % d)
  sd = {}
  for s in shards:
    sd.update(load_file(s))
  # ESM-C's tower is in the same file; only the folding trunk is wanted.
  return {k: v for k, v in sd.items() if k.startswith('msa_encoder.')}


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model', choices=sorted(_HUB))
  ap.add_argument('--n_tok', type=int, default=68)     # 6MRR, as msa_parity
  ap.add_argument('--n_msa', type=int, default=8)
  ap.add_argument('--nonuniform', action='store_true')
  ap.add_argument('--blocks', type=int, default=None,
                  help='truncate the native stack to the FIRST n blocks, to '
                       'localise a residual by depth. msa_parity.py reads the '
                       'matching npz when MSA_BLOCKS is set, and truncates our '
                       'stacked leaves the same way -- truncating one side only '
                       'compares a 1-block run against a 4-block one and reads '
                       'as a catastrophic gap.')
  ap.add_argument('--keep_final_update', action='store_true',
                  help="run the last block's msa update, which native drops")
  ap.add_argument('--out', default=None)
  args = ap.parse_args(argv)

  sd = _state_dict(args.model)
  if not sd:
    raise SystemExit('%s has no msa_encoder weights (msa=0 variant)' % args.model)
  n_layers = 1 + max(int(k.split('.')[2]) for k in sd if k.startswith('msa_encoder.blocks.'))
  d_msa, msa_in = sd['msa_encoder.embed.weight'].shape
  d_inputs = sd['msa_encoder.project_inputs.weight'].shape[1]
  d_pair = sd['msa_encoder.blocks.0.outer_product_mean.Wout.weight'].shape[0]
  d_hidden = sd['msa_encoder.blocks.0.outer_product_mean.W.weight'].shape[0] // 2
  # Wv is (n_heads * head_width, d_msa); the released line is 16 wide and the
  # experimental line 32, so it comes off the tensor rather than from a default.
  wv = sd.get('msa_encoder.blocks.0.msa_pair_weighted_averaging.Wv.weight')
  n_heads = sd['msa_encoder.blocks.0.msa_pair_weighted_averaging.compute_bias.1.weight'].shape[0]
  head_w = wv.shape[0] // n_heads
  print('  checkpoint: %d blocks, d_msa %d, msa_in %d, d_inputs %d, d_pair %d, '
        'opm hidden %d, %d heads x %d' % (n_layers, d_msa, msa_in, d_inputs,
                                          d_pair, d_hidden, n_heads, head_w))

  MSAEncoder, MSAEncoderBlock, experimental = _encoder_class(args.model)
  print('  native module: modeling_esmfold2%s (%s)'
        % ('_experimental' if experimental else '',
           'msa update BEFORE the outer product, every block'
           if experimental else 'outer product first, last block skips it'))
  net = MSAEncoder(d_msa=d_msa, d_pair=d_pair, d_inputs=d_inputs,
                   d_hidden=d_hidden, n_layers=n_layers, n_heads_msa=n_heads,
                   msa_head_width=head_w)
  # THE LAST BLOCK IS THE INTERESTING PART, and the two lines disagree about it.
  # `MSAEncoder.__init__` hardcodes `is_final_block=(i == n_layers - 1)`, so its
  # last block has no msa_pair_weighted_averaging / msa_transition at all. On the
  # RELEASED line that matches the checkpoint, which carries no such weights. On
  # the EXPERIMENTAL line the checkpoint DOES carry them (12 tensors) and
  # transformers' own loader drops them on the floor as unexpected keys -- so
  # native inference never runs weights the checkpoint was shipped with.
  #
  # `--keep_final_update` rebuilds that block WITH the update, which is what our
  # port does (converters/esmfold2._drops_msa_update reads the checkpoint rather
  # than the index). Dumping both is the only way to say which convention the
  # trained weights want, so the gate can compare against each.
  has_final = any(k.startswith('msa_encoder.blocks.%d.msa_pair_weighted_averaging'
                               % (n_layers - 1)) for k in sd)
  if args.keep_final_update:
    if not has_final:
      raise SystemExit('%s has no final-block msa update to keep' % args.model)
    if experimental:
      raise SystemExit('the experimental MSAEncoderBlock has no is_final_block: '
                       'it runs the msa update in EVERY block already, so there '
                       'is nothing for --keep_final_update to restore')
    net.blocks[n_layers - 1] = MSAEncoderBlock(
        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden, n_heads_msa=n_heads,
        msa_head_width=head_w, is_final_block=False)
  missing, unexpected = net.load_state_dict(
      {k[len('msa_encoder.'):]: v for k, v in sd.items()}, strict=False)
  print('  load_state_dict: %d missing, %d unexpected%s'
        % (len(missing), len(unexpected),
           '  (the final block\'s msa update, IGNORED by native)'
           if unexpected and not args.keep_final_update else ''))
  if missing:
    raise SystemExit('missing weights: %s' % sorted(missing)[:6])
  if args.blocks is not None:
    if not 1 <= args.blocks <= n_layers:
      raise SystemExit('--blocks must be 1..%d' % n_layers)
    net.blocks = net.blocks[:args.blocks]
    print('  TRUNCATED to the first %d of %d blocks' % (args.blocks, n_layers))
  net = net.float().eval()

  n, m = args.n_tok, args.n_msa
  # The SAME construction msa_parity.py uses for its other adapters, so the two
  # harnesses stay comparable: default_rng(0), scaled 0.5 normals.
  rng = np.random.default_rng(0)
  z = (rng.normal(size=(n, n, d_pair)) * 0.5).astype(np.float32)
  msa = (rng.normal(size=(m, n, msa_in)) * 0.5).astype(np.float32)
  s_inputs = (rng.normal(size=(n, d_inputs)) * 0.5).astype(np.float32)
  mask = np.ones((m, n), np.float32)
  if args.nonuniform:
    mask[m // 2:, n // 2:] = 0.0
    print('  NONUNIFORM msa mask: %d of %d entries zero'
          % (int((mask == 0).sum()), mask.size))

  # ESMFold2 hands the encoder [B, L, M, ...] -- tokens before depth -- and
  # splits the 35 raw columns as (33 one-hot, has_deletion, deletion_value).
  # fp32 both sides. torch's LayerNorm refuses a float64 input against float32
  # parameters and .double() does not reach every leaf here, so matching the
  # other adapters' fp32 is both simpler and the same comparison.
  T = lambda a: torch.as_tensor(a, dtype=torch.float32)
  msa_lm = np.transpose(msa, (1, 0, 2))                     # (L, M, msa_in)
  oh = T(msa_lm[None, ..., :msa_in - 2])
  hd = T(msa_lm[None, ..., msa_in - 2])
  dv = T(msa_lm[None, ..., msa_in - 1])
  am = T(np.transpose(mask, (1, 0))[None])                  # (1, L, M)
  with torch.no_grad():
    m_feat = torch.cat([oh, hd.unsqueeze(-1), dv.unsqueeze(-1)], -1)
    m_emb = net.embed(m_feat) + net.project_inputs(T(s_inputs)[None]).unsqueeze(2)
    out = net(T(z)[None], T(s_inputs)[None], oh, hd, dv, am)
    # The MSA half of block 0, as its own tensor. The pair output alone cannot
    # separate the three msa-side steps (pair-weighted averaging, transition,
    # outer product) from the pair-side ones, because everything reaches the
    # pair only through the outer product. Comparing the msa rows isolates
    # steps 1-2, which is the "compare the UPDATE, not the output" rule
    # [[esmfold2-trunk-per-block-method]].
    tok = am[:, :, 0]
    pam = tok.unsqueeze(2) * tok.unsqueeze(1)
    m0 = m_emb
    m1 = m0 + net.blocks[0].msa_pair_weighted_averaging(m0, T(z)[None], pam)
    m2 = m1 + net.blocks[0].msa_transition(m1)
    # and the pair after ONLY the outer product, so step 3 is separable too
    z_opm = T(z)[None] + net.blocks[0].outer_product_mean(m2, am)

  path = args.out or os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      'esmfold2_msa_%s%s%s%s.npz'
      % (args.model,
         '_nonuniform' if args.nonuniform else '',
         '_finalupd' if args.keep_final_update else '',
         '' if args.blocks is None else '_b%d' % args.blocks))
  np.savez(path,
           # (1, L, M, c) -> (M, L, c), the layout our msa_stack takes
           msa_emb=np.transpose(m_emb[0].float().numpy(), (1, 0, 2)),
           # (1, L, M, c) -> (M, L, c), same layout as msa_emb
           msa_after_pwa=np.transpose(m1[0].float().numpy(), (1, 0, 2)),
           msa_after_transition=np.transpose(m2[0].float().numpy(), (1, 0, 2)),
           pair_after_opm=z_opm[0].float().numpy(),
           pair_out=out[0].float().numpy(),
           z=z, msa=msa, s_inputs=s_inputs, mask=mask,
           n_layers=np.int32(n_layers),
           # For the EXPERIMENTAL class this is always 1: its block has no
           # is_final_block and runs the msa update in every block. Recording
           # the module's behaviour, not the flag, so the reader of the npz is
           # not told "OFF (native default)" about a block that ran it.
           final_update=np.int32(bool(args.keep_final_update) or experimental),
           experimental=np.int32(experimental))
  print('  wrote %s  pair_out %s' % (path, tuple(out.shape)))
  return 0


if __name__ == '__main__':
  sys.exit(main())
