"""Gate a converted ESM-C tower against native's own hidden states.

Written for the SMALL towers (300M / 600M), which ESMFold2's base300M/base600M
variants condition on. The 6B tower gates at corr 0.99973 the same way; this
runs the identical comparison for a tower whose dimensions the converter has
never seen -- 30 layers at d_model 960, 36 at 1152, against 80 at 2560.

Two halves, and the second is the one that matters: the 81 (or 31, or 37) hidden
states, and then the PAIR representation the trunk actually reads, because the
learned layer mix concentrates on the last layers and an error there counts for
more than one at layer 3.

  ~/venv_esm/bin/python converters/oracles/esmc_gate_small.py \
      biohub/ESMC-300M-1500000 <out.npz>     # dump native, then compare in jax
"""
import os
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/alphafold3')
sys.path.insert(0, '/home/ubuntu/alphafold3/src')


def dump_native(hub_dir, seq, out):
  """Native hidden states for `seq`, all layers, BOS/EOS stripped."""
  import torch
  from transformers.models.esmc.modeling_esmc import ESMCModel
  from converters.esmc_vocab import sequence_ids, lm_input_ids

  m = ESMCModel.from_pretrained(hub_dir).cuda().eval()
  ids = torch.as_tensor(lm_input_ids(sequence_ids(seq)))[None].cuda()
  with torch.no_grad():
    o = m(input_ids=ids, output_hidden_states=True)
  # (n_layers + 1, L, d) -- HF returns the embedding plus one per block, the
  # last already through the stack's final LayerNorm, which is the convention
  # converters.esmc_embed.forward reproduces.
  hs = o.hidden_states
  # HF returns a tuple for most models and an already-stacked tensor for this
  # one; take either.
  hs = hs if isinstance(hs, torch.Tensor) else torch.stack(tuple(hs), 0)
  if hs.shape[1] == 1:                       # (layers, batch, L, d)
    h = hs[:, 0]
  else:                                      # (batch, layers, L, d)
    h = hs[0]
  h = h[:, 1:-1, :].float().cpu().numpy()
  np.savez_compressed(out, lm_hidden=h.transpose(1, 0, 2))
  print('wrote %s  %s' % (out, h.transpose(1, 0, 2).shape))


def compare(model_dir, native_npz, seq):
  from converters import esmc_embed
  ours = esmc_embed.embed(seq, model_dir)
  nat = np.load(native_npz)['lm_hidden']
  print('%s: ours %s  native %s' % (model_dir, ours.shape, nat.shape))
  n = min(ours.shape[1], nat.shape[1])
  for li in sorted({0, 1, n // 2, n - 3, n - 2, n - 1}):
    o, x = ours[:, li], nat[:, li]
    print('  layer %-3d relerr %.3e  corr %.8f'
          % (li, np.abs(o - x).max() / max(np.abs(x).max(), 1e-9),
             np.corrcoef(o.ravel(), x.ravel())[0, 1]))
  print('  ALL layers corr %.8f'
        % np.corrcoef(ours[:, :n].ravel(), nat[:, :n].ravel())[0, 1])
  return ours, nat


if __name__ == '__main__':
  from converters.oracles.fold_check import parse_ca
  seq, _ = parse_ca(os.path.expanduser('~/6MRR.pdb'))
  if sys.argv[1] == '--dump':
    dump_native(sys.argv[2], seq, sys.argv[3])
  else:
    compare(sys.argv[1], sys.argv[2], seq)
