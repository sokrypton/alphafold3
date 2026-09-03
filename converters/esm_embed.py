"""Precompute the ESM2 token embeddings chai-1 folds with.

chai-1's TOKEN feature stream is mostly ESM2: zeroing it drops the token
embedding to corr 0.327 of chai's, and a natural protein folds to 5.70 A where
chai reaches 0.642. So running chai-1 without this is running a different model.

It lives here, on the converter side, for the same reason the weight converters
do: chai ships ESM as a traced TorchScript archive, and the served path does not
import torch. Produce the npz once, pass it to a run with --esm_embeddings.

    python -m converters.esm_embed --json fold_input.json --out esm.npz

Verified against chai itself rather than assumed: hooking FeatureEmbedding during
an ESM-on native run and comparing its ESMEmbeddings argument to this output
gives max|d| = 0 over all 121 x 2560 entries. The raw last hidden state IS the
feature verbatim -- no normalisation, no transform.
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np


# chai's own tokenisation (chai_lab/data/dataset/embeddings/esm.py): the standard
# ESM2 vocabulary, the sequence wrapped as <cls>{seq}<eos>, and the BOS/EOS rows
# dropped so the result is exactly one 2560-dim row per residue.
_VOCAB = ('<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B '
          'U Z O . - <null_1> <mask>').split()
_TOKENS = {t: i for i, t in enumerate(_VOCAB)}

_DEFAULT_ESM = '~/chai1_weights/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt'


def embed(sequences, esm_path=_DEFAULT_ESM, device='cuda'):
  """-> (sum(len(s)), 2560) float32, the chains' rows concatenated in order."""
  import torch

  model = torch.jit.load(os.path.expanduser(str(esm_path)),
                         map_location='cpu').to(device).eval()
  rows = []
  for seq in sequences:
    ids = [_TOKENS['<cls>']] + [_TOKENS[c] for c in seq] + [_TOKENS['<eos>']]
    with torch.no_grad():
      hidden = model(tokens=torch.as_tensor(ids)[None].to(device))
    out = np.asarray(hidden[0, 1:-1].float().cpu())
    if out.shape[0] != len(seq):
      raise ValueError(f'{out.shape[0]} rows for a {len(seq)}-residue chain')
    rows.append(out)
  return np.concatenate(rows, axis=0).astype(np.float32)


def protein_sequences(json_path):
  """The protein chain sequences of an AlphaFold 3 input JSON, in chain order.

  Order matters: the rows land on the batch's protein tokens in order, so they
  have to be concatenated the way the featuriser lays those tokens out.
  """
  from alphafold3.common import folding_input

  fold_input = folding_input.Input.from_json(
      pathlib.Path(json_path).read_text())
  return [chain.sequence for chain in fold_input.chains
          if isinstance(chain, folding_input.ProteinChain)]


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
  p.add_argument('--json', help='the fold input JSON to read sequences from')
  p.add_argument('--sequence', action='append', default=[],
                 help='a protein sequence, repeatable, in chain order. Use this'
                      ' rather than --json where alphafold3 is not importable'
                      ' (the torch environment that has chai ESM usually is not'
                      ' the one that has alphafold3).')
  p.add_argument('--out', required=True, help='npz to write')
  p.add_argument('--esm', default=_DEFAULT_ESM,
                 help="chai's traced ESM2 archive")
  p.add_argument('--device', default='cuda')
  args = p.parse_args(argv)

  if bool(args.json) == bool(args.sequence):
    raise SystemExit('pass exactly one of --json or --sequence')
  sequences = args.sequence or protein_sequences(args.json)
  if not sequences:
    raise SystemExit(f'{args.json} has no protein chains')
  rows = embed(sequences, args.esm, args.device)
  np.savez_compressed(args.out, esm=rows,
                      sequences=np.array(sequences, dtype=object),
                      allow_pickle=True)
  print(f'{len(sequences)} chain(s), {rows.shape[0]} residues -> {args.out} '
        f'{rows.shape}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
