"""L1-on-a-REAL-INPUT: the input embedder, the trunk, and recycling.

Every other gate here feeds the trunk RANDOM s/z, or compares one module on
synthetic inputs. That leaves three things measured by nothing:

  * the INPUT EMBEDDER (`create_target_feat_embedding` -> s_inputs), which five
    oracles build and none compares;
  * the trunk's actual output on a real input, rather than one block or one
    stack on noise;
  * the RECYCLING loop, since L1-L4 all measure a single pass.

This closes all three at once by comparing against native's own tensors, taken
from a real inference job -- which means native's featuriser too, not ours.

  # 1. native side (needs its own venv; see the script's header)
  PX_DEPS=/path/to/px_deps bash dev/oracles/native_trunk_dump.sh 5k9p_plain
  # 2. ours, and the comparison
  PYTHONPATH=src:. python dev/oracles/real_trunk_parity.py protenix2 \
      --native dev/oracles/native_trunk_5k9p_plain.npz --case 5k9p_plain

READ THE CONTROL BEFORE BELIEVING A NUMBER HERE. A trunk correlation well
below 1.0 is NORMAL: 48 blocks x 10 recycles amplify float differences, and on
6MRR -- where our fold matches native at 0.70 A -- the pair still only
correlates 0.960. So always run a target the port folds WELL beside the target
under suspicion, or a 0.87 will read as a bug when it is the weather. That
mistake cost most of a session.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _cmp(tag, ours, native):
  a = np.asarray(ours, np.float64).ravel()
  b = np.asarray(native, np.float64).ravel()
  corr = np.corrcoef(a, b)[0, 1]
  rms_a, rms_b = np.sqrt((a ** 2).mean()), np.sqrt((b ** 2).mean())
  print('  %-18s corr %.8f  rms ours/native %.4f  max|d| %.5f  rms(nat) %.4f'
        % (tag, corr, rms_a / max(rms_b, 1e-9), np.abs(a - b).max(), rms_b))
  return corr


def ours(model, seq, seed, ptms):
  """-> our own s_inputs / s_trunk / z_trunk, caught on the way to diffusion."""
  import fold_check
  from alphafold3.common import folding_input
  from alphafold3.model.network import diffusion_head

  caught = {}

  def _hook(self, positions_noisy, noise_level, batch, embeddings, *a, **kw):
    caught['single'] = np.asarray(embeddings['single'], np.float32)
    caught['pair'] = np.asarray(embeddings['pair'], np.float32)
    caught['target_feat'] = np.asarray(embeddings['target_feat'], np.float32)
    raise SystemExit(0)

  orig = diffusion_head.DiffusionHead.__call__
  diffusion_head.DiffusionHead.__call__ = _hook
  chains = [folding_input.ProteinChain(
      id='A', sequence=seq, ptms=list(ptms), unpaired_msa='', paired_msa='',
      templates=[])]
  try:
    fold_check.fold(model, '', chains=chains, seed=seed)
  except SystemExit:
    pass
  finally:
    diffusion_head.DiffusionHead.__call__ = orig
  if not caught:
    raise SystemExit('the diffusion head was never reached')
  return caught


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--native', required=True,
                  help='npz from native_trunk_dump.sh')
  ap.add_argument('--case', default=None,
                  help='json in dev/oracles/native_json (defaults from --native)')
  ap.add_argument('--seed', type=int, default=101)
  args = ap.parse_args(argv)
  sys.argv = sys.argv[:1]

  nat = np.load(args.native)
  case = args.case or os.path.basename(args.native)[len('native_trunk_'):-4]
  jpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'native_json', case + '.json')
  spec = json.load(open(jpath))[0]['sequences'][0]['proteinChain']
  seq = spec['sequence']
  ptms = [(m['ptmType'].replace('CCD_', ''), m['ptmPosition'])
          for m in spec.get('modifications', [])]
  print('%s / %s: %d residues, ptms %s, seed %d'
        % (args.model, case, len(seq), ptms or 'none', args.seed))

  got = ours(args.model, seq, args.seed, ptms)
  print('  shapes: single %s pair %s target_feat %s'
        % (got['single'].shape, got['pair'].shape, got['target_feat'].shape))

  # s_inputs: native's is the 449-wide VENDOR layout, ours the 447-wide AF3
  # one. Gather native down with the vendor's OWN permutation -- rf3's is not
  # of3's, and using the wrong one is worth ~2% on every token.
  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  if 's_inputs' in nat.files:
    _cmp('s_inputs', got['target_feat'], nat['s_inputs'][:, idx])
  if 's_trunk' in nat.files:
    _cmp('s_trunk (single)', got['single'], nat['s_trunk'])
  if 'z_trunk' in nat.files:
    _cmp('z_trunk (pair)', got['pair'], nat['z_trunk'])
  else:
    print('  no z_trunk in the dump -- the hook for it did not fire; a '
          '`pair_z` comparison is NOT a substitute (see the module docstring)')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
