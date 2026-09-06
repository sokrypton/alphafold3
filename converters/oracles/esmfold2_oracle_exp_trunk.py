"""Dump NATIVE per-pass trunk activations for an ESMFold2 EXPERIMENTAL release.

This is the generator for every experimental-variant comparison: it produces the
npz that `esmfold2_localise_exp.py` reads. Without it that harness has nothing to
compare against.

What it captures, and why each one earns its place:

  lm_z          the LM shim's pair representation. Injecting NATIVE's takes the
                ESM-C tower and the per-model shim out of a comparison entirely,
                which is how the 600M tier's divergence was shown NOT to be
                either of them.
  trunk_in{i}   the trunk stack's INPUT on pass i -- native's
                `z_init + pair_loop_proj(z)`, i.e. exactly what our `_add_prev`
                produces. Comparing here separates the injection from the blocks.
  trunk_pass{i} its OUTPUT on pass i. The pair (in, out) is what attributes a
                divergence to the recycle or to the 24 blocks; for
                esmfold2_lm600m the injection reads 0.9994 and one pass of the
                blocks drops it to 0.971.
  distogram_logits, sample_atom_coords, feat.*   the ends of the model, for
                whole-model comparisons.

Runs in ~/venv_esm (torch + transformers), NOT the jax venv:

    ~/venv_esm/bin/python converters/oracles/esmfold2_oracle_exp_trunk.py \
        ESMFold2-Experimental-Fast-base600M-step1500k out.npz [target.pdb]

The RELEASED line (esmfold2, esmfold2_fast) declares a different model class and
is not handled here -- use esmfold2_native_variants.py, which picks the class
from the checkpoint's own `architectures` field.
"""

import os
import sys

import numpy as np
import torch

from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel)
from transformers.models.esmfold2.protein_utils import prepare_protein_features

sys.path.insert(0, '/home/ubuntu/alphafold3')
from converters.oracles.esmfold2_native_variants import parse_ca


def dump(hub_name, out, pdb='~/6MRR.pdb', num_loops=3, num_sampling_steps=200):
  seq, _ = parse_ca(os.path.expanduser(pdb))
  src = os.path.expanduser('~/esmfold2_variants/%s' % hub_name)
  if not os.path.isdir(src):
    src = 'biohub/%s' % hub_name
  model = ESMFold2ExperimentalModel.from_pretrained(src).cuda().eval()

  cap = {}
  # The trunk is a plain nn.Module call inside forward and runs once per recycle
  # pass, so these hooks fire num_loops + 1 times. Keeping every pass rather than
  # the last is what lets a divergence be attributed to a pass instead of just
  # observed at the end.
  model.folding_trunk.register_forward_hook(
      lambda m, i, o: cap.setdefault('trunk_pass', []).append(
          o.detach().float().cpu().numpy()))
  model.folding_trunk.register_forward_pre_hook(
      lambda m, i: cap.setdefault('trunk_in', []).append(
          i[0].detach().float().cpu().numpy()))
  model.language_model.register_forward_hook(
      lambda m, i, o: cap.__setitem__('lm_z', o.detach().float().cpu().numpy()))

  feats = {k: v.cuda() for k, v in prepare_protein_features(seq).items()}
  torch.manual_seed(0)
  with torch.no_grad():
    o = model(**feats, num_loops=num_loops, num_diffusion_samples=1,
              num_sampling_steps=num_sampling_steps)

  save = {'lm_z': cap['lm_z']}
  for name in ('trunk_pass', 'trunk_in'):
    for i, a in enumerate(cap.get(name, [])):
      save['%s%d' % (name, i)] = a
  save['trunk_out'] = cap['trunk_pass'][-1]
  save['distogram_logits'] = o['distogram_logits'].float().cpu().numpy()
  save['sample_atom_coords'] = o['sample_atom_coords'].float().cpu().numpy()
  for k, v in feats.items():
    save['feat.' + k] = v.cpu().numpy()
  np.savez_compressed(os.path.expanduser(out), **save)
  print('wrote %s  %s' % (out, {k: np.asarray(v).shape for k, v in save.items()
                                if not k.startswith('feat')}))
  return save


if __name__ == '__main__':
  dump(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ['~/6MRR.pdb']))
