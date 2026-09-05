"""Fold a target with a converted model and report CA-RMSD -- the pipeline gate.

Module-equivalence (converters/oracles/prot_parity.py) says our converted blocks
compute what native's do on synthetic activations. It says nothing about
featurisation. This runs the whole model.

  PYTHONPATH=src:. python converters/oracles/fold_check.py protenix_tiny [6MRR.pdb]
"""
import os
import sys

import numpy as np

A3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
      'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
      'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
      'TYR': 'Y', 'VAL': 'V'}


def parse_ca(path, chain=None):
  """(sequence, CA coords). Dedupes altloc records -- 6MRR has three (LYS14,
  GLU43, ASP64), and counting them twice yields a 71-residue 68-mer."""
  seq, xyz, seen = [], [], set()
  for line in open(path):
    if not line.startswith('ATOM') or line[12:16].strip() != 'CA':
      continue
    if chain and line[21] != chain:
      continue
    key = line[21] + line[22:27]
    if key in seen:
      continue
    seen.add(key)
    res = line[17:20].strip()
    if res not in A3:
      continue
    seq.append(A3[res])
    xyz.append([float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)])
  return ''.join(seq), np.array(xyz)


def kabsch_rmsd(a, b):
  n = min(len(a), len(b))
  a, b = a[:n] - a[:n].mean(0), b[:n] - b[:n].mean(0)
  u, _, vt = np.linalg.svd(a.T @ b)
  d = np.sign(np.linalg.det(u @ vt))
  return float(np.sqrt((((a @ (u @ np.diag([1, 1, d]) @ vt)) - b) ** 2).sum(1).mean()))


def fold(model_name, seq, model_dir=None, seed=0, templates=None):
  import haiku as hk
  import jax
  import jax.numpy as jnp
  from alphafold3.common import folding_input
  from alphafold3.constants import decoded_ccd
  from alphafold3.data import featurisation
  from alphafold3.model import model as af3_model, model_registry
  from alphafold3.model import params as afp
  from alphafold3.model.components import utils
  from alphafold3.model.pipeline import model_features

  model_dir = model_dir or os.path.expanduser('~/ported/%s' % model_name)
  spec = model_registry.get(model_name)
  fold_input = folding_input.Input(
      name=model_name,
      chains=[folding_input.ProteinChain(id='A', sequence=seq, ptms=[],
                                         unpaired_msa='', paired_msa='',
                                         templates=templates or [])],
      rng_seeds=[seed])
  ccd = decoded_ccd.get_ccd()
  featurise = lambda **kw: featurisation.featurise_input(
      fold_input=fold_input, ccd=ccd, buckets=None, **kw)
  batch = featurise()[0]
  if spec.featurise:
    batch = model_features.apply(batch, spec, refeaturise=featurise,
                                 model_dir=model_dir, esm=None, has_msa=False,
                                 fold_input=fold_input)
  cfg = af3_model.Model.Config()
  cfg.global_config.flash_attention_implementation = 'xla'
  spec.configure(cfg)

  @hk.transform
  def forward(b):
    return af3_model.Model(cfg)(b)

  # run_inference's preprocessing: a bare apply on the raw batch raises
  # TracerArrayConversionError on its non-array fields.
  batch = jax.tree_util.tree_map(jnp.asarray,
                                 utils.remove_invalidly_typed_feats(batch))
  out = forward.apply(afp.get_model_haiku_params(model_dir=model_dir),
                      jax.random.PRNGKey(seed), batch)
  return out, batch


def main(argv):
  sys.argv = sys.argv[:1]                       # tokamax parses argv lazily
  model_name = argv[0]
  pdb = argv[1] if len(argv) > 1 else os.path.expanduser('~/6MRR.pdb')
  seq, native = parse_ca(pdb)
  print('%s: %d residues from %s' % (model_name, len(seq), os.path.basename(pdb)))
  out, batch = fold(model_name, seq)
  pos = np.asarray(out['diffusion_samples']['atom_positions'])   # (S, L, 24, 3)
  # dense per-token atom layout is N, CA, C, O, ... so CA is slot 1
  ca = pos[:, :, 1, :]
  rs = [kabsch_rmsd(ca[i], native) for i in range(ca.shape[0])]
  print('  %d samples, CA-RMSD: %s   best %.3f  mean %.3f'
        % (len(rs), ' '.join('%.3f' % r for r in rs), min(rs), float(np.mean(rs))))
  return pos, native, batch


if __name__ == '__main__':
  main(sys.argv[1:])
