"""Does an MSA substitute for ESM-C in OUR port?  5CAJ, depth 1 vs depth 256.

The counterpart of esmfold2_msa_vs_lm.py, which asks the same question of NATIVE
ESMFold2. The panel that folded 5CAJ and friends badly ran with NEITHER ESM-C
nor an MSA -- a depth-1 self-MSA is no evolutionary information at all, so those
numbers say nothing about the MSA path.

Scoring uses 5CAJ's auth numbering, not position: chain A carries a 5-residue
expression tag (PRGSH, auth -4..0) and six unresolved loops, so pred[:n] vs
nat[:n] drifts by a growing offset and reads ~6 A on a good structure.

  MSA=256 PYTHONPATH=src:. ~/venv/bin/python converters/oracles/esmfold2_msa_gate.py
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

sys.path.insert(0, '/home/ubuntu/alphafold3')
sys.path.insert(0, '/home/ubuntu/alphafold3/src')
sys.argv = sys.argv[:1]
from alphafold3.model import model as af3_model, model_registry, params as afp
from alphafold3.model.components import utils
from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model.pipeline import model_features

A3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
      'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
      'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
      'TYR': 'Y', 'VAL': 'V'}


def parse_cif(path, chain='A'):
  cols, rows, inloop = {}, [], False
  for line in open(path):
    s = line.strip()
    if s.startswith('_atom_site.'):
      cols[s.split('.')[1]] = len(cols); inloop = True; continue
    if inloop:
      if s.startswith('#') or s.startswith('loop_') or s.startswith('_'):
        if rows: break
        continue
      p = s.split()
      if len(p) >= len(cols): rows.append(p)
  out, seen = [], set()
  for p in rows:
    if p[cols['group_PDB']] != 'ATOM' or p[cols['label_atom_id']] != 'CA':
      continue
    if p[cols['auth_asym_id']] != chain:
      continue
    key = p[cols['auth_seq_id']]
    if key in seen:
      continue
    seen.add(key)
    out.append((int(key), A3.get(p[cols['label_comp_id']], 'X'),
                [float(p[cols['Cartn_%s' % a]]) for a in 'xyz']))
  return out


def read_a3m(path, max_depth):
  seqs, cur = [], None
  for line in open(path):
    if line.startswith('>'):
      if cur is not None:
        seqs.append(cur)
      cur = ''
      if len(seqs) >= max_depth:
        break
    else:
      cur += line.strip()
  if cur is not None and len(seqs) < max_depth:
    seqs.append(cur)
  return seqs


def kabsch(a, b):
  a = a - a.mean(0); b = b - b.mean(0)
  u, _, vt = np.linalg.svd(a.T @ b)
  d = np.sign(np.linalg.det(u @ vt))
  r = u @ np.diag([1.0, 1.0, d]) @ vt
  return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def main():
  depth = int(os.environ.get('MSA', '256'))
  a3m_path = os.environ.get('A3M', os.path.expanduser('~/msa_yaaa.a3m'))
  cif = os.environ.get('CIF', os.path.expanduser('~/5CAJ.cif'))
  a3m = read_a3m(a3m_path, max(depth, 1))
  seq = a3m[0]
  res = parse_cif(cif)
  pairs = [(n - 1, xyz) for n, c, xyz in res if 1 <= n <= len(seq) and seq[n - 1] == c]
  sel = np.array([i for i, _ in pairs])
  nat = np.array([x for _, x in pairs])

  model_dir = os.path.expanduser('~/ported/esmfold2')
  spec = model_registry.get('esmfold2')
  # depth 1 = the query alone, which is what a single-sequence run gets.
  msa_text = '' if depth <= 1 else ''.join(
      '>seq%d\n%s\n' % (i, s) for i, s in enumerate(a3m[:depth]))
  fold_input = folding_input.Input(
      name='msa_gate',
      chains=[folding_input.ProteinChain(id='A', sequence=seq, ptms=[],
                                         unpaired_msa=msa_text, paired_msa='',
                                         templates=[])],
      rng_seeds=[1])
  ccd = decoded_ccd.get_ccd()
  featurise = lambda **kw: featurisation.featurise_input(
      fold_input=fold_input, ccd=ccd, buckets=None, **kw)
  batch = featurise()[0]
  if spec.featurise:
    batch = model_features.apply(batch, spec, refeaturise=featurise,
                                 model_dir=model_dir, esm=None,
                                 has_msa=depth > 1, fold_input=fold_input)
  cfg = af3_model.Model.Config()
  cfg.global_config.flash_attention_implementation = 'xla'
  spec.configure(cfg)

  @hk.transform
  def forward(b):
    return af3_model.Model(cfg)(b)

  batch = jax.tree_util.tree_map(jnp.asarray,
                                 utils.remove_invalidly_typed_feats(batch))
  out = forward.apply(afp.get_model_haiku_params(model_dir=model_dir),
                      jax.random.PRNGKey(1), batch)
  pos = np.asarray(out['diffusion_samples']['atom_positions'])
  ca = pos[:, :, 1, :][:, sel, :]
  rs = [kabsch(ca[i], nat) for i in range(ca.shape[0])]
  print('5CAJ %d aa, %d resolved CA aligned by auth number, MSA depth %d'
        % (len(seq), len(sel), len(a3m[:depth]) if depth > 1 else 1))
  print('  %d samples, CA-RMSD: %s   best %.3f  mean %.3f'
        % (len(rs), ' '.join('%.3f' % r for r in rs), min(rs), float(np.mean(rs))))


if __name__ == '__main__':
  main()
