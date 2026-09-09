"""NATIVE ESMFold2 folded WITH a real MSA -- the only way to get that number.

The vendor's own MSA featuriser (`ESMFold2InputBuilder`) is not in the shipped
`transformers` release, which is why "native with an MSA" looked unobtainable.
It is not: the EXPERIMENTAL forward takes `msa`, `has_deletion`,
`deletion_value` and `msa_attention_mask` as arguments, so the tensors can be
built here from the same a3m our own fold reads.

    VAR=ESMFold2-Experimental DEPTH=256 \
      ~/venv_esm/bin/python dev/oracles/esmfold2_native_msa.py
    # then score the dump with the in-repo reference:
    PYTHONPATH=src:. python dev/oracles/esmfold2_score_native.py

This produced the measurement that turned esmfold2's MSA problem from a guess
into a direction (PARITY.md, "the experimental line's MSA path is INVERTED"):

    esmfold2_exp on 1STP    no MSA      with MSA
      native                18.728 A    3.184 A   (improves)
      ours                   0.476 A   14.364 A   (degrades)

Three things it takes to get right, all of which cost a run each:

  * `prepare_protein_features` ALREADY returns a depth-1 self `msa`, so a real
    MSA has to REPLACE that key -- passing it alongside raises "got multiple
    values for keyword argument 'msa'", and torch buries that under the module
    repr so the message looks empty.
  * a3m rows carry lowercase INSERTIONS relative to the query. They have to be
    stripped (and counted, as has_deletion / deletion_value) or the row is not
    query-length and cannot be stacked.
  * the CA reference must come from `modality_check.reference()`, which resolves
    1STP's numbering. A hand-rolled CA list read 18.786 A for coordinates the
    in-repo scorer also reads 18.786 A -- so that one happened to agree, but
    only because the numbering lines up here; do not rely on it.
"""
import json
import os

import numpy as np
import torch
from transformers.models.esmfold2 import protein_utils as U
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel)

VAR = os.environ.get('VAR', 'ESMFold2-Experimental')
JSON = os.path.expanduser('~/1stp_msa.json')
CIF = os.path.expanduser('~/1STP.cif')
MAXD = int(os.environ.get('DEPTH', '256'))

d = json.load(open(JSON))
prot = [c['protein'] for c in d['sequences'] if 'protein' in c][0]
seq, a3m = prot['sequence'], prot.get('unpairedMsa', '')
rows = [l.strip() for l in a3m.splitlines() if l and not l.startswith('>')]
rows = [r for r in rows if r]
print('%s: %d residues, %d MSA rows in the json' % (VAR, len(seq), len(rows)))

# letters -> ESMFold2's 33-class res_type ids; '-' is MSA_GAP_TOKEN_ID
res = {a: U.PROTEIN_RESIDUE_TO_RES_TYPE[t] for a, t in U.PROTEIN_1TO3.items()
       if t in U.PROTEIN_RESIDUE_TO_RES_TYPE}
unk = U.PROTEIN_UNK_RES_TYPE


def encode(a3m_row):
  """a3m -> (ids over the QUERY columns, has_deletion, deletion_value)."""
  ids, hasdel, delval = [], [], []
  pending = 0
  for ch in a3m_row:
    if ch.islower():                        # an insertion relative to the query
      pending += 1
      continue
    ids.append(U.MSA_GAP_TOKEN_ID if ch == '-' else res.get(ch.upper(), unk))
    hasdel.append(1.0 if pending else 0.0)
    delval.append(float(pending))
    pending = 0
  return ids, hasdel, delval


keep = [rows[0]] + rows[1:MAXD]
enc = [encode(r) for r in keep]
L = len(seq)
enc = [e for e in enc if len(e[0]) == L]
print('  usable rows (query-length after removing insertions): %d' % len(enc))
msa = torch.tensor([e[0] for e in enc], dtype=torch.long)[None]           # (1,M,L)
hd = torch.tensor([e[1] for e in enc], dtype=torch.float32)[None]
dv = torch.tensor([e[2] for e in enc], dtype=torch.float32)[None]
dv = 2.0 / np.pi * torch.arctan(dv / 3.0)                                 # AF3's scaling
mm = torch.ones_like(msa, dtype=torch.float32)

local = os.path.expanduser('~/esmfold2_variants/%s' % VAR)
m = ESMFold2ExperimentalModel.from_pretrained(local).cuda().eval()
feats = {k: v.cuda() for k, v in U.prepare_protein_features(seq).items()}
feats.pop('input_ids', None)


def ca_rmsd(x, ref):
  a, b = x - x.mean(0), ref - ref.mean(0)
  u, _, vt = np.linalg.svd(a.T @ b)
  s = np.sign(np.linalg.det(u @ vt))
  return float(np.sqrt((((a @ (u @ np.diag([1, 1, s]) @ vt)) - b) ** 2).sum(1).mean()))


# The crystal CAs for chain A, in sequence order. Parsed from the mmCIF's
# atom_site loop by hand: gemmi is not in the vendor venv and this needs one
# column set.
def crystal_ca(path, chain='A'):
  cols, rows, in_loop = [], [], False
  for line in open(path):
    t = line.strip()
    if t.startswith('_atom_site.'):
      cols.append(t.split('.', 1)[1]); in_loop = True; continue
    if in_loop:
      if t.startswith(('#', 'loop_', '_')) or not t:
        break
      rows.append(t.split())
  ix = {c: i for i, c in enumerate(cols)}
  out, seen = [], set()
  for r in rows:
    if len(r) < len(cols):
      continue
    if r[ix['label_atom_id']] != 'CA':
      continue
    if r[ix.get('auth_asym_id', ix['label_asym_id'])] != chain:
      continue
    key = r[ix.get('auth_seq_id', ix['label_seq_id'])]
    if key in seen:
      continue
    seen.add(key)
    out.append([float(r[ix['Cartn_%s' % k]]) for k in 'xyz'])
  return np.array(out)

nat = crystal_ca(CIF)[:L]
print('  crystal CAs: %d' % len(nat))

ch = feats['ref_atom_name_chars'][0].cpu().numpy().astype(int)
a2t = feats['atom_to_token'][0].cpu().numpy().astype(int)
msk = feats['atom_attention_mask'][0].cpu().numpy().astype(bool)
nm = lambda i: ''.join(chr(c + 32) for c in ch[i]).strip()
rep = np.full(int(a2t[msk].max()) + 1, -1)
for i in range(len(a2t)):
  if msk[i] and nm(i) == 'CA':
    rep[a2t[i]] = i

out = {}
for label, kw in (('NO msa', {}),
                  ('WITH msa (%d rows)' % msa.shape[1],
                   dict(msa=msa.cuda(), has_deletion=hd.cuda(),
                        deletion_value=dv.cuda(), msa_attention_mask=mm.cuda()))):
  torch.manual_seed(0)
  try:
    with torch.no_grad():
      # `prepare_protein_features` already supplies a depth-1 self `msa`, so
      # the real MSA has to REPLACE that key rather than be passed alongside it
      # ("got multiple values for keyword argument 'msa'").
      call = dict(feats)
      call.update(kw)
      o = m(**call, num_loops=3, num_diffusion_samples=1,
            num_sampling_steps=200)
  except (TypeError, RuntimeError) as e:
    import traceback
    tb = traceback.format_exc().splitlines()
    # torch puts the module repr in the message, so print the FRAMES instead
    print('  NATIVE %-22s %s' % (label, type(e).__name__))
    for l in [x for x in tb if x.strip().startswith('File')][-3:]:
      print('     ', l.strip()[:150])
    for l in tb[-3:]:
      print('     ', l.strip()[:200])
    continue
  x = o['sample_atom_coords'].float().cpu().numpy()
  if x.ndim == 3:
    x = x[0]
  n = min(len(rep), len(nat))
  print('  NATIVE %-22s CA-RMSD %.3f A (hand-rolled reference)'
        % (label, ca_rmsd(x[rep[:n]], nat[:n])))
  out['ca|' + label] = x[rep[:n]]
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dumps'), exist_ok=True)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dumps', 'esmfold2_native_1stp.npz'), seq=np.array([seq], dtype=object),
         **out)
print('  wrote native_exp_1stp.npz -- score it with the in-repo reference, which '
      'handles 1STP\'s numbering; the hand-rolled one above almost certainly '
      'does not (chain A does not start at residue 1)')
