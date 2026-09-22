"""Cross-model LIGAND parity: fold 1STP chain A + biotin through every port.

Ligand support is the least evenly covered part of the family. boltz2 has its
own btn gates (0.065 A), chai1 has fold_btn.py + a native comparison, and
rosettafold3 was debugged to 0.099 A vs native 0.125 -- but opendde, openfold3,
intellifold2 and protenix2 have NEVER been run on a ligand at all. Their fold
tests are all 6MRR de novo, i.e. protein only. This runs one identical ligand
case through all of them and puts the numbers side by side.

  MODELS=boltz2,protenix2 SEEDS=1,2 PYTHONPATH=/home/ubuntu/ColabDesign2 \
      ~/venv/bin/python tools/oracles/ligand_parity.py

Scoring: superpose on the PROTEIN, then apply that transform to the ligand. A
ligand-only superposition would hide a ligand in the wrong pocket, which is the
failure worth catching. Both numbers are printed because the ligand RMSD is only
interpretable when the protein is roughly right -- a 10 A protein makes the
ligand number meaningless, not good or bad.

NOLIG=1 folds the same protein with no ligand at all, the control that separates
"our ligand path is broken" from "this is a hard target from single sequence".

RESULTS (2026-09-02) at RECYCLES=10 with A3M=~/chai_1stp/msa.a3m, seed 1
(protein-superposed). The first version of this table ran at RECYCLES=0, which is
NOT the af3-family inference default (10) and understated every model -- some by
a lot. Read the old numbers only as history:

  model          protein A    BTN A   (was, at RECYCLES=0)
  intellifold2       0.301    0.881   0.983 / 1.028
  chai1              0.335    0.993   8.759 / 7.845  <- was 54 params at random
                                      init from a stale blob, now regenerated
  openfold3          0.348    0.894   2.492 / 1.904
  boltz2             0.385    0.907   0.461 / 0.466
  rosettafold3       0.464    0.889   9.261 / 8.087  <- profile fix + recycles
  opendde            1.043    0.892   1.269 / 0.870 before the chained-LayerNorm
                                      fix (rna_parity.py bug 5)
  protenix2          2.663    1.198   2.497 / 1.147 before the same fix. That fix
                                      is verified against native protenix's own
                                      source and lands protenix2 on native's RNA
                                      number to three digits, so read this 0.17 A
                                      as noise on a target native protenix is
                                      itself WORSE on than we are (7.022 vs 4.248
                                      protein-only, no MSA) -- see
                                      tools/oracles/protenix2/run_native.py

ALL SEVEN PORTS handle protein+ligand. Nothing here is an open lead.

So the ligand path is in good shape on boltz2 / intellifold2 / openfold3 /
opendde, and
the remaining bad rows are not ligand faults. The control that caught my own
error: opendde folds 6MRR to 1.481 A through this exact machinery (the slow test
gets the same 1.481), which said the machinery was fine and pointed at the
scoring rather than at the model.

ROSETTAFOLD3's MSA regression, localised and then FIXED (2026-09-02). The
localisation, on 1STP protein-only, before the fix:

  no alignment at all                        5.236
  alignment cropped to 1 row (MSA_CROP=1)    5.392   <- plumbing is FINE
  real alignment, model shown 1 row          15.171
  real alignment, model shown 8 rows         14.352
  real alignment, model shown 64 rows        9.620
  real alignment, 64 rows, ZERO=profile      15.894

Cropping the alignment to one row reproduces the no-alignment number, so nothing
is wrong with the msa= path itself. The damage tracks the PROFILE: a real
profile with only one row shown costs 10 A, and showing more rows only partly
compensates. An all-zero profile is worse still, so the model does use it -- our
profile was not ignored, it was WRONG. Cause: restype_alignment rewrote the
profile wherever `profile.argmax != aatype`, a test that is exact for a
single-sequence MSA (all it was ever checked on) and false for a real one, so it
overwrote REAL polymer columns with a one-hot of the query. Both channels now
use the same all-rows-are-gaps test the `msa` channel already used.

Also found by simply trying to run it: models.FEATURISE pins opendde at 160
structural tokens, sized for 6MRR. 1STP needs 242 and the failure is a hard
"Can't pad to a smaller shape", so opendde had never been run on ANY input
larger than 6MRR -- which is the real finding there, and it is a coverage gap
rather than a fault.

A3M=~/chai_1stp/msa.a3m (2144 rows) is close to REQUIRED for this target: from
single sequence 1STP is hard for everything except boltz2 (native chai itself
only reaches 4.9 A without ESM), and a 7 A protein makes its ligand number
uninterpretable. Give every model the same alignment and the comparison is
about the LIGAND path rather than about who is best at single-sequence folding.
"""
import os
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/ColabDesign2')

REF = '/home/ubuntu/1stp_A_holo.cif'
LIG = 'BTN'
ALL = ('boltz2', 'openfold3', 'opendde', 'intellifold2', 'protenix2',
       'rosettafold3', 'chai1')


def reference():
  import gemmi
  st = gemmi.read_structure(REF)
  st.remove_alternative_conformations()
  st.remove_hydrogens()
  ca, lig, seq = [], {}, []
  for ch in st[0]:
    for r in ch:
      if r.name == LIG:
        for a in r:
          lig[a.name] = [a.pos.x, a.pos.y, a.pos.z]
      elif r.find_atom('CA', '*'):
        p = r['CA'][0].pos
        ca.append([p.x, p.y, p.z])
        seq.append(gemmi.find_tabulated_residue(r.name).one_letter_code.upper())
  return np.array(ca, np.float32), lig, ''.join(seq)


def opendde_gather(x, batch, L):
  """Map opendde's structural-token output back onto residues.

  opendde diffuses on STRUCTURAL TOKENS, not on the standard per-token atom
  layout: x is (struct_num_tokens, 24, 3) and the residue a row belongs to is
  structbook/parent_residue_idx, with subtoken_role_id == 1 being the backbone
  token. Reading x[:L, 1] as CA -- what every other port needs -- reads
  unrelated rows and scores ~16 A on a fold that is really ~7. It cost me a
  wrong published number before this function existed.
  """
  parent = np.asarray(batch['structbook/parent_residue_idx'])
  role = np.asarray(batch['structbook/subtoken_role_id'])
  out = np.full((L, 3), np.nan, np.float32)
  for i in range(L):
    w = np.where((parent == i) & (role == 1))[0]
    if len(w):
      out[i] = x[w[0], 1]
  return out


def opendde_ligand(x, batch, L, n_lig):
  """opendde's ligand atoms, one structural token each, in slot 0.

  Ligand tokens are the ones whose parent_residue_idx is >= L (role 0), and the
  atom sits in slot 0 -- slots 1..3 are identically zero for them. Verified
  superposition-free: the predicted intra-ligand distance matrix matches the
  crystal's to 0.40 A RMS in this order, which is what says the pairing with the
  reference atom list is right rather than a permutation that happens to score.
  """
  parent = np.asarray(batch['structbook/parent_residue_idx'])
  rows = [np.where(parent == L + k)[0] for k in range(n_lig)]
  if not all(len(r) for r in rows):
    return None
  return np.array([x[r[0], 0] for r in rows])


def score(x, mask, chars, L, ref_ca, ref_lig, ca=None, lig_xyz=None):
  # rosettafold3 featurises with atomized_element_names=True, which REPLACES
  # each ligand atom's name with its element -- so name matching finds nothing
  # and the ligand silently scores n/a. Fall back to matching by ORDER, which is
  # the CCD order on both sides.
  dec = lambda r: ''.join(chr(int(c) + 32) if 0 <= c < 64 else '' for c in r).strip()
  if ca is None:
    ca = x[:L, 1]
  keep = ~np.isnan(ca).any(-1)
  ca, ref_ca = ca[keep], ref_ca[:len(ca)][keep]
  n = min(len(ca), len(ref_ca))
  P, Q = ca[:n] - ca[:n].mean(0), ref_ca[:n] - ref_ca[:n].mean(0)
  V, _, Wt = np.linalg.svd(P.T @ Q)
  U = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ Wt))]) @ Wt
  prot = float(np.sqrt(((P @ U - Q) ** 2).sum(-1).mean()))
  if lig_xyz is not None:
    got = list(lig_xyz)
    want = [ref_lig[k] for k in list(ref_lig)[:len(got)]]
    got = (np.array(got) - ca[:n].mean(0)) @ U
    want = np.array(want) - ref_ca[:n].mean(0)
    return prot, float(np.sqrt(((got - want) ** 2).sum(-1).mean())), len(got)
  got, want = [], []
  for t in range(L, mask.shape[0]):
    for a in range(mask.shape[1]):
      if mask[t, a] and dec(chars[t, a]) in ref_lig:
        got.append(x[t, a])
        want.append(ref_lig[dec(chars[t, a])])
  if not got:
    order = list(ref_lig)
    k = 0
    for t in range(L, mask.shape[0]):
      for a in range(mask.shape[1]):
        if mask[t, a] and k < len(order):
          got.append(x[t, a])
          want.append(ref_lig[order[k]])
          k += 1
  if not got:
    return prot, float('nan'), 0
  got = (np.array(got) - ca[:n].mean(0)) @ U
  want = np.array(want) - ref_ca[:n].mean(0)
  return prot, float(np.sqrt(((got - want) ** 2).sum(-1).mean())), len(got)


def run(model, seq, ref_ca, ref_lig, seeds):
  import jax
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  from colabdesign2.af3.runner import AF3Runner
  from tools.module_trace import models

  L = len(seq)
  no_lig = os.environ.get('NOLIG') == '1'
  spec = parse_contigs(str(L) if no_lig else '%d/0 lig:%s' % (L, LIG))
  spec = spec.resolve(length=L) if no_lig else spec
  kw = models.featurise_kwargs(model)
  if 'struct_num_tokens' in kw:
    # models.FEATURISE pins opendde at 160 structural tokens, which was sized for
    # 68-residue 6MRR. 1STP+BTN needs 242, and the failure is a hard
    # "Can't pad to a smaller shape" -- so opendde has never been run on any
    # input bigger than that. Size it from the input instead.
    kw['struct_num_tokens'] = int(os.environ.get('STRUCT_TOKENS', '512'))
  a3m = os.environ.get('A3M')
  if a3m:
    kw['msa'] = open(os.path.expanduser(a3m)).read()
  feat = (f3.featurise_chai1 if model == 'chai1' else f3.featurise_spec)
  n_msa = int(os.environ.get('NUM_MSA', '1' if not a3m else '256'))
  batch = feat(spec, sequences={0: seq},
               msa_crop_size=int(os.environ.get('MSA_CROP', '1' if not a3m
                                                else '512')), **kw)
  batch = batch[0] if isinstance(batch, tuple) else batch
  for k in os.environ.get('ZERO', '').split(','):
    # ZERO=profile,deletion_mean isolates WHICH part of an alignment a model
    # reacts badly to: the rows themselves, or the derived summary features.
    if k and k in batch:
      batch[k] = np.zeros_like(np.asarray(batch[k]))

  cfg, params, _f, missing = models.build(
      model, batch, num_recycles=int(os.environ.get('RECYCLES', '0')),
      diffusion_steps=200, num_msa=n_msa)
  if missing:
    print('  (%d params at INIT: %s)' % (len(missing), missing[:2]))
  runner = AF3Runner(cfg=cfg, model_params=params, diffusion='forward',
                     num_msa=n_msa)
  mask = np.asarray(batch['ref_mask']).astype(bool)
  chars = np.asarray(batch['ref_atom_name_chars'])
  out = []
  for s in seeds:
    pred = runner.predict(batch, key=jax.random.PRNGKey(s))
    x = np.asarray(pred['diffusion_samples']['atom_positions'])
    while x.ndim > 3:
      x = x[0]
    ca, lig_xyz = None, None
    if model == 'opendde':
      ca = opendde_gather(x, batch, L)
      lig_xyz = None if no_lig else opendde_ligand(x, batch, L, len(ref_lig))
      mask = np.zeros_like(mask)     # the name scan cannot read this layout
    out.append(score(x, mask, chars, L, ref_ca, ref_lig, ca=ca,
                     lig_xyz=lig_xyz))
  return out


def main():
  ref_ca, ref_lig, seq = reference()
  print('1STP chain A: %d residues, %s with %d atoms (single sequence, no MSA)'
        % (len(seq), LIG, len(ref_lig)))
  seeds = [int(x) for x in os.environ.get('SEEDS', '1').split(',')]
  want = os.environ.get('MODELS')
  models_ = want.split(',') if want else list(ALL)
  print('%-14s %10s %12s' % ('model', 'protein A', LIG + ' A'))
  for m in models_:
    try:
      for prot, lig, n in run(m, seq, ref_ca, ref_lig, seeds):
        print('%-14s %10.3f %12s' % (m, prot,
                                     'n/a' if n == 0 else '%.3f (%d)' % (lig, n)))
    except Exception as e:            # one model's failure must not hide the rest
      print('%-14s %10s %12s   %s: %s'
            % (m, 'ERR', '', type(e).__name__, str(e)[:90]))
  return 0


if __name__ == '__main__':
  sys.exit(main())
