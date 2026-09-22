"""Do the ports get a protein-protein INTERFACE right? One complex, all seven.

The gap this closes: every af3-family fold gate is a single chain, or a chain
plus a ligand. The machinery for complexes -- chain-pair PAE/PDE, ipTM,
cross-chain templates, paired MSAs -- exists and was exercised by nothing. That
is the case the project actually needs, because binder design is multi-chain and
i_con / i_pae / ipTM are its objectives.

The case is an AlphaFold3-server job (~/af3_complex): a 145 + 74 residue
heterodimer it predicts confidently (ipTM 0.91, pTM 0.91, ranking 0.94), shipping
its own per-chain PAIRED and unpaired MSAs, templates, and five ranked models.
Both chains' chai ESM2 embeddings are precomputed alongside.

READ THE REFERENCE HONESTLY. `model_0.cif` is AF3's own PREDICTION, not an
experimental structure. So for openfold3 -- AF3's own lineage and weights -- close
agreement is near-tautological, and it is the CONTROL that says the screen is
wired up. For the other six, agreement is cross-model agreement: a large number
means they disagree with AF3, which is evidence of a port bug only when the other
screens are clean, and evidence about the model otherwise.

WHY GLOBAL RMSD IS THE WRONG METRIC. A prediction that folds both subunits well
and docks them wrongly can still post a respectable global CA-RMSD, because the
two good chains dominate. So the columns are separated:

  A / B     each chain's INTERNAL RMSD, superposed on itself -- subunit quality
  iface     chain B's RMSD after superposing on chain A ALONE -- the docking.
            Anything that gets the subunits right and the interface wrong shows
            up here and nowhere else.
  fnat      fraction of the reference's inter-chain CA-CA contacts (< 8 A) that
            the prediction also makes. RMSD-free, so it survives a rigid-body
            offset that iface punishes, and it is the DockQ component that speaks
            to whether the right surfaces are touching.
  pLDDT     mean per chain, now comparable across ports (see confidence_parity).

  MODELS=openfold3,boltz2 MSA=1 RECYCLES=10 \\
  PYTHONPATH=/home/ubuntu/ColabDesign2 \\
      ~/venv/bin/python tools/oracles/multimer_parity.py

MSA=1 (default) uses the job's own per-chain paired + unpaired alignments. The
PAIRED rows are the interesting half for a complex -- they are what tell the model
which subunits co-evolve, and IsPairedMSA is derived from cross-chain coverage, so
it is only ever non-zero here. MSA=0 asks the much harder single-sequence
question instead, which is a different experiment, not a cheaper one.

RESULTS (2026-09-02, the job's own paired + unpaired MSAs, num_msa 256, crop 512,
10 recycles, seed 1). ALL SEVEN PORTS GET THE INTERFACE RIGHT -- nothing here is a
port bug:

  model            A      B    iface   fnat   pLDDT A/B
  intellifold2  0.45   0.31     0.92   0.94   97.5/96.9
  boltz2        0.36   0.43     0.92   0.91   97.4/96.2
  protenix2     0.43   0.49     1.02   0.92   95.3/94.5
  rosettafold3  0.39   0.67     1.12   0.95   89.2/87.5
  opendde       1.02   0.94     1.49   0.84   93.6/92.4
  chai1         1.16   0.71     1.57   0.79   92.0/90.8
  openfold3     0.73   0.42     1.60   0.93   95.5/96.4

Every subunit is sub-Angstrom to 1.2 A, every interface is under 1.6 A, and every
port makes 79-95% of the reference's inter-chain contacts. Paired MSAs, chain-pair
features and cross-chain attention are therefore all live in all seven -- which is
what this screen existed to find out.

TWO THINGS NOT TO OVER-READ IN THAT ORDERING.

1. The reference job ran with `useStructureTemplate: true` on BOTH chains and this
   screen feeds NO templates. So no row is a strict reproduction of the reference;
   every row is "predict this complex from alignments alone, scored against
   AF3-with-templates". That is a fair cross-model comparison and it is why
   openfold3 -- AF3's own lineage, the row that should otherwise be tautologically
   best -- is not.
2. Seed spread is real but small. openfold3 over seeds 1/2/3 reads iface
   1.60 / 1.33 / 1.35 and intellifold2 0.92 / 1.04 / 0.97, so treat differences
   inside ~0.2 A as noise and do not rank the top four from one seed.

openfold3 is also the clearest illustration of why `iface` and `fnat` are separate
columns: it has the WORST rigid-body placement (1.60) and among the BEST contact
recovery (0.93). Right surfaces touching, placement slightly off -- a single
global RMSD would have blurred exactly that distinction.

TEMPLATES ON A COMPLEX (2026-09-02), the case this screen was extended for --
`TEMPLATE=A` shows chain A's structure from the reference and chain B nothing,
which is the binder-design case: show the TARGET, not the binder. Run with MSA=0,
because with the job's alignments every port already solves this complex and a
template could only move it a little; single sequence is the regime where the
model FAILS and an auxiliary input can be told apart from an inert one.

Per-chain assignment verified at featurisation first: `TEMPLATE=A` templates
146/146 tokens of chain A and 0/74 of chain B. The graph masks cross-chain
template pairs itself (evoformer builds
multichain_mask = asym_id[:, None] == asym_id[None, :]).

  model          iface, no template   iface, template on A   chain A / B
  rosettafold3         19.78                  0.94            0.08 / 0.57
  intellifold2         45.15                  0.99            0.27 / 0.32
  opendde              44.22                  1.58            0.90 / 1.16
  openfold3            46.35                  1.84            0.47 / 0.60
  boltz2               35.20                 14.03            0.55 / 5.58
  protenix2            31.76                 46.76           14.20 / 2.42
  chai1                 FAILED                FAILED               --

FOUR PORTS ARE RESCUED, and note what they do: a template on chain A alone also
folds and docks chain B, which never had one (openfold3 chain B 5.25 -> 0.60).
That is the binder case working end to end.

chai1's row is UNINFORMATIVE, not a pass. Its no-template baseline on this
complex is already 1.39 A -- ESM does for it what an MSA does for the others --
so 1.18 A with a template shows nothing about whether the template is connected.
The screen's own premise (use a case the model FAILS) does not hold for chai here.

boltz2 is PARTIAL -- chain A rescued (12.66 -> 0.55), chain B not (3.27 -> 5.58)
-- and that is NATIVE BEHAVIOUR, not a port bug. Native boltz2 run on the same
input (~/boltz_gpu_venv, templates: [{cif, chain_id: [A], template_id: [A]}],
msa: empty, 10 recycles) does the same thing, and our port tracks it on every
column:

                     A      B    iface   fnat
    native, none   10.14   3.81   42.80   0.00
    native, tmpl A  0.32   6.25   14.91   0.44
    ours,   tmpl A  0.55   5.58   14.03   0.39

So boltz2 really does rescue the templated chain and really does fail to dock the
other one from a single sequence. Nothing to fix; this row is parity.

protenix2 does not use the template HERE, and the obvious explanation is WRONG.
Cross-chain leakage is ruled out with a number: our builder masks the distogram
by pb2d = cb_mask_i * cb_mask_j, and with a template on chain A only the chain-B
side of every cross-chain pair has cb_mask 0 already, so the features computed
with the real multichain mask and with an all-ones mask differ by max|d| 0.0000
over 44.6% cross-chain pairs. (The builder now applies the multichain mask to the
distogram anyway, matching protenix's own forward, which multiplies dgram by
multichain_mask * pair_mask; it only bites when BOTH chains are templated.)

What is left is narrower: protenix2 handles this complex FINE with alignments
(interface 1.02 A in the table above) and its template rescues a MONOMER, so what
it cannot do is let a template stand in for an MSA on a two-chain input. Recorded
as a characteristic, not diagnosed as a bug.

protenix2's template path is LIVE: on
streptavidin from single sequence -- a monomer it fails outright -- it goes
6.47 -> 3.37 A, against openfold3's 8.51 -> 3.23 on the same probe. So the
failure is specific to the two-chain case, not the embedder. (An earlier reading
of this table called it inert and blamed the converter's "task #31"; the 108-d
builder is in fact implemented, as Protenix2TemplateEmbedding, and that
converter docstring is stale.)

ipTM is deliberately not a column yet: the shared head exposes
tmscore_adjusted_pae_* but opendde's own head does not, so an ipTM row would
compare six models against one blank. Worth adding once opendde's head grows the
tm-adjusted PAE.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/ubuntu/ColabDesign2')

REF_DIR = os.path.expanduser(os.environ.get('REF_DIR', '~/af3_complex'))
JOB = 'fold_2026_09_01_10_17'
ALL = ('openfold3', 'intellifold2', 'boltz2', 'opendde', 'protenix2',
       'rosettafold3', 'chai1')


def kabsch(P, Q):
  """Rotation aligning P onto Q, plus both centroids."""
  pc, qc = P.mean(0), Q.mean(0)
  V, _, Wt = np.linalg.svd((P - pc).T @ (Q - qc))
  d = np.sign(np.linalg.det(V @ Wt))
  return V @ np.diag([1, 1, d]) @ Wt, pc, qc


def rmsd_after(U, pc, qc, P, Q):
  return float(np.sqrt((((P - pc) @ U - (Q - qc)) ** 2).sum(-1).mean()))


def ref_chains(cif):
  import gemmi
  st = gemmi.read_structure(cif)
  st.remove_alternative_conformations()
  st.remove_hydrogens()
  out = []
  for ch in st[0]:
    ca = [[r['CA'][0].pos.x, r['CA'][0].pos.y, r['CA'][0].pos.z]
          for r in ch if r.find_atom('CA', '*')]
    if ca:
      out.append(np.array(ca, np.float32))
  return out


def fnat(a, b, ra, rb, cutoff=8.0):
  """Fraction of the reference's inter-chain CA-CA contacts the model makes."""
  ref = np.linalg.norm(ra[:, None] - rb[None, :], axis=-1) < cutoff
  got = np.linalg.norm(a[:, None] - b[None, :], axis=-1) < cutoff
  return float((ref & got).sum() / max(ref.sum(), 1))


def main():
  import jax
  from colabdesign2 import parse_contigs
  from colabdesign2.af3 import features as f3
  from colabdesign2.af3.runner import AF3Runner
  from tools.module_trace import models

  req = json.load(open('%s/%s_job_request.json' % (REF_DIR, JOB)))[0]
  seqs = [c['proteinChain']['sequence'] for c in req['sequences']]
  la, lb = len(seqs[0]), len(seqs[1])
  ref = ref_chains(os.environ.get('REF', '%s/%s_model_0.cif' % (REF_DIR, JOB)))
  ra, rb = ref[0][:la], ref[1][:lb]
  print('%d + %d = %d tokens; reference %s (AF3 server, ipTM 0.91)'
        % (la, lb, la + lb, os.path.basename(
            os.environ.get('REF', '%s_model_0.cif' % JOB))))

  use_msa = os.environ.get('MSA', '1') == '1'
  msa = None
  if use_msa:
    d = '%s/msas/%s_' % (REF_DIR, JOB)
    rd = lambda p: open(p).read()
    msa = {0: (rd(d + 'unpaired_msa_chains_a.a3m'),
               rd(d + 'paired_msa_chains_a.a3m')),
           1: (rd(d + 'unpaired_msa_chains_b.a3m'),
               rd(d + 'paired_msa_chains_b.a3m'))}
  n_msa = int(os.environ.get('NUM_MSA', '256' if use_msa else '1'))
  crop = int(os.environ.get('MSA_CROP', '512' if use_msa else '1'))
  recycles = int(os.environ.get('RECYCLES', '10'))
  seeds = [int(s) for s in os.environ.get('SEEDS', '1').split(',')]

  # TEMPLATE=A|B|both shows the model a chain's structure from the reference.
  # "A" is the binder-design case: show the TARGET and not the binder. Per-chain
  # assignment verified at featurisation -- chain A only templates 146/146 of A
  # and 0/74 of B -- and the graph masks cross-chain template pairs itself
  # (evoformer: multichain_mask = asym_id[:, None] == asym_id[None, :]).
  templates = None
  want_t = os.environ.get('TEMPLATE')
  if want_t:
    import numpy as _np
    from colabdesign.af.prep import prep_pdb
    from colabdesign2.af3.features import spec_to_templates
    pdb = os.environ.get('TEMPLATE_PDB', '%s/%s_model_0.pdb' % (REF_DIR, JOB))
    p = prep_pdb(pdb, chain='A,B', ignore_missing=True)
    tspec = parse_contigs('A,B').resolve(idx=p['idx'])
    full = spec_to_templates(tspec, p['batch'])
    keep = {'A': [0], 'B': [1], 'both': [0, 1]}[want_t]
    templates = {i: full[i] for i in keep if i in full}
    print('templates on chain(s) %s' % want_t)

  print('MSA %s (num_msa %d, crop %d), %d recycles'
        % ('the job\'s own paired + unpaired' if use_msa else 'OFF (single seq)',
           n_msa, crop, recycles))
  print('%-14s %6s %6s %7s %6s %11s' %
        ('model', 'A', 'B', 'iface', 'fnat', 'pLDDT A/B'))

  for m in (os.environ.get('MODELS').split(',') if os.environ.get('MODELS')
            else list(ALL)):
    try:
      kw = models.featurise_kwargs(m)
      if 'struct_num_tokens' in kw:
        kw['struct_num_tokens'] = int(os.environ.get('STRUCT_TOKENS', '512'))
      if m == 'chai1':
        # chai's token stream is mostly ESM2 -- without it streptavidin goes
        # 0.642 -> 5.70 A -- so a chai row with the ESM path dark would measure
        # nothing. Per-chain embeddings concatenated in token order; chai embeds
        # each chain separately, there is no cross-chain context in ESM.
        esm = np.concatenate([np.load('%s/esm_%s.npz' % (REF_DIR, c))['esm']
                              for c in ('a', 'b')])
        assert len(esm) == la + lb, (len(esm), la + lb)
        # templates go through **kw to featurise_spec; omitting them here would
        # have run chai1 template-free in a template sweep and read as "chai's
        # template path is dead on a complex"
        batch = f3.featurise_chai1(
            parse_contigs('%d/0 %d' % (la, lb)),
            sequences={0: seqs[0], 1: seqs[1]}, msa_crop_size=crop,
            esm=esm, msa=msa if msa is not None else False,
            templates=templates)
      else:
        batch = f3.featurise_spec(
            parse_contigs('%d/0 %d' % (la, lb)).resolve(),
            sequences={0: seqs[0], 1: seqs[1]}, msa_crop_size=crop,
            msa=msa, templates=templates, **kw)
      batch = batch[0] if isinstance(batch, tuple) else batch

      cfg, params, _f, missing = models.build(
          m, batch, num_recycles=recycles, diffusion_steps=200, num_msa=n_msa)
      runner = AF3Runner(cfg=cfg, model_params=params, diffusion='forward',
                         num_msa=n_msa)
      for seed in seeds:
        out = runner.predict(batch, key=jax.random.PRNGKey(seed))
        x = np.asarray(out['diffusion_samples']['atom_positions'])
        while x.ndim > 3:
          x = x[0]
        pl = np.asarray(out['predicted_lddt'])
        if m == 'opendde':
          # opendde diffuses on STRUCTURAL tokens, so both the coordinates and
          # the confidence come back through structbook/residue_atom_gather.
          from colabdesign2.af3 import structural_features as sf
          g = np.asarray(batch['structbook/residue_atom_gather'])
          x = np.asarray(sf.structural_to_residue_positions(x, g))
          d = np.asarray(batch['struct/ref_mask']).shape[1]
          while pl.ndim > 1:
            pl = pl[0]
          pl = np.asarray(sf._gather_struct_feature(pl.reshape(-1, d), g))
        while pl.ndim > 2:
          pl = pl[0]
        ca = x[:la + lb, 1]
        a, b = ca[:la], ca[la:la + lb]
        pa = pl[:la, 1] if pl.ndim == 2 else pl[:la]
        pb = pl[la:la + lb, 1] if pl.ndim == 2 else pl[la:la + lb]
        ua = kabsch(a, ra)
        print('%-14s %6.2f %6.2f %7.2f %6.2f %5.1f/%-5.1f%s'
              % (m, rmsd_after(*kabsch(a, ra), a, ra),
                 rmsd_after(*kabsch(b, rb), b, rb),
                 rmsd_after(ua[0], ua[1], ua[2], b, rb),
                 fnat(a, b, ra, rb), pa.mean(), pb.mean(),
                 '' if not missing else '  (%d at INIT)' % len(missing)),
              flush=True)
    except Exception as e:                                  # noqa: BLE001
      print('%-14s FAILED  %s: %s' % (m, type(e).__name__, str(e)[:80]),
            flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
