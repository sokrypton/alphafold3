"""KNOWN-ANSWER gate for AF2 templates: fold a target given its OWN structure.

A template the model actually reads pulls the fold onto it, so a self-template
is a test with a known answer and no native oracle needed. It exists because
templates failed here in three different SILENT ways, none of which a plain fold
would have shown:

  * the fold input's template was accepted and then dropped (the features were
    all zeros);
  * `use_templates` was never threaded through, so populated features reached a
    graph with the template embedder disabled AND its weights stripped;
  * the residue alphabet. `template_aatype` is written by the hhsearch pipeline
    in HHBLITS order and converted to our restype order by AF2's own
    `fix_templates_aatype` before the model. Feeding the raw order is not
    catastrophic -- 1.48 A against 0.26 A here -- so it looks like a slightly
    worse model rather than a bug.

  PYTHONPATH=src:.:dev/oracles python dev/oracles/af2_template_check.py af2_ptm
  MODE=none  ... (baseline)   MODE=hhblits ... (the wrong alphabet, for contrast)

ONE FOLD PER PROCESS: AF2 is bimodal across processes here because XLA autotunes
by timing (see af2_fold_check.py), so each configuration gets its own run.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modality_check as mc                                  # noqa: E402
import template_parity as tp                                 # noqa: E402

# best-of-one self-template targets, per model, measured 2026-09-13
EXPECTED = {'af2_ptm': 0.5, 'af2_multimer': 1.8}


def main(argv=None):
  argv = sys.argv[1:] if argv is None else argv
  model = argv[0] if argv else 'af2_ptm'
  cif = os.path.expanduser(argv[1] if len(argv) > 1 else '~/5K9P.cif')
  mode = os.environ.get('MODE', 'restype')

  from alphafold3.common import folding_input
  from alphafold3.model import model_registry
  from alphafold3.af2 import features as af2_features
  from alphafold3.af2 import inference as af2_inference
  from alphafold3.constants import decoded_ccd
  from alphafold3.data import featurisation
  import jax

  seq, tmpl = tp._self_template(cif, 'A')                    # noqa: SLF001
  ch = folding_input.ProteinChain(
      id='A', sequence=seq, ptms=[], unpaired_msa='', paired_msa='',
      templates=([] if mode == 'none' else [tmpl]))
  fold_input = folding_input.Input(name='tmpl', chains=[ch], rng_seeds=[0])
  if mode != 'none':
    os.environ['AF2_TEMPLATE_ALPHABET'] = mode

  runner = af2_inference.AF2ModelRunner(
      model_registry.get(model), None, os.path.expanduser('~/params'),
      num_recycles=int(os.environ.get('RECYCLES', 3)), use_bfloat16=False,
      use_templates=(mode != 'none'))
  af2_features.protein_chains(fold_input)
  batch = featurisation.featurise_input(
      fold_input=fold_input, ccd=decoded_ccd.get_ccd(), buckets=None)[0]
  result = runner.run_inference(batch, jax.random.PRNGKey(0))
  ca = np.asarray(result['structure_module']['final_atom_positions'])[:, 1, :]

  # ALIGN BY label_seq_id. A cif carries alt locs and can carry rows the query
  # does not, so taking the first N in file order compares different residues --
  # it read 6.4 A for a template that is 0.01 A from native.
  rows = [r for r in mc.read_cif_atoms(cif)
          if r.get('label_atom_id') == 'CA' and r.get('label_asym_id') == 'A'
          and r.get('label_alt_id') in ('.', '?', '', 'A')]
  by_id = {int(r['label_seq_id']): [float(r['Cartn_x']), float(r['Cartn_y']),
                                    float(r['Cartn_z'])] for r in rows}
  nat = np.array([by_id[i] for i in sorted(by_id)[:len(ca)]])
  n = min(len(nat), len(ca))
  rmsd = mc.kabsch(ca[:n], nat[:n])[0]
  print('%s templates=%s: CA-RMSD %.3f over %d CA' % (model, mode, rmsd, n))
  if mode == 'none':
    return 0
  want = EXPECTED.get(model)
  if want is not None and rmsd > want:
    print('  FAIL: a self-template should reach %.1f A; the template is not '
          'reaching the model, or the alphabet is wrong' % want)
    return 1
  print('  OK')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
