"""Build the Colab end-to-end test notebook FROM the notebook we ship.

The test used to be a hand-kept copy of the install cell, which meant it
drifted the moment the real one changed -- and a test of last week's notebook
is not a test. This takes cells 2/3/4 of `ColabFold2_preview.ipynb` verbatim
(install, input, run) and appends one cell that asserts on the result, so the
thing under test is the thing users get.

    python dev/bench/results/make_e2e.py
    colab --auth adc exec -f dev/bench/results/e2e.ipynb --timeout 1400

Form fields are overridden through AF3_NB_OVERRIDES, which the install cell
reads; see the headless-overrides block there. The defaults here keep the run
small and off the MSA server so a failure is ours.
"""

import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OVERRIDES = {
    'model': os.environ.get('E2E_MODEL', 'openbind0'),
    'protein': 'PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK',
    'msa_mode': 'single_sequence',
    'num_diffusion_samples': 1,
    'num_recycles': 3,
    'jobname': 'e2e',
}

CHECK = '''
import glob, json, os
cifs = sorted(glob.glob(f'{job_dir}/**/*.cif', recursive=True))
atoms = sum(1 for l in open(cifs[0]) if l.startswith('ATOM')) if cifs else 0
print('CIFS:', len(cifs), 'ATOMS:', atoms)
conf = sorted(glob.glob(f'{job_dir}/**/*summary_confidences.json', recursive=True))
if conf:
  print('PLDDT/PTM:', {k: v for k, v in json.load(open(conf[0])).items()
                       if k in ('ptm', 'iptm', 'fraction_disordered')})
print('RESULT:', 'PASS' if (cifs and atoms > 100) else 'FAIL')
'''


def main():
  nb = json.loads((ROOT / 'ColabFold2_preview.ipynb').read_text())
  setenv = ('import os\n'
            f'os.environ["AF3_NB_OVERRIDES"] = {json.dumps(json.dumps(OVERRIDES))}\n'
            'print("overrides:", os.environ["AF3_NB_OVERRIDES"])\n')
  cells = [setenv] + [''.join(nb['cells'][i]['source']) for i in (2, 3, 4)] + [CHECK]
  for src in cells:
    assert '\n%%time' not in src, 'a cell magic after #@title aborts under colab-cli'
  out = {'cells': [{'cell_type': 'code', 'metadata': {}, 'source': s.splitlines(keepends=True),
                    'outputs': [], 'execution_count': None} for s in cells],
         'metadata': nb['metadata'], 'nbformat': 4, 'nbformat_minor': 0}
  dst = ROOT / 'dev/bench/results/e2e.ipynb'
  dst.write_text(json.dumps(out, indent=1) + '\n')
  print(f'wrote {dst} from the shipped notebook ({len(cells)} cells, '
        f'model={OVERRIDES["model"]})')


if __name__ == '__main__':
  main()
