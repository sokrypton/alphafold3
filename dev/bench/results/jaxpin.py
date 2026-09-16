# Does the jax pin still earn its cost? Colab ships 0.11.1; the notebook pins
# 0.10.1, which downgrades jax AND re-pulls the CUDA stack. Time it, then prove
# the library imports and folds on whatever we end up with.
import subprocess, sys, time, importlib.metadata as md
def sh(c):
    t = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return time.time() - t, r.returncode, (r.stdout + r.stderr)[-600:]

print('base jax:', md.version('jax'))
t, rc, out = sh("pip install -q dm-haiku==0.0.17 rdkit==2025.9.4 tokamax==0.0.11 ml_collections")
print('deps WITHOUT touching jax: %.1f s rc=%d' % (t, rc))
t, rc, out = sh("pip install -q git+https://github.com/sokrypton/py2Dmol.git")
print('py2Dmol from git:          %.1f s rc=%d' % (t, rc))
if rc: print(out)
V='3.1.7'
whl=(f"https://github.com/sokrypton/alphafold3/releases/download/v{V}/"
     f"alphafold3_colabfold-{V}%2Bdata-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
t, rc, out = sh(f"pip install -q --no-deps '{whl}'")
print('fat wheel:                 %.1f s rc=%d' % (t, rc))
print('jax now:', md.version('jax'))
r = subprocess.run([sys.executable,'-c',
  'from alphafold3.common import folding_input;'
  'from alphafold3.model import model_registry as R;'
  'import jax; print("jax", jax.__version__, "devices", jax.devices());'
  'print("IMPORT OK", len(R.MODEL_SPECS))'], capture_output=True, text=True)
print(r.stdout); print(r.stderr[-1200:] if r.returncode else '')
print('UNPINNED_RESULT:', 'PASS' if r.returncode == 0 else 'FAIL rc=%d' % r.returncode)
