# A REAL fold on jax 0.11.1 (unpinned). Import success proves nothing about the
# graph: haiku 0.0.17 reaches into jax internals (hence the DropVar sed) and
# tokamax is pinned, so either could break at trace time rather than import.
import json, os, subprocess, sys, time
V='3.1.7'
os.system(f'wget -q -O run_alphafold.py https://raw.githubusercontent.com/sokrypton/alphafold3/v{V}/run_alphafold.py')
# haiku 0.0.17 still calls the moved jax.core.DropVar
os.system("sed -i 's/jax.core.DropVar/jax.extend.core.DropVar/g' /usr/local/lib/python*/dist-packages/haiku/_src/jaxpr_info.py")
seq='ACSEFGHIKLWYMNPQRSTV'
json.dump({'dialect':'alphafold3','version':4,'name':'t','modelSeeds':[1],
  'sequences':[{'protein':{'id':'A','sequence':seq,'unpairedMsa':f'>q\n{seq}\n',
                           'pairedMsa':'','templates':[]}}]}, open('t.json','w'))
# T4 is compute capability 7.5 and run_alphafold REFUSES to start without this
# (the notebook sets it too); my harness did not, which is why the first attempt
# died after featurisation rather than in it.
env=dict(os.environ, XLA_FLAGS='--xla_disable_hlo_passes=custom-kernel-fusion-rewriter')
t=time.time()
r=subprocess.run([sys.executable,'run_alphafold.py','--json_path=t.json',
  '--model=openbind0','--output_dir=out','--norun_data_pipeline',
  '--num_diffusion_samples=1','--flash_attention_implementation=xla',
  '--buckets=32'], capture_output=True, text=True, env=env)
dt=time.time()-t
print('fold rc=%d in %.0f s' % (r.returncode, dt))
tail=(r.stdout+r.stderr)[-1500:]
print(tail if r.returncode else tail[-400:])
import glob
cifs=glob.glob('out/**/*model.cif', recursive=True)
print('cif files:', len(cifs))
if cifs:
    n=sum(1 for l in open(cifs[0]) if l.startswith('ATOM'))
    print('atoms in first cif:', n)
print('FOLD_RESULT:', 'PASS' if (r.returncode==0 and cifs) else 'FAIL')
