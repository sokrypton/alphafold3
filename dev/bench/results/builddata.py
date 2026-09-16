import subprocess, sys, time, glob, os
import importlib.metadata as md
pkg = os.path.dirname(md.distribution('alphafold3-colabfold').locate_file('alphafold3'))
conv = os.path.join(pkg, 'alphafold3', 'constants', 'converters')
print('pickles before:', [os.path.basename(f) for f in glob.glob(conv + '/*.pickle')] or 'NONE')
t = time.time()
r = subprocess.run(['build_data'], capture_output=True, text=True)
print('build_data rc=%d in %.0f s' % (r.returncode, time.time() - t))
print((r.stdout + r.stderr)[-500:])
print('pickles after:', [(os.path.basename(f), os.path.getsize(f) >> 20) for f in glob.glob(conv + '/*.pickle')])
