import importlib.metadata as md, shutil, subprocess, sys, os
print('python', sys.version.split()[0])
for p in ('jax','jaxlib','rdkit','zstandard','tokamax','dm-haiku','ml_collections',
          'dm-tree','py3Dmol','py2Dmol','awscli','numpy','scipy','absl-py'):
    try: print('  PREINSTALLED %-16s %s' % (p, md.version(p)))
    except Exception: print('  absent       %-16s' % p)
print('nvidia cuda pip pkgs:', len([d for d in md.distributions()
      if (d.metadata['Name'] or '').startswith('nvidia-')]))
print('GPU:', subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
      capture_output=True, text=True).stdout.strip())
print('disk free:', shutil.disk_usage('/content').free >> 30, 'GiB')
