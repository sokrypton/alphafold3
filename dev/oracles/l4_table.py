"""Turn dev/oracles/l4_all.sh output into the PARITY.md table rows.

Transcribing four numbers x six models by hand is how a wrong figure gets into
a document that is then cited as evidence.

  python dev/oracles/l4_table.py /tmp/l4.txt
"""
import re
import sys

rows = {}
model = None
for line in open(sys.argv[1]):
  if line.startswith('== '):
    model = line.split()[1]
    rows[model] = {}
  m = re.match(r'\s+(full_pae|full_pde|plddt|resolved)\s+corr (\S+)', line)
  if m and model:
    rows[model][m.group(1)] = m.group(2)

print('| model | `full_pae` | `full_pde` | `plddt` | `resolved` |')
print('|---|---|---|---|---|')
for k, v in rows.items():
  print('| `%s` | %s | %s | %s | %s |'
        % (k, v.get('full_pae', '-'), v.get('full_pde', '-'),
           v.get('plddt', '-'), v.get('resolved', '-')))
