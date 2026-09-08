"""Native ESMFold2 on 6MRR: features, ESM-C hidden states, coordinates, pLDDT.

The dump every ESMFold2 localiser compares against. Run it from the vendor venv
(`~/venv_esm/bin/python`), which is the only one with transformers.

  MODEL=esmfold2_exp ~/venv_esm/bin/python dev/oracles/esmfold2_oracle_6mrr.py

MODEL selects the release and NAMES THE OUTPUT, because every variant has its
own weights and its own shim: handing one variant another's reads corr 0.026
against native and folds 6MRR at 8.8 A where its own gives 0.96. One filename
for all of them is exactly the mistake that invites.

6MRR is 68 residues, not 71 -- three CA records are altloc B, and the dedupe
below is what gets that right.
"""
import os, numpy as np, torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features

MODEL = os.environ.get('MODEL', 'esmfold2')
_HUB = {
    'esmfold2': 'ESMFold2', 'esmfold2_fast': 'ESMFold2-Fast',
    'esmfold2_exp': 'ESMFold2-Experimental',
    'esmfold2_exp_fast': 'ESMFold2-Experimental-Fast',
    'esmfold2_exp_cutoff2025': 'ESMFold2-Experimental-Cutoff2025',
    'esmfold2_exp_fast_cutoff2025': 'ESMFold2-Experimental-Fast-Cutoff2025',
    'esmfold2_lm600m': 'ESMFold2-Experimental-Fast-base600M-step1500k',
    'esmfold2_lm300m': 'ESMFold2-Experimental-Fast-base300M-step1500k',
}
A3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
      'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
      'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
seq, xyz, seen = [], [], set()
for l in open('/home/ubuntu/6MRR.pdb'):
    if not l.startswith('ATOM') or l[12:16].strip() != 'CA': continue
    k = l[21] + l[22:27]
    if k in seen: continue                       # <- the altloc dedupe I was missing
    seen.add(k); seq.append(A3[l[17:20].strip()])
    xyz.append([float(l[30+8*i:38+8*i]) for i in range(3)])
SEQ = ''.join(seq); NAT = np.array(xyz)
print('6MRR: %d residues' % len(SEQ)); print(SEQ)

# The experimental line is a different class; pick it from the config, the same
# way esmfold2_native_variants.py does, rather than assuming the base model.
_local = os.path.expanduser('~/esmfold2_variants/%s' % _HUB[MODEL])
_src = _local if os.path.exists(_local + '/config.json') else 'biohub/%s' % _HUB[MODEL]
_arch = 'ESMFold2Model'
if os.path.exists(_local + '/config.json'):
    import json
    _arch = json.load(open(_local + '/config.json'))['architectures'][0]
if _arch == 'ESMFold2Model':
    m = ESMFold2Model.from_pretrained(_src, esmc_precision="bf16").cuda().eval()
else:
    from transformers.models.esmfold2 import modeling_esmfold2_experimental as _X
    m = getattr(_X, _arch).from_pretrained(_src).cuda().eval()
print('%s -> %s (%s)' % (MODEL, _src, _arch))
feats = {k: v.cuda() for k, v in prepare_protein_features(SEQ).items()}
cap = {}
m.language_model.register_forward_pre_hook(
    lambda mod, i: cap.__setitem__('lm_hidden', i[0].detach().float().cpu().numpy()))
torch.manual_seed(0)
with torch.no_grad():
    o = m(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=200)
d = {'feat.' + k: v.detach().float().cpu().numpy() for k, v in feats.items()}
d['lm_hidden'] = cap['lm_hidden']
d['native_ca'] = NAT
d['out.sample_atom_coords'] = o['sample_atom_coords'].float().cpu().numpy()
# The experimental fast/lm* releases carry a different confidence head and
# return no 'plddt' key at all -- optional, or the dump dies after the forward
# pass it just spent a minute on.
if 'plddt' in o:
    d['out.plddt'] = o['plddt'].float().cpu().numpy()
import os as _os
_D = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'dumps')
_os.makedirs(_D, exist_ok=True)
np.savez_compressed(_os.path.join(_D, 'esmfold2_%s_6mrr68.npz' % MODEL), **d)

ch = feats['ref_atom_name_chars'][0].cpu().numpy().astype(int)
a2t = feats['atom_to_token'][0].cpu().numpy().astype(int)
msk = feats['atom_attention_mask'][0].cpu().numpy().astype(bool)
nm = lambda i: ''.join(chr(c+32) for c in ch[i]).strip()
rep = np.full(int(a2t[msk].max())+1, -1)
for i in range(len(a2t)):
    if msk[i] and nm(i) == 'CA': rep[a2t[i]] = i
np.save(_os.path.join(_D, 'ca_idx68_%s.npy' % MODEL), rep)
def rmsd(a, b):
    n = min(len(a), len(b)); a, b = a[:n]-a[:n].mean(0), b[:n]-b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T@b); s = np.sign(np.linalg.det(u@vt))
    return float(np.sqrt((((a@(u@np.diag([1,1,s])@vt))-b)**2).sum(1).mean()))
x = d['out.sample_atom_coords'][0]
print('NATIVE 6MRR (68 res, CA gather): %.3f A   pLDDT %s'
      % (rmsd(x[rep], NAT),
         '%.3f' % d['out.plddt'].mean() if 'out.plddt' in d else 'n/a'))
