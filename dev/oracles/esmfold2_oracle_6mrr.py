"""Native ESMFold2 on 6MRR: features, ESM-C hidden states, coordinates, pLDDT.

The dump every ESMFold2 localiser compares against. Run it from the vendor venv
(`~/venv_esm/bin/python`), which is the only one with transformers.

  MODEL=esmfold2 ~/venv_esm/bin/python dev/oracles/esmfold2_oracle_6mrr.py

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

# --- module I/O for the two gates that had no reference to compare against.
#
# L2.atom_decoder and L4.confidence were holes for the whole ESMFold2 family for
# one reason: `esmfold2_dumps.native()` reads its inputs from an npz and the dump
# carried the features, the LM hidden states and the final coordinates -- not the
# intermediate module I/O. There is no way to run these two modules on OUR inputs
# from this venv (the graph is JAX and lives in the other one), so the dump is
# where the reference has to come from.
#
# Hooked by NAME on the submodule, with kwargs: both are called with keyword
# arguments only (`self.atom_decoder(a_i=..., q_l=..., ...)`), so a hook without
# `with_kwargs=True` would record an empty `args` and silently dump nothing.
_IO = {}


def _numpy(x):
    import torch as _t
    return x.detach().float().cpu().numpy() if _t.is_tensor(x) else None


def _record(tag, args, kwargs, out):
    for i, v in enumerate(args):
        a = _numpy(v)
        if a is not None:
            _IO['%s.in.%d' % (tag, i)] = a
    for k, v in kwargs.items():
        a = _numpy(v)
        if a is not None:
            _IO['%s.in.%s' % (tag, k)] = a
    outs = out if isinstance(out, (tuple, list)) else [out]
    for i, v in enumerate(outs):
        a = _numpy(v)
        if a is not None:
            _IO['%s.out.%d' % (tag, i)] = a
        elif isinstance(v, dict):
            for k, vv in v.items():
                a = _numpy(vv)
                if a is not None:
                    _IO['%s.out.%s' % (tag, k)] = a


def _hook(tag, suffix):
    # FOUND BY WALKING named_modules(), not by attribute path. The two release
    # lines nest these differently and the obvious guess was wrong: the decoder
    # is not under a `structure_module` -- this model calls it `structure_head`
    # -- and a getattr chain that misses returns None, which would have written
    # a dump with the hole still in it and no error.
    hits = [(n, mo) for n, mo in m.named_modules() if n.split('.')[-1] == suffix]
    if not hits:
        print('  no submodule named %r; the %s gate stays a hole' % (suffix, tag))
        return
    if len(hits) > 1:
        print('  NOTE %d modules named %r: %s -- hooking all, later wins'
              % (len(hits), suffix, [n for n, _ in hits]))
    for name, mod in hits:
        mod.register_forward_hook(
            lambda mo, a, kw, o: _record(tag, a, kw, o), with_kwargs=True)
        print('  hooked %s at %s (%s)' % (tag, name, type(mod).__name__))


_hook('dec', 'atom_decoder')
_hook('conf', 'confidence_head')
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
# The hooks fire once per sampling step for the decoder, so what lands in the
# dump is the LAST step -- stated because a reader comparing against it needs to
# know which step's inputs they hold, and the inputs are dumped alongside the
# output precisely so the comparison does not depend on reproducing the step.
d.update(_IO)
print('module I/O captured: %s' % sorted(_IO))
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
