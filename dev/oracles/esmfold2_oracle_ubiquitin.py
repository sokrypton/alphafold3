"""Correct-sequence oracle: 6MRR is 68 residues (3 CA records are altloc B)."""
import numpy as np, torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features
A3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
      'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
      'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
seq, xyz, seen = [], [], set()
for l in open('1ubq.pdb'):
    if not l.startswith('ATOM') or l[12:16].strip() != 'CA' or l[21] != 'A': continue
    k = l[21] + l[22:27]
    if k in seen: continue                       # <- the altloc dedupe I was missing
    seen.add(k); seq.append(A3[l[17:20].strip()])
    xyz.append([float(l[30+8*i:38+8*i]) for i in range(3)])
SEQ = ''.join(seq); NAT = np.array(xyz)
print('1UBQ: %d residues' % len(SEQ)); print(SEQ)

m = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16").cuda().eval()
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
d['out.plddt'] = o['plddt'].float().cpu().numpy()
np.savez_compressed('esmfold2_1ubq.npz', **d)

ch = feats['ref_atom_name_chars'][0].cpu().numpy().astype(int)
a2t = feats['atom_to_token'][0].cpu().numpy().astype(int)
msk = feats['atom_attention_mask'][0].cpu().numpy().astype(bool)
nm = lambda i: ''.join(chr(c+32) for c in ch[i]).strip()
rep = np.full(int(a2t[msk].max())+1, -1)
for i in range(len(a2t)):
    if msk[i] and nm(i) == 'CA': rep[a2t[i]] = i
np.save('ca_idx_ubq.npy', rep)
def rmsd(a, b):
    n = min(len(a), len(b)); a, b = a[:n]-a[:n].mean(0), b[:n]-b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T@b); s = np.sign(np.linalg.det(u@vt))
    return float(np.sqrt((((a@(u@np.diag([1,1,s])@vt))-b)**2).sum(1).mean()))
x = d['out.sample_atom_coords'][0]
print('NATIVE 1UBQ (68 res, CA gather): %.3f A   pLDDT %.3f' % (rmsd(x[rep], NAT), d['out.plddt'].mean()))
