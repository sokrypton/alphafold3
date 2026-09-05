"""Does native ESMFold2 fold anything well?  Natural vs designed targets."""
import numpy as np, torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features
A3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
      'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
      'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def parse(path, chain=None):
    seq, xyz, seen = [], [], set()
    for l in open(path):
        if not l.startswith('ATOM') or l[12:16].strip() != 'CA': continue
        if chain and l[21] != chain: continue
        key = l[21] + l[22:27]
        if key in seen: continue
        seen.add(key)
        seq.append(A3.get(l[17:20].strip(), 'X'))
        xyz.append([float(l[30+8*i:38+8*i]) for i in range(3)])
    return ''.join(seq), np.array(xyz)
def rmsd(a, b):
    n = min(len(a), len(b)); a, b = a[:n]-a[:n].mean(0), b[:n]-b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T @ b); d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt((((a @ (u @ np.diag([1,1,d]) @ vt)) - b) ** 2).sum(1).mean()))

m = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16").cuda().eval()
print("%-28s %-5s %-8s %-8s" % ("target", "len", "CA-RMSD", "pLDDT"))
for tag, path, chain in [("1UBQ ubiquitin (natural)", "1ubq.pdb", 'A'),
                         ("6MRR Foldit1 (DESIGNED)", "/home/ubuntu/6MRR.pdb", None),
                         ("1QYS Top7 (DESIGNED)", "/home/ubuntu/1QYS.pdb", 'A')]:
    seq, nat = parse(path, chain)
    feats = {k: v.cuda() for k, v in prepare_protein_features(seq).items()}
    ch = feats['ref_atom_name_chars'][0].cpu().numpy().astype(int)
    a2t = feats['atom_to_token'][0].cpu().numpy().astype(int)
    msk = feats['atom_attention_mask'][0].cpu().numpy().astype(bool)
    nm = lambda i: ''.join(chr(c+32) for c in ch[i]).strip()
    rep = np.full(int(a2t[msk].max())+1, -1)
    for i in range(len(a2t)):
        if msk[i] and nm(i) == 'CA': rep[a2t[i]] = i
    torch.manual_seed(0)
    with torch.no_grad():
        o = m(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=200)
    x = o['sample_atom_coords'].float().cpu().numpy()
    if x.ndim == 3: x = x[0]
    print("%-28s %-5d %-8.3f %-8.3f" % (tag, len(seq), rmsd(x[rep], nat),
                                        float(o['plddt'].mean())))
