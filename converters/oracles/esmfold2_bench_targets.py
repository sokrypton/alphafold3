"""Build features + native CA + a native-reference fold for a benchmark set."""
import numpy as np, torch, json
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features
A3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
      'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
      'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def parse_pdb(path, chain):
    out, seen = [], set()
    for l in open(path):
        if not l.startswith('ATOM') or l[12:16].strip() != 'CA': continue
        if l[21] != chain: continue
        k = l[22:27]
        if k in seen: continue                       # altloc dedupe
        seen.add(k)
        r = l[17:20].strip()
        if r not in A3: continue
        out.append((int(l[22:26]), A3[r], [float(l[30+8*i:38+8*i]) for i in range(3)]))
    return out

TARGETS = [("1CRN", 'A', "natural"), ("1ENH", 'A', "natural"), ("1PGB", 'A', "natural"),
           ("1SHG", 'A', "natural"), ("2CI2", "I", "natural"), ("1UBQ", 'A', "natural"),
           ("3CHY", 'A', "natural"), ("1MBA", 'A', "natural"), ("2LZM", 'A', "natural"),
           ("6MRR", None, "DESIGNED"), ("1QYS", 'A', "DESIGNED")]
PATHS = {"6MRR": "/home/ubuntu/6MRR.pdb", "1QYS": "/home/ubuntu/1QYS.pdb", "1UBQ": "1ubq.pdb"}

m = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16").cuda().eval()
def rmsd(a, b):
    n = min(len(a), len(b)); a, b = a[:n]-a[:n].mean(0), b[:n]-b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T@b); s = np.sign(np.linalg.det(u@vt))
    return float(np.sqrt((((a@(u@np.diag([1,1,s])@vt))-b)**2).sum(1).mean()))

meta = {}
store = {}
print("%-8s %-10s %-5s %-9s %s" % ("target", "kind", "len", "native", "pLDDT"))
for name, chain, kind in TARGETS:
    res = parse_pdb(PATHS.get(name, "%s.pdb" % name), chain or 'A')
    seq = ''.join(c for _, c, _ in res)
    nat = np.array([x for _, _, x in res])
    feats = {k: v.cuda() for k, v in prepare_protein_features(seq).items()}
    ch = feats['ref_atom_name_chars'][0].cpu().numpy().astype(int)
    a2t = feats['atom_to_token'][0].cpu().numpy().astype(int)
    msk = feats['atom_attention_mask'][0].cpu().numpy().astype(bool)
    nm = lambda i: ''.join(chr(c+32) for c in ch[i]).strip()
    rep = np.full(int(a2t[msk].max())+1, -1)
    for i in range(len(a2t)):
        if msk[i] and nm(i) == 'CA': rep[a2t[i]] = i
    assert (rep >= 0).all() and len(rep) == len(seq), (name, len(rep), len(seq))
    torch.manual_seed(0)
    with torch.no_grad():
        o = m(**feats, num_loops=3, num_diffusion_samples=1, num_sampling_steps=200)
    x = o['sample_atom_coords'].float().cpu().numpy()
    if x.ndim == 3: x = x[0]
    r = rmsd(x[rep], nat)
    for k, v in feats.items(): store['%s.feat.%s' % (name, k)] = v.detach().float().cpu().numpy()
    store['%s.native_ca' % name] = nat
    store['%s.rep' % name] = rep
    meta[name] = dict(kind=kind, n=len(seq), native_rmsd=r, plddt=float(o['plddt'].mean()))
    print("%-8s %-10s %-5d %-9.3f %.3f" % (name, kind, len(seq), r, meta[name]['plddt']))
np.savez_compressed('bench_targets.npz', **store)
json.dump(meta, open('bench_meta.json', 'w'), indent=1)
