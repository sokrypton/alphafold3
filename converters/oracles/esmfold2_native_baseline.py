import numpy as np, torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features
SEQ = "GWSTELEKHREELKKEFLKKEGITNVEIRIDNGRLEVRVEGGTEERLKRFLEELRQKLEKKGYTVDDIKIE"
def ca(p): return np.array([[float(l[30+8*i:38+8*i]) for i in range(3)] for l in open(p) if l.startswith('ATOM') and l[12:16].strip()=='CA'])
def rmsd(a,b):
    n=min(len(a),len(b)); a,b=a[:n]-a[:n].mean(0),b[:n]-b[:n].mean(0)
    u,_,vt=np.linalg.svd(a.T@b); d=np.sign(np.linalg.det(u@vt))
    return float(np.sqrt((((a@(u@np.diag([1,1,d])@vt))-b)**2).sum(1).mean()))
nat = ca('/home/ubuntu/6MRR.pdb')
lmh = torch.tensor(np.load('esmfold2_nomsa.npz')['lm_z.in']).cuda()

m = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16").cuda().eval()
feats = {k: v.cuda() for k, v in prepare_protein_features(SEQ).items()}
rep = feats['distogram_atom_idx'][0].cpu().numpy().astype(int)
noids = {k: v for k, v in feats.items() if k != 'input_ids'}

print("%-34s %s" % ("native config", "6MRR CA-RMSD (2 seeds)"))
for tag, kw, steps in [
    ("ESM-C internal, 200 steps", feats, 200),
    ("ESM-C internal,  14 steps", feats, 14),
    ("hidden replayed, 200 steps", None, 200),
    ("hidden replayed,  14 steps", None, 14)]:
    out = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        with torch.no_grad():
            o = m(**(kw if kw is not None else noids),
                  **({} if kw is not None else {'lm_hidden_states': lmh}),
                  num_loops=3, num_diffusion_samples=1, num_sampling_steps=steps)
        out.append(rmsd(o['sample_atom_coords'][0].float().cpu().numpy()[rep], nat))
    print("  %-32s %s" % (tag, "  ".join("%.3f" % v for v in out)))
