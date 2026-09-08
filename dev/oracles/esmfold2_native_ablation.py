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
r = dict(np.load('esmfold2_nomsa.npz'))
lmh = torch.tensor(r['lm_z.in']).cuda()

m = ESMFold2Model.from_pretrained("biohub/ESMFold2", load_esmc=False).cuda().eval()
feats = {k: v.cuda() for k, v in prepare_protein_features(SEQ).items()}
rep = feats['distogram_atom_idx'][0].cpu().numpy().astype(int)
base = dict(feats); base.pop('input_ids', None)
nomsa = {k: v for k, v in base.items() if k not in ('msa','msa_attention_mask','has_deletion','deletion_value')}

print("%-46s %s" % ("native config (ESM-C replayed)", "CA-RMSD"))
for tag, kw, dropout in [
    ("with MSA feats,  lm_dropout ON  (as shipped)", base,  True),
    ("no  MSA feats,   lm_dropout ON",               nomsa, True),
    ("no  MSA feats,   lm_dropout OFF",              nomsa, False)]:
    m.config.lm_encoder.per_loop_lm_dropout = dropout
    m.config.lm_encoder.lm_dropout = 0.25 if dropout else 0.0
    outs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        with torch.no_grad():
            o = m(**kw, lm_hidden_states=lmh, num_loops=3,
                  num_diffusion_samples=1, num_sampling_steps=14)
        outs.append(rmsd(o['sample_atom_coords'][0].float().cpu().numpy()[rep], nat))
    print("  %-44s %s" % (tag, "  ".join("%.3f" % v for v in outs)))
