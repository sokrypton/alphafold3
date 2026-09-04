"""Deterministic fp32 with-MSA reference: LM dropout off, no column mask, no subsample."""
import numpy as np, torch, math, contextlib, torch.amp
torch.amp.autocast = lambda *a, **k: contextlib.nullcontext()      # force fp32
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features
SEQ = "GWSTELEKHREELKKEFLKKEGITNVEIRIDNGRLEVRVEGGTEERLKRFLEELRQKLEKKGYTVDDIKIE"

m = ESMFold2Model.from_pretrained("biohub/ESMFold2", load_esmc=False).cuda().eval()
m.config.lm_encoder.per_loop_lm_dropout = False
m.config.lm_encoder.lm_dropout = 0.0
r = dict(np.load('esmfold2_nomsa.npz'))
feats = {k: v.cuda() for k, v in prepare_protein_features(SEQ).items()}
feats.pop('input_ids', None)
lmh = torch.tensor(r['lm_z.in']).cuda()

cap = {}
def pre_kw(mod, args, kwargs):
    for k, v in kwargs.items():
        if torch.is_tensor(v): cap.setdefault('in_' + k, []).append(v.detach().clone().float().cpu().numpy())
def post(mod, inp, out):
    cap.setdefault('out', []).append(out.detach().clone().float().cpu().numpy())
hs = [m.msa_encoder.register_forward_pre_hook(pre_kw, with_kwargs=True),
      m.msa_encoder.register_forward_hook(post)]

def seeded(ref):
    g = torch.Generator().manual_seed(1234)
    std = math.sqrt(2.0/(5.0*ref.shape[-1]))
    s = torch.empty(ref.shape, dtype=torch.float32)
    torch.nn.init.trunc_normal_(s, 0.0, std, -3*std, 3*std, generator=g)
    cap['init'] = [s.numpy()]
    return s.to(device=ref.device, dtype=ref.dtype)
m._init_pair_state = seeded

with torch.no_grad():
    out = m(**feats, lm_hidden_states=lmh, num_loops=3,
            num_diffusion_samples=1, num_sampling_steps=14,
            msa_column_mask_rate=0.0, msa_subsample_at_inference=False)
for h in hs: h.remove()
d = {'%s.%d' % (k, i): v for k, vs in cap.items() for i, v in enumerate(vs)}
d['out.distogram_logits'] = out['distogram_logits'].float().cpu().numpy()
np.savez_compressed('esmfold2_msa.npz', **d)
for k in sorted(d): print("  %-26s %s" % (k, d[k].shape))
