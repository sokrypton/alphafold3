"""Capture diffusion conditioning I/O in fp32."""
import numpy as np, torch, math, contextlib, torch.amp
torch.amp.autocast = lambda *a, **k: contextlib.nullcontext()
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features
SEQ = "GWSTELEKHREELKKEFLKKEGITNVEIRIDNGRLEVRVEGGTEERLKRFLEELRQKLEKKGYTVDDIKIE"
m = ESMFold2Model.from_pretrained("biohub/ESMFold2", load_esmc=False).cuda().eval()
m.config.lm_encoder.per_loop_lm_dropout = False; m.config.lm_encoder.lm_dropout = 0.0
r = dict(np.load('esmfold2_nomsa.npz'))
feats = {k: v.cuda() for k, v in prepare_protein_features(SEQ).items()}
for k in ('msa','msa_attention_mask','has_deletion','deletion_value','input_ids'): feats.pop(k, None)

cap = {}
cond = m.structure_head.diffusion_module.conditioning
def pre(mod, args, kwargs):
    if 'in_t' in cap: return                       # first call only
    for k, v in kwargs.items():
        if torch.is_tensor(v): cap['in_' + k] = v.detach().clone().float().cpu().numpy()
        elif isinstance(v, float): cap['in_' + k] = np.array([v], np.float32)
    cap['in_t'] = np.array([1.0])
def post(mod, inp, out):
    if 'out_s' in cap: return
    cap['out_s'] = out[0].detach().clone().float().cpu().numpy()
    cap['out_z'] = out[1].detach().clone().float().cpu().numpy()
hs = [cond.register_forward_pre_hook(pre, with_kwargs=True), cond.register_forward_hook(post)]
def seeded(ref):
    g = torch.Generator().manual_seed(1234); std = math.sqrt(2.0/(5.0*ref.shape[-1]))
    s = torch.empty(ref.shape, dtype=torch.float32)
    torch.nn.init.trunc_normal_(s, 0.0, std, -3*std, 3*std, generator=g)
    return s.to(device=ref.device, dtype=ref.dtype)
m._init_pair_state = seeded
with torch.no_grad():
    m(**feats, lm_hidden_states=torch.tensor(r['lm_z.in']).cuda(), num_loops=3,
      num_diffusion_samples=1, num_sampling_steps=14)
for h in hs: h.remove()
np.savez_compressed('esmfold2_cond.npz', **cap)
for k in sorted(cap): print("  %-34s %s" % (k, cap[k].shape))
