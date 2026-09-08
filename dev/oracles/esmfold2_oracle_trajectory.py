"""Capture the FULL sampling trajectory: every step's t_hat, x_noisy, x_denoised, x."""
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

traj, dm = [], m.structure_head.diffusion_module
cur = {}
def pre(mod, args, kwargs):
    cur['x_noisy'] = kwargs['x_noisy'].detach().clone().float().cpu().numpy()
    cur['t_hat'] = float(kwargs['t_hat'].reshape(-1)[0].item())
def post(mod, inp, out):
    cur['x_denoised'] = out['x_denoised'].detach().clone().float().cpu().numpy()
    traj.append(dict(cur))
hs = [dm.register_forward_pre_hook(pre, with_kwargs=True), dm.register_forward_hook(post)]
cond = {}
def cpre(mod, args, kwargs):
    if 'z' in cond: return
    for k in ('s_inputs','z_trunk','relative_position_encoding'):
        if k in kwargs: cond[k] = kwargs[k].detach().clone().float().cpu().numpy()
    cond['z'] = 1
hs.append(dm.conditioning.register_forward_pre_hook(cpre, with_kwargs=True))
def seeded(ref):
    g = torch.Generator().manual_seed(1234); std = math.sqrt(2.0/(5.0*ref.shape[-1]))
    s = torch.empty(ref.shape, dtype=torch.float32)
    torch.nn.init.trunc_normal_(s, 0.0, std, -3*std, 3*std, generator=g)
    return s.to(device=ref.device, dtype=ref.dtype)
m._init_pair_state = seeded
torch.manual_seed(0)
with torch.no_grad():
    out = m(**feats, lm_hidden_states=torch.tensor(r['lm_z.in']).cuda(), num_loops=3,
            num_diffusion_samples=1, num_sampling_steps=14)
for h in hs: h.remove()
d = {'final': out['sample_atom_coords'].float().cpu().numpy()}
for i, s in enumerate(traj):
    d['t_hat.%d' % i] = np.array([s['t_hat']])
    d['x_noisy.%d' % i] = s['x_noisy']
    d['x_denoised.%d' % i] = s['x_denoised']
for k in ('s_inputs','z_trunk','relative_position_encoding'): d['cond_'+k] = cond[k]
np.savez_compressed('esmfold2_traj.npz', **d)
print("steps:", len(traj))
print("t_hat schedule:", [round(s['t_hat'], 4) for s in traj])
