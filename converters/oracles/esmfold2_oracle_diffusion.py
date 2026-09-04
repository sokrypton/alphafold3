"""Capture one full DiffusionModule denoise step in fp32."""
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

cap, dm = {}, m.structure_head.diffusion_module
def pre(mod, args, kwargs):
    if 'done' in cap: return
    for k, v in kwargs.items():
        if torch.is_tensor(v): cap['in_' + k] = v.detach().clone().float().cpu().numpy()
        elif isinstance(v, (int, float)): cap['in_' + k] = np.array([v], np.float64)
def post(mod, inp, out):
    if 'done' in cap: return
    cap['out_x_denoised'] = out['x_denoised'].detach().clone().float().cpu().numpy()
    cap['done'] = np.array([1])
def dec_post(mod, inp, out):
    if 'r_update' in cap: return
    cap['r_update'] = out[0].detach().clone().float().cpu().numpy()
def enc_post(mod, inp, out):
    if 'enc_a' in cap: return
    cap['enc_a'] = out[0].detach().clone().float().cpu().numpy()
    cap['enc_q'] = out[1].detach().clone().float().cpu().numpy()
    cap['enc_c'] = out[2].detach().clone().float().cpu().numpy()
def tt_post(mod, inp, out):
    if 'tok_out' in cap: return
    cap['tok_out'] = out[0].detach().clone().float().cpu().numpy()
hs = [dm.register_forward_pre_hook(pre, with_kwargs=True), dm.register_forward_hook(post),
      dm.atom_decoder.register_forward_hook(dec_post),
      dm.atom_encoder.register_forward_hook(enc_post),
      dm.token_transformer.register_forward_hook(tt_post)]
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
np.savez_compressed('esmfold2_diff.npz', **cap)
for k in sorted(cap): print("  %-34s %s" % (k, cap[k].shape))
