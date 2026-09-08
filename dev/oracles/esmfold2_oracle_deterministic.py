"""Deterministic ESMFold2 reference: per-loop LM dropout OFF, ESM-C replayed."""
import numpy as np, torch, math
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features
SEQ = "GWSTELEKHREELKKEFLKKEGITNVEIRIDNGRLEVRVEGGTEERLKRFLEELRQKLEKKGYTVDDIKIE"

m = ESMFold2Model.from_pretrained("biohub/ESMFold2", load_esmc=False).cuda().eval()
m.config.lm_encoder.per_loop_lm_dropout = False       # <-- the 25% inference dropout
m.config.lm_encoder.lm_dropout = 0.0
print("per_loop_lm_dropout ->", m.config.lm_encoder.per_loop_lm_dropout)

r = dict(np.load('esmfold2_nomsa.npz'))
feats = {k: v.cuda() for k, v in prepare_protein_features(SEQ).items()}
for k in ('msa', 'msa_attention_mask', 'has_deletion', 'deletion_value', 'input_ids'):
    feats.pop(k, None)
lmh = torch.tensor(r['lm_z.in']).cuda()

pre, post = {}, {}
def ph(n):
    def h(mod, inp): pre.setdefault(n, []).append(inp[0].detach().clone().float().cpu().numpy())
    return h
def qh(n):
    def h(mod, inp, out): post.setdefault(n, []).append(out.detach().clone().float().cpu().numpy())
    return h
hs = [m.lm_encoder.register_forward_pre_hook(ph('lmenc_in')),
      m.folding_trunk.register_forward_pre_hook(ph('trunk_in')),
      m.parcae_input_norm.register_forward_pre_hook(ph('pnorm_in')),
      m.parcae_coda.register_forward_pre_hook(ph('coda_in')),
      m.lm_encoder.register_forward_hook(qh('lmenc')),
      m.folding_trunk.register_forward_hook(qh('trunk')),
      m.parcae_input_norm.register_forward_hook(qh('pnorm')),
      m.parcae_coda.register_forward_hook(qh('coda')),
      m.parcae_readout.register_forward_hook(qh('readout')),
      m.language_model.register_forward_hook(qh('lm_z')),
      m.inputs_embedder.register_forward_hook(qh('x_inputs')),
      m.rel_pos.register_forward_hook(qh('rel_pos')),
      m.token_bonds.register_forward_hook(qh('token_bonds'))]

init_state = {}
def seeded(ref):
    g = torch.Generator().manual_seed(1234)
    std = math.sqrt(2.0/(5.0*ref.shape[-1]))
    s = torch.empty(ref.shape, dtype=torch.float32)
    torch.nn.init.trunc_normal_(s, 0.0, std, -3*std, 3*std, generator=g)
    init_state['s'] = s.numpy()
    return s.to(device=ref.device, dtype=ref.dtype)
m._init_pair_state = seeded

with torch.no_grad():
    out = m(**feats, lm_hidden_states=lmh, num_loops=3,
            num_diffusion_samples=1, num_sampling_steps=14)
for h in hs: h.remove()

d = {'parcae.init_state': init_state['s'],
     'lm_hidden': r['lm_z.in'],
     'out.distogram_logits': out['distogram_logits'].float().cpu().numpy(),
     'out.sample_atom_coords': out['sample_atom_coords'].float().cpu().numpy(),
     'out.plddt': out['plddt'].float().cpu().numpy()}
for k, vs in pre.items():
    for i, v in enumerate(vs): d['pre_%s.%d' % (k, i)] = v
for k, vs in post.items():
    for i, v in enumerate(vs): d['%s.%d' % (k, i)] = v
np.savez_compressed('esmfold2_det.npz', **d)
b = pre['lmenc_in'][0]
print("lm_encoder inputs now loop-invariant?", [float(np.abs(a-b).max()) for a in pre['lmenc_in']])
print("dumped", len(d), "tensors; ptm %.3f" % float(out['ptm'][0]))
