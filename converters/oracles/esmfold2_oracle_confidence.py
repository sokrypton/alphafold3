"""Capture confidence-head I/O in fp32."""
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
def pre(mod, args, kwargs):
    for k, v in kwargs.items():
        if torch.is_tensor(v): cap['in_' + k] = v.detach().clone().float().cpu().numpy()
def post(mod, inp, out):
    for k, v in out.items():
        if torch.is_tensor(v): cap['out_' + k] = v.detach().clone().float().cpu().numpy()
hs = [m.confidence_head.register_forward_pre_hook(pre, with_kwargs=True),
      m.confidence_head.register_forward_hook(post)]
# also tap the head's internal pair, post-trunk
inner = {}
# NB: a pre-hook that RETURNS something replaces the module's args -- returning
# the numpy array here fed an ndarray straight into FoldingTrunk.  Return None.
def _tin(mod, i):
    inner.setdefault('trunk_in', i[0].detach().clone().float().cpu().numpy())
def _tout(mod, i, o):
    inner.setdefault('trunk_out', o.detach().clone().float().cpu().numpy())
m.confidence_head.folding_trunk.register_forward_pre_hook(_tin)
m.confidence_head.folding_trunk.register_forward_hook(_tout)
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
cap.update({'inner_' + k: v for k, v in inner.items()})
np.savez_compressed('esmfold2_conf.npz', **cap)
for k in sorted(cap): print("  %-36s %s" % (k, cap[k].shape))
