"""Can an MSA substitute for ESM-C?  2x2 on a NATURAL protein (5CAJ YaaA, 261 aa)."""
import numpy as np, torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import (
    prepare_protein_features, PROTEIN_RESIDUE_TO_RES_TYPE, PROTEIN_1TO3, PROTEIN_UNK_RES_TYPE)

A3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
      'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
      'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def parse_cif(path, chain='A'):
    cols, rows, inloop = {}, [], False
    for l in open(path):
        s = l.strip()
        if s.startswith('_atom_site.'):
            cols[s.split('.')[1]] = len(cols); inloop = True; continue
        if inloop:
            if s.startswith('#') or s.startswith('loop_') or s.startswith('_'):
                if rows: break
                continue
            p = s.split()
            if len(p) >= len(cols): rows.append(p)
    out, seen = [], set()
    for p in rows:
        if p[cols['group_PDB']] != 'ATOM' or p[cols['label_atom_id']] != 'CA': continue
        if p[cols['auth_asym_id']] != chain: continue
        key = p[cols['auth_seq_id']]
        if key in seen: continue
        seen.add(key)
        out.append((int(key), A3.get(p[cols['label_comp_id']], 'X'),
                    [float(p[cols['Cartn_%s' % a]]) for a in 'xyz']))
    return out                      # (auth_seq_id, letter, xyz) -- KEEP the numbering

def read_a3m(path, max_depth):
    seqs, cur = [], None
    for l in open(path):
        if l.startswith('>'):
            if cur is not None: seqs.append(cur)
            cur = ''
            if len(seqs) >= max_depth: break
        else: cur += l.strip()
    if cur is not None and len(seqs) < max_depth: seqs.append(cur)
    return seqs

def rmsd(a, b):
    n = min(len(a), len(b)); a, b = a[:n]-a[:n].mean(0), b[:n]-b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T@b); d = np.sign(np.linalg.det(u@vt))
    return float(np.sqrt((((a@(u@np.diag([1,1,d])@vt))-b)**2).sum(1).mean()))

res = parse_cif('/home/ubuntu/5CAJ.cif')
a3m = read_a3m('/home/ubuntu/msa_yaaa.a3m', 256)
query = a3m[0]
SEQ = query
# 5CAJ chain A carries a 5-residue expression tag (PRGSH, auth -4..0) and SIX
# unresolved loops, so a positional pred[:n] vs nat[:n] comparison is misaligned
# by a growing offset.  Map by auth NUMBER: model token j <-> auth_seq_id j+1,
# keep only resolved residues, and assert the residue letters agree.
pairs = [(n - 1, xyz, c) for n, c, xyz in res if 1 <= n <= len(query) and query[n - 1] == c]
sel = np.array([i for i, _, _ in pairs])
nat = np.array([x for _, x, _ in pairs])
print('5CAJ: %d resolved CA, %d aligned to the query (%d res); %d dropped '
      '(tag + mismatches)' % (len(res), len(sel), len(query), len(res) - len(sel)))

m = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16").cuda().eval()
feats = {k: v.cuda() for k, v in prepare_protein_features(SEQ).items()}
ch = feats['ref_atom_name_chars'][0].cpu().numpy().astype(int)
a2t = feats['atom_to_token'][0].cpu().numpy().astype(int)
msk = feats['atom_attention_mask'][0].cpu().numpy().astype(bool)
nm = lambda i: ''.join(chr(c+32) for c in ch[i]).strip()
rep = np.full(int(a2t[msk].max())+1, -1)
for i2 in range(len(a2t)):
    if msk[i2] and nm(i2) == 'CA': rep[a2t[i2]] = i2

L = len(SEQ)
def build_msa(seqs):
    """a3m -> (msa ids [1,M,L], mask, has_deletion, deletion_value)."""
    M = len(seqs)
    ids = np.full((M, L), PROTEIN_UNK_RES_TYPE, np.int64)
    hasdel = np.zeros((M, L), np.float32); delval = np.zeros((M, L), np.float32)
    for r, s in enumerate(seqs):
        col, ndel = 0, 0
        for c in s:
            if c.islower(): ndel += 1; continue
            if col >= L: break
            if c != '-':
                ids[r, col] = PROTEIN_RESIDUE_TO_RES_TYPE.get(PROTEIN_1TO3.get(c.upper(), 'UNK'),
                                                              PROTEIN_UNK_RES_TYPE)
            if ndel: hasdel[r, col] = 1.0; delval[r, col] = 2/np.pi*np.arctan(ndel/3.0)
            ndel = 0; col += 1
    t = lambda a, d: torch.tensor(a, dtype=d).unsqueeze(0).cuda()
    return dict(msa=t(ids, torch.int64), msa_attention_mask=t(np.ones((M, L)), torch.bool),
                has_deletion=t(hasdel, torch.float32), deletion_value=t(delval, torch.float32))

print()
print('%-30s %-10s %s' % ('5CAJ (natural, 261 aa)', 'CA-RMSD', 'pLDDT'))
for tag, use_lm, depth in [('ESM-C on,  MSA depth 1', True, 1),
                           ('ESM-C on,  MSA depth 256', True, 256),
                           ('ESM-C OFF, MSA depth 1', False, 1),
                           ('ESM-C OFF, MSA depth 256', False, 256)]:
    kw = dict(feats)
    if not use_lm: kw.pop('input_ids')
    if depth > 1: kw.update(build_msa(a3m[:depth]))
    torch.manual_seed(0)
    with torch.no_grad():
        o = m(**kw, num_loops=3, num_diffusion_samples=1, num_sampling_steps=200)
    x = o['sample_atom_coords'].float().cpu().numpy()
    if x.ndim == 3: x = x[0]
    pred = x[rep][sel]
    print('  %-28s %-10.3f %.3f' % (tag, rmsd(pred, nat), float(o['plddt'].mean())))
