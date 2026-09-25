"""The two ESM-C checkpoint layouts must convert to the same tower."""
import sys

import numpy as np

sys.path.insert(0, '.')
from converters import esm_lm  # noqa: E402


def _hf(n_layers=2, d=8, ffn=16, vocab=5):
  rng = np.random.default_rng(0)
  r = lambda *s: rng.standard_normal(s).astype(np.float32)
  sd = {'esmc.embed_tokens.weight': r(vocab, d), 'esmc.norm.weight': r(d)}
  for i in range(n_layers):
    p = 'esmc.layers.%d.' % i
    sd.update({
        p + 'input_layernorm.weight': r(d), p + 'input_layernorm.bias': r(d),
        p + 'self_attn.q_proj.weight': r(d, d),
        p + 'self_attn.k_proj.weight': r(d, d),
        p + 'self_attn.v_proj.weight': r(d, d),
        p + 'self_attn.o_proj.weight': r(d, d),
        p + 'self_attn.q_norm.weight': r(d), p + 'self_attn.k_norm.weight': r(d),
        p + 'post_attention_layernorm.weight': r(d),
        p + 'post_attention_layernorm.bias': r(d),
        p + 'mlp.gate_proj.weight': r(ffn, d), p + 'mlp.up_proj.weight': r(ffn, d),
        p + 'mlp.down_proj.weight': r(d, ffn),
    })
  return sd


def test_the_hf_layout_becomes_the_native_one():
  sd = _hf()
  out = esm_lm.normalise_esmc_layout(sd)
  assert 'esmc.embed.weight' in out
  assert 'esmc.transformer.norm.weight' in out
  assert 'esmc.transformer.blocks.0.attn.layernorm_qkv.weight' in out
  # the two orderings a NAME cannot settle, both measured against a native copy
  # of the same 300M tower: the reverse of either is a silently wrong tower.
  b = 'esmc.transformer.blocks.0.'
  q, k, v = (sd['esmc.layers.0.self_attn.%s_proj.weight' % x] for x in 'qkv')
  assert np.array_equal(out[b + 'attn.layernorm_qkv.weight'],
                        np.concatenate([q, k, v], 0))
  g = sd['esmc.layers.0.mlp.gate_proj.weight']
  u = sd['esmc.layers.0.mlp.up_proj.weight']
  assert np.array_equal(out[b + 'ffn.fc1_weight'], np.concatenate([g, u], 0))


def test_a_native_checkpoint_passes_through_untouched():
  native = {'esmc.embed.weight': np.zeros((4, 4), np.float32)}
  assert esm_lm.normalise_esmc_layout(native) is native


def test_something_that_is_not_an_esmc_tower_is_left_alone():
  other = {'layers.0.fc1.weight': np.zeros((2, 2), np.float32)}
  assert esm_lm.normalise_esmc_layout(other) is other


def test_the_normalised_tree_derives_the_same_dims():
  out = esm_lm.normalise_esmc_layout(_hf(n_layers=3, d=8, ffn=16))
  dims = esm_lm.derive_dims(out, 'esmc')
  assert dims['n_layers'] == 3 and dims['d_model'] == 8 and dims['ffn_hidden'] == 16
