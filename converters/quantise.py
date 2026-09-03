"""Store a converted blob smaller: bfloat16, or int8 with per-channel scales.

Why this exists: a ported blob is written as float32 (converters/common.
write_params_blob) while the model runs in bfloat16, so two of every four bytes
we ask a Colab user to download are thrown away on load. int8 goes further, at a
cost that has to be measured rather than assumed -- see docs/ported_models.md.

Layout. A quantised array is written as two records: the int8 payload under the
parameter's own name, and its scales under `<name>__q_scale`. Scales are
per-CHANNEL over the last axis, which is the output axis of every Linear in this
graph, so each output feature gets its own range; a single scale per tensor is
markedly worse on the attention projections, whose columns differ in magnitude.
`dequantise_records` puts them back together, so nothing downstream of the
loader knows the difference.

Left alone in every scheme:
  * anything that is not float32 -- integer tables and the __meta__ identifier.
  * 1-D arrays. Biases and LayerNorm scales/offsets are a rounding error of the
    total bytes and sit where an error is not attenuated by a matmul average.
  * small arrays (< `min_elements`), for the same reason: the bytes are not
    worth the risk.
"""

import numpy as np

BF16 = 'bfloat16'
SCALE_SUFFIX = '__q_scale'
MIN_ELEMENTS = 2 ** 16


def to_bfloat16(a):
  """float32 -> the uint16 bit pattern of bfloat16, round-to-nearest-even.

  Stored as uint16 rather than ml_dtypes.bfloat16 so the record dtype string
  stays something numpy can always read back; `dequantise_records` converts.
  """
  u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
  return (((u >> 16) + ((u >> 15) & 1)) & 0xFFFF).astype(np.uint16)


def from_bfloat16(u16):
  return (u16.astype(np.uint32) << 16).view(np.float32)


def quantise_int8(a):
  """symmetric per-last-axis-channel int8. Returns (int8 array, fp32 scales)."""
  a = np.asarray(a, dtype=np.float32)
  flat = a.reshape(-1, a.shape[-1])
  scale = np.abs(flat).max(axis=0) / 127.0
  scale[scale == 0] = 1.0                  # an all-zero channel stays all-zero
  q = np.rint(flat / scale).clip(-127, 127).astype(np.int8)
  return q.reshape(a.shape), scale.astype(np.float32)


def dequantise_int8(q, scale):
  shape = q.shape
  return (q.reshape(-1, shape[-1]).astype(np.float32)
          * scale).reshape(shape)


def to_half(a, half):
  """float32 -> the chosen half-precision storage dtype.

  float16 is the better default despite the name bfloat16 appearing everywhere
  in this graph. Both are two bytes; float16 spends three more of them on the
  mantissa and two fewer on the exponent, and every weight in every ported blob
  fits its range with room to spare (the largest anywhere is chai-1's 70.4,
  against a 65504 ceiling). That buys about 8x the precision -- rms/std 0.0004
  against bfloat16's 0.0031 -- for roughly 19% more compressed bytes, because a
  truncated mantissa is what made bfloat16 compress better in the first place.

  It matters only for the parameters that stay float32 at RUNTIME: the
  bfloat16 getter casts the rest on load, so for those, storing bfloat16 and
  storing float16 give the same number. The ones that remain are the
  `precision='highest'` Linears and the coordinate and chirality projections --
  which is to say the numerically touchy ones.
  """
  if half == 'bf16':
    return to_bfloat16(a)
  return np.asarray(a, dtype=np.float16)


def _is_float(arr):
  """Any floating record, whatever a converter chose to write it as.

  Not every blob is float32: intellifold-2's converter wrote 61% of its bytes as
  bfloat16 already, and an earlier version of this module tested `dtype ==
  float32` and so skipped all of it -- if2 shrank 1.34x where every other model
  shrank 2.4x. Whatever the stored dtype, the material is the same.
  """
  # ml_dtypes.bfloat16 is a numpy EXTENSION dtype: its kind is 'V' (void) and
  # np.issubdtype(..., np.floating) is False for it, so test the name too.
  return (np.issubdtype(arr.dtype, np.floating)
          or arr.dtype.name in ('bfloat16', 'float8_e4m3fn', 'float8_e5m2'))


def _quantisable(arr, min_elements):
  return _is_float(arr) and arr.ndim >= 2 and arr.size >= min_elements


SCHEMES = ('fp32', 'fp16', 'bf16', 'int8', 'int8-bf16')


def quantise_records(records, scheme, *, min_elements=MIN_ELEMENTS):
  """(scope, name, arr) records -> the same, stored under `scheme`.

    fp32        unchanged, what the converters wrote originally
    fp16/bf16   every float32 array in half precision
    int8        int8 for the big matrices, float16 for the rest
    int8-bf16   the same with bfloat16 for the rest, kept to measure the two
  """
  if scheme not in SCHEMES:
    raise ValueError(f'unknown scheme {scheme!r}, want one of {SCHEMES}')
  if scheme == 'fp32':
    return list(records)
  half = 'bf16' if scheme in ('bf16', 'int8-bf16') else 'fp16'
  use_int8 = scheme.startswith('int8')
  out = []
  for scope, name, arr in records:
    arr = np.asarray(arr)
    if not _is_float(arr):
      out.append((scope, name, arr))       # integer tables, the __meta__ ident
      continue
    if arr.dtype != np.float32:
      arr = arr.astype(np.float32)         # bfloat16 in, from an older converter
    if use_int8 and _quantisable(arr, min_elements):
      q, scale = quantise_int8(arr)
      out.append((scope, name, q))
      out.append((scope, name + SCALE_SUFFIX, scale))
    else:
      out.append((scope, name, to_half(arr, half)))
  return out


def dequantise_records(records):
  """The inverse: -> (scope, name, float32 array) with the scales consumed.

  Tolerates a blob written in any of the three schemes, so an old fp32 blob and
  a new int8 one load through the same path.
  """
  scales = {}
  payload = []
  for scope, name, arr in records:
    if name.endswith(SCALE_SUFFIX):
      scales[(scope, name[: -len(SCALE_SUFFIX)])] = np.asarray(arr)
    else:
      payload.append((scope, name, np.asarray(arr)))

  out = []
  for scope, name, arr in payload:
    scale = scales.get((scope, name))
    if scale is not None:
      if arr.dtype != np.int8:
        raise ValueError(f'{scope}/{name} has a scale but is {arr.dtype}, not int8')
      out.append((scope, name, dequantise_int8(arr, scale)))
    elif arr.dtype == np.uint16:
      # the only uint16 this format carries is a bfloat16 bit pattern; a real
      # uint16 table would be an integer feature, and there are none.
      out.append((scope, name, from_bfloat16(arr)))
    elif arr.dtype == np.float16:
      out.append((scope, name, arr.astype(np.float32)))
    else:
      out.append((scope, name, arr))
  unused = set(scales) - {(s, n) for s, n, _ in payload}
  if unused:
    raise ValueError(f'{len(unused)} scale records with no payload: '
                     f'{sorted(unused)[:3]}')
  return out


def requantise_blob(src, dst, scheme, *, level=10, min_elements=MIN_ELEMENTS):
  """Rewrite a .bin.zst under a different storage scheme. Returns (bytes, ratio)."""
  import os
  import zstandard
  from alphafold3.model.params import encode_record
  from converters.common import read_blob

  records = quantise_records(read_blob(src), scheme, min_elements=min_elements)
  with zstandard.ZstdCompressor(level=level).stream_writer(open(dst, 'wb')) as w:
    for scope, name, arr in records:
      w.write(encode_record(scope, name, arr))
  before, after = os.path.getsize(src), os.path.getsize(dst)
  return after, before / after


if __name__ == '__main__':
  import sys
  if len(sys.argv) < 4:
    print(__doc__)
    print('\nusage: python -m converters.quantise <src.bin.zst> <dst.bin.zst> '
          '<' + '|'.join(SCHEMES) + '>')
    raise SystemExit(2)
  size, ratio = requantise_blob(sys.argv[1], sys.argv[2], sys.argv[3])
  print(f'{sys.argv[3]}: {size/1e9:.3f} GB  ({ratio:.2f}x smaller)')
