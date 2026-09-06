'''
pick attention kernels and XLA flags from the actual device

This is the portability matrix from ColabFold's AlphaFold3_of3.ipynb, moved out of
a notebook cell and into tested code. It is knowledge that was expensive to
acquire -- each row is a real failure someone hit -- and it should not have to be
re-derived in every notebook that wants a fast kernel.

The headline rule: **every GPU gets a fused attention kernel; only the kernel
differs.** Triton is the fastest on real datacenter cards (A100/H100); Ada and
consumer Ampere cannot launch Triton but *can* run cuDNN fused attention, which
still beats XLA; pre-Ampere and CPU fall back to XLA.

  device                        cap        attention  extra XLA flags
  ----------------------------  ---------  ---------  -----------------------------
  CPU                           -          xla        (also prefer nojit)
  V100, T4        (pre-Ampere)  < 8.0      xla        disable custom-kernel-fusion-rewriter
  A100            (datacenter)  == 8.0     triton     disable Triton GEMM
  L4, RTX 30/40   (Ada)         8.6, 8.9   cudnn      disable Triton GEMM
  H100 and newer  (datacenter)  >= 9.0     triton     disable Triton GEMM

Why Ada does not get Triton even though it is >= 8.0: its shared memory is smaller
than the Triton attention kernels need, so they fail with "Shared memory size
limit exceeded" *at launch*, which tokamax's trace-time fallback cannot catch
(tokamax gates Triton on `compute_capability >= 8.0`, wrong for these cards -- see
needs_tokamax_patch()). But cuDNN's fused attention launches fine on Ada and is
measurably faster than XLA (1.48x on the A10 i_con scorer, 1.26x on the diffusion
path, i_con/ipSAE identical to ~0.007), so it -- not XLA -- is the Ada default.
Pass flash_attention='xla' explicitly for bit-comparable differential tests.
'''

from __future__ import annotations

import subprocess

XLA = 'xla'
TRITON = 'triton'
CUDNN = 'cudnn'

# XLA passes that have to be turned off per device family
NO_TRITON_GEMM = '--xla_gpu_enable_triton_gemm=false'
NO_CUSTOM_KERNEL_FUSION = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'


def detect_device() -> tuple[str, float | None]:
  '''("gpu", compute_capability) or ("cpu", None), via nvidia-smi'''
  try:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=15)
    caps = [float(x) for x in out.stdout.split() if x.strip()]
    if caps:
      return 'gpu', min(caps)      # the weakest GPU sets the policy
  except Exception:
    pass
  return 'cpu', None


def is_datacenter_gpu(cap: float | None) -> bool:
  '''A100 (8.0) and H100+ (>= 9.0) have enough shared memory for Triton

  Deliberately excludes 8.6 and 8.9 (Ada / consumer Ampere), which report a
  capability above 8.0 but cannot launch the kernels.
  '''
  if cap is None:
    return False
  return cap == 8.0 or cap >= 9.0


def attention_config(device: str = None, cap: float | None = None) -> dict:
  '''attention implementation and XLA flags for a device

  Pass device/cap to reason about a machine you are not on (and to test this);
  omit them to detect the current one.
  '''
  if device is None:
    device, cap = detect_device()

  if device == 'cpu':
    return {'attention': XLA, 'xla_flags': [], 'nojit': True,
            'reason': 'no GPU: XLA attention, and prefer nojit to skip the compile'}

  if cap is not None and cap < 8.0:
    return {'attention': XLA, 'xla_flags': [NO_CUSTOM_KERNEL_FUSION],
            'nojit': False,
            'reason': f'pre-Ampere GPU (cc {cap}): no Triton support, and the '
                      'custom-kernel fusion pass has to be disabled'}

  if is_datacenter_gpu(cap):
    return {'attention': TRITON, 'xla_flags': [NO_TRITON_GEMM], 'nojit': False,
            'reason': f'datacenter GPU (cc {cap}): Triton flash attention, with '
                      'Triton GEMM disabled per AlphaFold 3 guidance'}

  return {'attention': CUDNN, 'xla_flags': [NO_TRITON_GEMM], 'nojit': False,
          'reason': f'Ada/consumer GPU (cc {cap}): Triton kernels cannot launch '
                    '(shared memory too small), but cuDNN fused attention can and '
                    'is faster than XLA -- measured 1.48x on the A10 i_con scorer '
                    'and 1.26x on the diffusion path, numerically within ~0.007. '
                    'Triton GEMM stays off.'}


def needs_tokamax_patch(cap: float | None = None) -> bool:
  '''True on cards where tokamax would wrongly enable Triton

  tokamax gates on `compute_capability >= 8.0`, which includes Ada (8.6/8.9). Those
  cards then fail at kernel launch rather than at trace time, so its own fallback
  never fires. ColabFold's notebook patches tokamax's gpu_utils to
  `cc == 8.0 or cc >= 9.0`.

  This is an upstream bug worth reporting rather than patching forever.
  '''
  if cap is None:
    _device, cap = detect_device()
  return cap is not None and cap >= 8.0 and not is_datacenter_gpu(cap)


def apply_xla_flags(flags, env=None) -> str:
  '''append flags to XLA_FLAGS without clobbering what is already there'''
  import os
  env = os.environ if env is None else env
  cur = env.get('XLA_FLAGS', '')
  for f in flags:
    if f not in cur:
      cur = (cur + ' ' + f).strip()
  if cur:
    env['XLA_FLAGS'] = cur
  return cur


# --------------------------------------------------------- compilation cache

# Our own cache dir, deliberately NOT colabdesign2's. Sharing one directory
# between packages is how a benchmark ends up silently measuring cache hits
# built by the other one.
DEFAULT_CACHE = '~/.cache/alphafold3/jax'


def enable_compilation_cache(path: str = DEFAULT_CACHE, min_seconds: float = 1.0):
  '''persist compiled executables across processes

  AF2 and AF3 both compile for a minute or more before the first design step --
  measured at L=300, 120 s for AF2's einsum attention and 79 s with the flash
  kernel. That cost is paid again on every run, for shapes that have already
  been compiled a hundred times, because JAX's cache is per-process by default.

  This is the largest single lever on time-to-first-step, and unlike a kernel
  change it cannot alter a number: the cache is keyed on the HLO plus the
  backend and target config, so a hit is the same executable that would have
  been built. A stale entry is a cache miss, not a wrong answer.

  min_seconds skips caching anything trivial, so the cache holds the handful of
  large executables that matter rather than thousands of small ones.

  Returns the path in use, or None if this build of JAX has no cache API.
  '''
  import os

  import jax

  path = os.path.expanduser(path)
  try:
    os.makedirs(path, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', path)
    # only cache what is worth the disk
    jax.config.update('jax_persistent_cache_min_compile_time_secs', min_seconds)
    # without this, JAX declines to cache anything it considers not worth it
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    return path
  except Exception:
    return None


def cache_stats(path: str = DEFAULT_CACHE) -> dict:
  '''how many executables are cached and how much disk they use'''
  import os
  path = os.path.expanduser(path)
  if not os.path.isdir(path):
    return {'path': path, 'entries': 0, 'bytes': 0}
  n = total = 0
  for root, _dirs, files in os.walk(path):
    for f in files:
      n += 1
      total += os.path.getsize(os.path.join(root, f))
  return {'path': path, 'entries': n, 'bytes': total}
