'''
pick attention kernels and XLA flags from the actual device

This is the portability matrix from ColabFold's AlphaFold3_of3.ipynb, by Milot
Mirdita, moved out of a notebook cell and into tested code. It is knowledge that
was expensive to acquire -- each row is a real failure someone hit -- and it
should not have to be re-derived in every notebook that wants a fast kernel.

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

import glob
import os
import subprocess
import sys

XLA = 'xla'
VOLTA = 'volta'
PALLAS = 'pallas'
TOKAMAX = 'tokamax'
TRITON = 'triton'
CUDNN = 'cudnn'

# XLA passes that have to be turned off per device family
NO_TRITON_GEMM = '--xla_gpu_enable_triton_gemm=false'
NO_CUSTOM_KERNEL_FUSION = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'


def detect_device() -> tuple[str, float | None]:
  '''("gpu", compute_capability), ("tpu", None) or ("cpu", None)

  A TPU HAS TO BE TOLD APART FROM A CPU, and nvidia-smi cannot do it: it is
  simply absent on both. Colab offers TPU runtimes (v5e1, v6e1), and one
  misread as a CPU takes the CPU row below -- which sets `nojit`. Measured on
  a Colab TPU v5 lite: eager is **14.3x** the jit time. That is the whole cost
  of one missing branch.

  jax is asked first when it is ALREADY imported, because it is the authority
  on the backend it will use; it is never imported here, since this module is
  called from a notebook cell that wants an answer in milliseconds.
  '''
  jax = sys.modules.get('jax')
  if jax is not None:
    try:
      backend = jax.default_backend()
      if backend == 'tpu':
        return 'tpu', None
      if backend == 'gpu':
        caps = [float(d.compute_capability) for d in jax.local_devices()
                if getattr(d, 'compute_capability', None)]
        if caps:
          return 'gpu', min(caps)
    except Exception:
      pass
  try:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=15)
    caps = [float(x) for x in out.stdout.split() if x.strip()]
    if caps:
      return 'gpu', min(caps)      # the weakest GPU sets the policy
  except Exception:
    pass
  # No nvidia-smi is not the same as no accelerator. A TPU VM exposes its
  # chips as /dev/accel*, and Colab sets these in the environment.
  if (glob.glob('/dev/accel*') or os.environ.get('COLAB_TPU_ADDR')
      or os.environ.get('TPU_WORKER_ID') or os.environ.get('TPU_ACCELERATOR_TYPE')):
    return 'tpu', None
  return 'cpu', None


def is_datacenter_gpu(cap: float | None) -> bool:
  '''A100 (8.0) and H100+ (>= 9.0) have enough shared memory for Triton

  Deliberately excludes 8.6 and 8.9 (Ada / consumer Ampere), which report a
  capability above 8.0 but cannot launch the kernels.
  '''
  if cap is None:
    return False
  return cap == 8.0 or cap >= 9.0


def attention_config(device: str = None, cap: float | None = None,
                     differentiable: bool = False) -> dict:
  '''attention implementation and XLA flags for a device

  Pass device/cap to reason about a machine you are not on (and to test this);
  omit them to detect the current one.

  `differentiable=True` says this graph will be backpropagated through, which
  rules out the pre-Ampere kernels: they are forward-only ("The FFI call to
  `VoltaMma` cannot be differentiated"). The design path backprops through the
  trunk and GridSelfAttention is in it, so a design caller MUST set this.
  '''
  if device is None:
    device, cap = detect_device()

  if device == 'tpu':
    # NOT the CPU row. XLA attention is right -- neither Triton nor cuDNN has
    # a TPU backend -- but `nojit` would be a disaster: measured on a Colab
    # TPU v5 lite, eager is 14.3x the jit time. The xla_gpu_* flags are
    # meaningless here, so none are passed.
    return {'attention': XLA, 'xla_flags': [], 'nojit': False,
            'reason': 'TPU: XLA attention (no CUDA kernel has a TPU backend), '
                      'and jit stays ON -- eager measured 14.3x the jit time. '
                      'Worth having: on a Colab v5e, 1STP (121 tokens, 10 '
                      'recycles, 5 samples, warm) took 7.45 s against an A10 '
                      "+cuDNN's 17.5 s, and the structures agree to 0.018-"
                      '0.030 A CA-RMSD at the same pTM 0.92. Verified with '
                      'openbind0; the other models have not been run here.'}

  if device == 'cpu':
    return {'attention': XLA, 'xla_flags': [], 'nojit': True,
            'reason': 'no GPU: XLA attention, and prefer nojit to skip the compile'}

  if cap is not None and cap < 8.0:
    from alphafold3.model.components import volta_attn
    # A DIFFERENTIABLE CALLER NEEDS A BACKWARD, and wheels before 0.4.0 have
    # none: jax refuses with `The FFI call to VoltaMma cannot be
    # differentiated`. 0.4.0 ships one for both families -- `VoltaMmaBwd`
    # (sm_75) and `VoltaWmmaBwd` (sm_70), each giving dQ, dK, dV and dBias,
    # and AF3 reaches the pair representation through the bias, so dBias is
    # not optional -- so a design run on a T4 or a V100 keeps the kernel
    # instead of falling back to XLA. An older wheel answers False below and
    # this row is not taken.
    usable = volta_attn.installed() and (
        not differentiable or volta_attn.bwd_installed(int(cap * 10)))
    if usable:
      return {'attention': VOLTA, 'xla_flags': [NO_CUSTOM_KERNEL_FUSION],
              'nojit': False,
              'reason': f'pre-Ampere GPU (cc {cap}): the only fused attention '
                        'this card can run is colabfold-legacy-kernels '
                        "(Milot Mirdita's sm_70/sm_75 kernels) -- 3.0-3.35x "
                        'the XLA path on a T4, in float16. The same answer '
                        "also sends the triangle multiplication's input "
                        'LayerNorm and GLU to that package.'
                        + (' colabfold-legacy-kernels 0.4.0 added the '
                           'backward for both families, so this serves a '
                           'gradient too.' if differentiable else '')}

  if cap is not None and cap < 8.0:
    # XLA gates Pallas/Triton at sm_80, cuDNN's SDPA needs SM80, and tokamax
    # offers nothing here -- so XLA attention is the only thing this fork can
    # currently run on a T4 or V100.
    #
    # It is NOT the only thing that exists. Milot Mirdita builds
    # `colabfold-legacy-kernels` (sm_70/sm_75 CUDA kernels registered as XLA
    # FFI targets: attention with a nonbatched bias, layer_norm, and a gated
    # dual projection) and drives them from `alphafold/model/volta_attn.py` in
    # alphafold-colabfold; ColabFold's AF2 path selects them below sm_80 as
    # `cuda_legacy`, in float16, because Volta and Turing tensor cores have no
    # bfloat16. Measured on a Colab T4 at our triangle-attention shape, they
    # are 3.0-3.35x XLA (max|d| 1e-4). Wiring them in here is unstarted work,
    # and it belongs to him.
    return {'attention': XLA, 'xla_flags': [NO_CUSTOM_KERNEL_FUSION],
            'nojit': False,
            'reason': f'pre-Ampere GPU (cc {cap}): no Triton support, and the '
                      'custom-kernel fusion pass has to be disabled'}

  if cap is not None and not is_datacenter_gpu(cap) and not differentiable:
    from alphafold3.model.components import pallas_attn
    if pallas_attn.installed():
      # The Ada / consumer-Ampere row, where tokamax refuses to launch at all.
      # Measured on an A10 at this model's triangle-attention shape (N=384):
      # XLA 8.635 ms, cuDNN 2.409, this 0.723 -- 3.3x the cuDNN this row used
      # to take. The datacenter row is NOT sent here: tokamax's Triton does
      # launch there and has not been compared against this yet.
      #
      # `not differentiable` IS NO LONGER A CAPABILITY LIMIT, IT IS A
      # MEASUREMENT. colabfold-kernels 0.4.0 gave this kernel a real flash
      # backward (dQ/dK/dV and dBias through atomics), so it CAN be
      # backpropagated through -- gated here on an A10 at 3.1e-03 to 6.3e-03
      # against an fp32 XLA reference. It is simply slower at it than the
      # Triton the differentiable branch below picks, at the triangle shape
      # this model actually runs (forward+backward, ms, min of 5, A10):
      #
      #     N=256   triton  2.351   pallas  3.897   cudnn  3.925
      #     N=384   triton  7.085   pallas 12.605   cudnn 11.702
      #     N=512   triton 15.193   pallas 30.130   cudnn 27.760
      #
      # while its FORWARD is the faster of the two (1.484 vs 1.722 at N=384).
      # So the split stays: pallas predicts, triton differentiates -- and the
      # backward is now insurance rather than a hole, because a graph that
      # reaches it under jax.grad returns a gradient instead of raising.
      return {'attention': PALLAS, 'xla_flags': [NO_TRITON_GEMM], 'nojit': False,
              'reason': f'Ada/consumer GPU (cc {cap}): colabfold-kernels '
                        "(Milot Mirdita's Pallas flash attention) sizes its "
                        'blocks to the device, so it runs where tokamax will '
                        'not -- 3.3x cuDNN and 11.9x XLA on an A10. It can '
                        'serve a gradient since 0.4.0, but Triton is 1.8x '
                        'faster at one, so a differentiable caller is sent '
                        'there instead.'}

  if is_datacenter_gpu(cap):
    # ATTENTION STAYS ON TRITON, THE GLU DOES NOT. Measured on a dedicated A100
    # (not Colab): the three fused attentions are within noise of each other
    # (triton 3.318 ms, Milot's pallas 3.265, the Anthropic kit 3.490 at
    # N=768), so there is nothing to win by moving attention and tokamax is the
    # better-tested path. tokamax's GLU, though, is worth NOTHING over plain
    # XLA here -- 1.013 ms against 1.009 at N=384, 4.026 against 4.021 at
    # N=768 -- while the Pallas GLU is 1.20x at both. End to end on a
    # 768-residue fold, 10 recycles, warm, n=4 interleaved:
    #     triton + tokamax GLU   31.96 31.96 31.96 31.93 s
    #     triton + pallas  GLU   31.20 31.19 31.21 31.21 s   -2.35%
    # Small, but 25x the spread, and free. Below 512 tokens it is invisible:
    # 6LU7 at 306 is 21.8 s warm whatever the kernels.
    glu = TOKAMAX
    if not differentiable:
      from alphafold3.model.components import pallas_attn
      if pallas_attn.installed():
        glu = PALLAS
    return {'attention': TRITON, 'glu': glu, 'xla_flags': [NO_TRITON_GEMM],
            'nojit': False,
            'reason': f'datacenter GPU (cc {cap}): Triton flash attention, with '
                      'Triton GEMM disabled per AlphaFold 3 guidance'
                      + ('; the triangle multiplication takes the Pallas GLU '
                         "(Milot Mirdita's colabfold-kernels), 1.20x the op and "
                         '-2.35% on a 768-residue fold, because tokamax\'s GLU '
                         'is worth nothing over XLA here' if glu == PALLAS else '')}

  if differentiable:
    # A GRADIENT ON ADA GOES TO TRITON, NOT cuDNN -- and the reason is memory,
    # not speed. tokamax refuses this card through a blanket
    # `cc == 8.0 or cc >= 9.0` in gpu_utils.has_triton_support, commented
    # "Ada/L4 lack shared memory". That premise does not hold for any shape
    # this model uses: with the gate bypassed, triangle attention at
    # c=16/32/64/128, the diffusion transformer's h=16/c=48, the ESM tower's
    # h=20/c=64 and sequences to 1024 all launch, forward AND backward, on an
    # A10. Measured, gradient through one trunk pass:
    #
    #     150 residues   cuDNN 0.643 s   triton 0.554 s
    #     300 residues   cuDNN 3.352 s   triton 2.609 s
    #     384 residues   cuDNN OOM (22.96 GiB)   triton 4.201 s
    #
    # cuDNN's backward materialises what a flash backward does not, so on a
    # 23 GB card a 384-residue design step is impossible with it and fine with
    # Triton. The op-level gap is larger (4.921 ms against 10.364 at N=384)
    # and, as ever, does not survive to the end-to-end measure.
    #
    # attention.py enables tokamax for this case and falls back to cuDNN if it
    # refuses -- the refusal is a NotImplementedError at TRACE time, which is
    # catchable; a shared-memory failure at launch would not be.
    return {'attention': TRITON, 'xla_flags': [NO_TRITON_GEMM], 'nojit': False,
            'reason': f'Ada/consumer GPU (cc {cap}), differentiable: Triton. '
                      "tokamax's blanket refusal of this card does not hold "
                      'for any shape here, and cuDNN\'s backward OOMs at 384 '
                      'residues where Triton runs in 4.2 s.'}

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


def preinit_tokamax_context() -> None:
  '''Create tokamax's JAX user context BEFORE the first trace.

  tokamax builds its autotuning-cache overlay lazily, and the overlay carries a
  `jax.make_user_context(())` (ops/op.py: get_autotuning_cache_overlay_state).
  The first tokamax op to run creates it -- which happens INSIDE the first trace
  of the model. JAX includes the user context in the jit cache key, so the entry
  cached during that trace is keyed WITHOUT the context while every later call is
  keyed with it: a guaranteed miss, and a full retrace plus recompile of the
  whole model on call 2.

  Measured on an A100 (alphafold3, 64 tokens, identical arguments both calls):

      without      call 0 62.5 s   call 1 45.3 s   call 2 2.5 s   2 traces
      with         call 0 62.5 s   call 1  2.5 s   call 2 2.5 s   1 trace

  So it costs a second cold compile on every fresh process. Invisible on
  hardware where tokamax's Pallas/Triton kernels are unavailable (an A10 raises
  NotImplementedError for them and never creates the context), which is why this
  only shows up on datacentre GPUs -- exactly the ones people rent.

  IT LIVES HERE, not in run_alphafold.py, because anything that builds a model
  in-process needs it and must not import a CLI script to get it.

  Best-effort: the import path is tokamax-internal, so a version without it must
  not break inference.
  '''
  try:
    from tokamax._src.ops import op as _tokamax_op

    _tokamax_op.get_autotuning_cache_overlay_state()
  except Exception:  # pylint: disable=broad-except
    pass
