# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under Apache License, Version 2.0. To view a
copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import enum
import functools
import io
import logging
import os
import pickle
import pathlib

# Silence tokamax CPU-fallback errors before any other imports fire them.
# tokamax logs at ERROR when a GPU kernel isn't available, then falls back
# silently — these are not real errors on CPU-only machines.
logging.getLogger('tokamax').setLevel(logging.CRITICAL)
import shutil
import string
import textwrap
import time
import typing
from typing import overload
import warnings

from absl import app
from absl import flags
from absl import logging as absl_logging
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
from alphafold3.constants import decoded_ccd
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.data.tools import shards
from alphafold3.model import features
from alphafold3.model import model_config
from alphafold3.model import model
from alphafold3.model import model_registry
from alphafold3.model.pipeline import model_features
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model import weights
from alphafold3.model.components import utils
from etils import epath
import haiku as hk
# Suppress "Unable to initialize backend 'tpu'" at JAX import time.
logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)
# Suppress XLA C++ delay-kernel timing noise (cuda_timer.cc).
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import jax
from jax import numpy as jnp
import numpy as np
import tokamax

_HOME_DIR = epath.Path('~').expanduser()
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


@enum.unique
class JaxBackend(enum.StrEnum):
  AUTO = enum.auto()
  CPU = enum.auto()
  GPU = enum.auto()
  TPU = enum.auto()


# Input and output paths.
_JSON_PATH = epath.DEFINE_path(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = epath.DEFINE_path(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = epath.DEFINE_path(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)
MODEL_DIR = epath.DEFINE_path(
    'model_dir',
    None,
    'Path to the model to use for inference. Defaults to $HOME/models for'
    ' AlphaFold 3 itself, and to a per-model cache directory for the ported'
    ' models, whose converted weights are fetched on first use.',
)
_DOWNLOAD_WEIGHTS = flags.DEFINE_bool(
    'download_weights',
    True,
    'Whether a ported model whose weights are not in --model_dir may fetch the'
    ' published conversion. Never applies to AlphaFold 3 itself, whose'
    ' parameters must be requested from Google DeepMind.',
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    'db_dir',
    (_DEFAULT_DB_DIR.as_posix(),),
    'Path to the directory containing the databases. Can be specified multiple'
    ' times to search multiple directories in order.',
)
_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_SMALL_BFD_Z_VALUE = flags.DEFINE_integer(
    'small_bfd_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_MGNIFY_Z_VALUE = flags.DEFINE_integer(
    'mgnify_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_Z_VALUE = flags.DEFINE_integer(
    'uniprot_cluster_annot_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_UNIREF90_Z_VALUE = flags.DEFINE_integer(
    'uniref90_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_NTRNA_Z_VALUE = flags.DEFINE_float(
    'ntrna_z_value',
    None,
    'The Z-value representing the database size in megabases for E-value'
    ' calculation. Must be set for sharded databases.',
    lower_bound=0.0,
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RFAM_Z_VALUE = flags.DEFINE_float(
    'rfam_z_value',
    None,
    'The Z-value representing the database size in megabases for E-value'
    ' calculation. Must be set for sharded databases.',
    lower_bound=0.0,
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_RNA_CENTRAL_Z_VALUE = flags.DEFINE_float(
    'rna_central_z_value',
    None,
    'The Z-value representing the database size in megabases for E-value'
    ' calculation. Must be set for sharded databases.',
    lower_bound=0.0,
)
_PDB_DATABASE_PATH = epath.DEFINE_path(
    'pdb_database_path',
    '${DB_DIR}/mmcif_files',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)


def _resolve_jax_backend() -> JaxBackend:
  """Returns the backend to run on, auto-detecting a GPU if --jax_backend=auto."""
  if _JAX_BACKEND.value != JaxBackend.AUTO:
    return _JAX_BACKEND.value
  try:
    has_gpu = bool(jax.local_devices(backend='gpu'))
  except RuntimeError:
    has_gpu = False  # JAX has no GPU backend registered at all.
  if has_gpu:
    return JaxBackend.GPU
  # A TPU IS NOT A CPU. Without this the next line called a Colab TPU runtime
  # "CPU-only", and everything below then placed the fold on the VM's CPU --
  # with the accelerator sitting idle and nothing in the output saying so.
  try:
    has_tpu = bool(jax.local_devices(backend='tpu'))
  except RuntimeError:
    has_tpu = False
  if has_tpu:
    return JaxBackend.TPU
  print('No JAX GPU or TPU backend found, falling back to CPU-only inference.')
  return JaxBackend.CPU


def _num_cpus_for_msa_tools() -> int:
  try:
    # Unfortunately, os.process_cpu_count() is only available in Python 3.13+.
    num_cpus = len(os.sched_getaffinity(0))
  except AttributeError:
    num_cpus = os.cpu_count()  # MacOS doesn't have os.sched_getaffinity().
  return min(num_cpus if num_cpus is not None else 8, 8)


# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    _num_cpus_for_msa_tools(),
    'Number of CPUs to use for Jackhmmer. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
_JACKHMMER_MAX_PARALLEL_SHARDS = flags.DEFINE_integer(
    'jackhmmer_max_parallel_shards',
    None,
    'Maximum number of shards to search against in parallel. If unset, one'
    ' Jackhmmer instance will be run per shard. Only applicable if the'
    ' database is sharded.',
    lower_bound=1,
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    _num_cpus_for_msa_tools(),
    'Number of CPUs to use for Nhmmer. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
_NHMMER_MAX_PARALLEL_SHARDS = flags.DEFINE_integer(
    'nhmmer_max_parallel_shards',
    None,
    'Maximum number of shards to search against in parallel. If unset, one'
    ' Nhmmer instance will be run per shard. Only applicable if the'
    ' database is sharded.',
    lower_bound=1,
)

# Data pipeline configuration.
_RESOLVE_MSA_OVERLAPS = flags.DEFINE_bool(
    'resolve_msa_overlaps',
    True,
    'Whether to deduplicate unpaired MSA against paired MSA. The default'
    ' behaviour matches the method described in the AlphaFold 3 paper. Set this'
    ' to false if providing custom paired MSA using the unpaired MSA field to'
    ' keep it exactly as is as deduplication against the paired MSA could break'
    ' the manually crafted pairing between MSA sequences.',
)
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    'max_template_date',
    '2021-09-30',  # By default, use the date from the AlphaFold 3 paper.
    'Maximum template release date to consider. Format: YYYY-MM-DD. All'
    ' templates released after this date will be ignored. Controls also whether'
    ' to allow use of model coordinates for a chemical component from the CCD'
    ' if RDKit conformer generation fails and the component does not have ideal'
    ' coordinates set. Only for components that have been released before this'
    ' date the model coordinates can be used as a fallback.',
)
_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
    lower_bound=0,
)
_FIX_STANDALONE_GLYCANS = flags.DEFINE_bool(
    'fix_standalone_glycans',
    False,
    'AlphaFold 3 model training and evaluation filtered out leaving atoms from'
    ' glycan ligands even if they were not bonded to anything ("standalone"'
    ' glycans). Setting this flag to True fixes this undesirable behavior, but'
    ' moves away from the regime where AlphaFold 3 was trained and evaluated.',
)

# JAX inference performance tuning.
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir',
    '/tmp/alphafold_cache',
    'Directory for all inference caches (JAX compilation and tokamax'
    ' autotuning). Persists within a session. For cross-session persistence on'
    ' Colab, point to a mounted Google Drive path.',
)
_GPU_DEVICE = flags.DEFINE_integer(
    'gpu_device',
    0,
    'Optional override for the GPU device to use for inference, uses zero-based'
    ' indexing. Defaults to the 0th GPU on the system. Useful on multi-GPU'
    ' systems to pin each run to a specific GPU. Note that if GPUs are already'
    ' pre-filtered by the environment (e.g. by using CUDA_VISIBLE_DEVICES),'
    ' this flag refers to the GPU index after the filtering has been done.',
)
_JAX_BACKEND = flags.DEFINE_enum_class(
    'jax_backend',
    default=JaxBackend.AUTO,
    enum_class=JaxBackend,
    help=(
        'JAX backend to use. "gpu" uses a GPU for inference. "cpu" uses a CPU'
        ' only for inference. This is much slower than using a GPU, but can be'
        ' useful for testing or running on systems without a GPU supported by'
        ' JAX. If you set this flag to "cpu", you must also set'
        ' --flash_attention_implementation=xla. "auto" (the default) uses a GPU'
        ' if JAX exposes one and otherwise falls back to the CPU, setting'
        ' --flash_attention_implementation=xla for you.'
    ),
)
_BUCKETS = flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    ['32', '64', '128', '256', '384', '512', '768', '1024', '1280', '1536',
     '2048', '2560', '3072', '3584', '4096', '4608', '5120'],
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)
_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    'flash_attention_implementation',
    default='auto',
    enum_values=['auto', 'triton', 'cudnn', 'xla', 'volta', 'pallas'],
    help=(
        "Flash attention implementation to use. 'auto' (the default) asks"
        ' alphafold3.model.components.platform, which picks per compute'
        ' capability: Triton on A100/H100, cuDNN on Ada and consumer Ampere'
        ' (which cannot launch the Triton kernels -- too little shared memory'
        ' -- but run cuDNN fine), XLA below Ampere and on CPU. The old default'
        " was 'triton', which is unrunnable on most of those cards, so every"
        ' caller had to carry its own copy of that table -- and the notebook'
        ' copy said XLA for Ada, costing 2.6x on triangle attention (11.89 ms'
        ' -> 4.50 ms at 384 tokens, bit-identical; 20% off a whole fold).'
        " 'triton' and 'cudnn' are the fused kernels and need Ampere or later;"
        " 'volta' is Milot Mirdita's colabfold-legacy-kernels, the only fused"
        ' attention sm_70/sm_75 can run (3x XLA on a T4, in float16, and'
        ' FORWARD ONLY -- it cannot be differentiated);'
        " 'pallas' is his colabfold-kernels, a Pallas flash attention with a"
        ' non-batched bias that sizes its blocks to the device, so it launches'
        ' on the Ada and consumer-Ampere cards tokamax refuses -- 3.3x cuDNN'
        ' and 11.9x XLA on an A10, and FORWARD ONLY for the same reason (a'
        ' Pallas call has no VJP);'
        " 'xla' is portable and the one every device has."
    ),
)
_GLU_KERNEL = flags.DEFINE_enum(
    'glu_kernel',
    default='auto',
    enum_values=['auto', 'tokamax', 'pallas', 'volta'],
    help=(
        "Which GLU kernel the triangle multiplication uses. 'auto' (the"
        ' default) follows --flash_attention_implementation, which is what'
        ' every caller got before this flag existed. They are separable'
        ' because an A100 wants them different: there the fused attentions are'
        ' within noise of each other while tokamax\'s GLU is worth NOTHING'
        " over plain XLA (1.013 ms against 1.009 at N=384) and Milot Mirdita's"
        ' Pallas GLU is 1.20x at every size measured.'
    ),
)
_NUM_SAMPLING_STEPS = flags.DEFINE_integer(
    'num_sampling_steps',
    None,
    'Denoising steps the diffusion sampler takes. None keeps the model\'s own'
    " default, which is what every released number was measured at. It is a"
    ' flag because step count is a REAL variable when something is placed'
    ' wrongly: a group that gets worse with more steps is being pushed by the'
    ' loop, and one that is flat is not short of budget. ESMFold2 defaults to'
    ' 15, AF3 to 200.',
    lower_bound=1,
)
_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_DROPOUT = flags.DEFINE_bool(
    'dropout',
    False,
    'Run the network WITH the dropout it was trained with, instead of the'
    ' deterministic inference path. AlphaFold 3 was trained with 0.25 dropout'
    ' in the pair stacks and 0.15 in the MSA module (SI Algorithms 8/16/17,'
    ' section 5.5); the released inference code strips it, and this package'
    ' carries it so a run can use it as a stochastic regulariser -- which makes'
    ' repeated seeds explore rather than repeat. Applies to every af3-family'
    ' model, since they share the graph. Off by default: prediction is meant to'
    ' be deterministic.',
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
    lower_bound=1,
)
_NUM_SEEDS = flags.DEFINE_integer(
    'num_seeds',
    None,
    'Number of seeds to use for inference. If set, only a single seed must be'
    ' provided in the input JSON. AlphaFold 3 will then generate random seeds'
    ' in sequence, starting from the single seed specified in the input JSON.'
    ' The full input JSON produced by AlphaFold 3 will include the generated'
    ' random seeds. If not set, AlphaFold 3 will use the seeds as provided in'
    ' the input JSON.',
    lower_bound=1,
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    'save_embeddings',
    False,
    'Whether to save the final trunk single and pair embeddings in the output.'
    ' Note that the embeddings are large float16 arrays: num_tokens * 384'
    ' + num_tokens * num_tokens * 128.',
)
_SAVE_DISTOGRAM = flags.DEFINE_bool(
    'save_distogram',
    False,
    'Whether to save the final distogram in the output. Note that the distogram'
    ' is a large float16 array: num_tokens * num_tokens * 64.',
)
_SAVE_TERMS_OF_USE = flags.DEFINE_bool(
    'save_terms_of_use',
    True,
    'Whether to save the terms of use as an MD file in the output directory.',
)
_FORCE_OUTPUT_DIR = flags.DEFINE_bool(
    'force_output_dir',
    False,
    'Whether to force the output directory to be used even if it already exists'
    ' and is non-empty. Useful to set this to True to run the data pipeline and'
    ' the inference separately, but use the same output directory.',
)
_COMPRESS_LARGE_OUTPUT_FILES = flags.DEFINE_bool(
    'compress_large_output_files',
    False,
    'If True, compresses the output mmCIF and confidences JSON files (the two'
    ' largest files) using zstandard. Note that embeddings and distogram, if'
    ' saved, are already stored in a compressed format.',
)

# Which model family to run. `alphafold3` is DeepMind's own; the rest are ported
# AF3-family models whose weights we convert offline (see converters/README.md)
# and load from --model_dir. The name selects the forward branches, the config
# shapes and the sampler constants all at once -- see
# alphafold3.model.model_registry.
# AlphaFold 2 is in this list too. It does not ride the AF3 graph -- see
# alphafold3.af2 -- so it is dispatched on `spec.engine`, and its weights are
# DeepMind's own params_model_*.npz read from --model_dir rather than a
# converted blob.
_MODEL = flags.DEFINE_enum(
    'model',
    'alphafold3',
    sorted(list(model_registry.MODEL_SPECS) + list(model_registry.AF2_SPECS)),
    'Which model to run. Its weights must already be in --model_dir: conversion'
    ' is a separate offline step (python -m converters.convert --model NAME'
    ' --out DIR), so a run never needs torch or the original checkpoint.'
    ' The af2_* models instead want a directory holding AlphaFold 2\'s'
    ' params/params_model_*.npz.',
)

_USE_ESM_EMBEDDINGS = flags.DEFINE_bool(
    'use_esm_embeddings',
    False,
    'Run this model with its protein language model. Two of the ported models'
    ' fold from one and are a different model without it: chai-1 from ESM2 3B'
    ' (its token feature stream is mostly ESM2 -- a natural protein folds to'
    ' 5.70 A without it where chai-1 reaches 0.642), and ESMFold2 from ESM-C,'
    ' which is its alternative to an MSA rather than an extra on top of one --'
    ' a variant with no MSA encoder folds to ~14 A without it.'
    ' The tower runs in-process and is downloaded on demand (2.4 GB for ESM2,'
    ' 5.1 GB for ESM-C 6B), which is why this is opt-in rather than the'
    ' default. WHICH tower is read from the model, not chosen here: ESMFold2'
    ' variants are trained against step-matched ESM-C snapshots.',
)

_FEATURISE_OFF = flags.DEFINE_list(
    'featurise_off',
    [],
    'Input conventions to switch OFF for this model, by name (see'
    ' model_registry._FEATURISE), e.g. --featurise_off=padded_keys. For'
    ' measuring whether a convention is load-bearing: each one is silent when'
    ' wrong, so the only way to know what it buys is to run without it.',
)
_CYCLIC = flags.DEFINE_list(
    'cyclic',
    [],
    'Chains to treat as CYCLIC -- their relative-position encoding wraps, so'
    ' they have no N- or C-terminus. Pass chain ids ("A,B"), or "all" for every'
    ' polymer chain. This is not an AlphaFold 3 feature and is not specific to'
    ' any one model: the encoding is shared, so every model here honours it,'
    ' and a chain left out is byte-identical to before.',
)
_WEIGHTS_PRECISION = flags.DEFINE_enum(
    'weights_precision', 'int8', ['fp32', 'fp16', 'int8'],
    'Which published form of the weights to fetch. int8 is the DEFAULT: the same'
    ' weights stored smaller (12.4 GB of models becomes 2.6) and expanded on'
    ' load, which is what makes them practical on a metered or slow connection'
    ' such as Colab. Measured cost on rosettafold3: within sampling noise on'
    ' protein, ligand, RNA and a D/L peptide, with stereochemistry unchanged --'
    ' see docs/ported_models.md; every published int8 blob is gated by'
    ' dev/int8_roundtrip_check.py at the 1/256 int8 floor. fp32 is published too'
    ' and is the exact bytes the converters wrote, if you want them. Ignored'
    ' when --model_dir points at weights you already have.')

_LOWERCACHE_DIR = flags.DEFINE_string(
    'lowercache_dir',
    None,
    'Directory of SERIALIZED EXECUTABLES (jax.experimental.serialize_executable).'
    ' The persistent compile cache stores the compiled binary but NOT the trace'
    ' and lowering in front of it, and on this model that floor is 10.75 s of a'
    ' 24.22 s first fold at 256 tokens (measured: same input, two seeds in one'
    ' process -> 24.22 s then 13.47 s). A serialized executable skips it. The'
    ' blob is valid only for the same jax/jaxlib, GPU model, XLA flags, model'
    ' config and input signature; a miss falls back to tracing and WRITES the'
    ' blob, so the first run populates the directory and later runs load it.',
)
_PRECOMPILE = flags.DEFINE_list(
    'precompile',
    [],
    'Token counts to compile the model for, then exit without folding, e.g.'
    ' --precompile=128,256. XLA compiles per input shape, and with --cache_dir'
    ' the executable persists -- so this moves a cold compile out of a user\'s'
    ' first fold and into a step that can run at install time. Measured on'
    ' esmfold2_fast at 128 tokens: a first real fold goes from 102 s to 44 s,'
    ' and to 29 s once the language-model tower is warm too. It folds a DUMMY'
    ' sequence, so no real input is needed; only the token count has to match,'
    ' which is what --buckets makes predictable.',
)

_DYNAMIC_RECYCLES = flags.DEFINE_bool(
    'dynamic_recycles',
    False,
    'Pass the recycle count into the model as a traced argument instead of'
    ' baking it into the graph. The count is otherwise a Python int, so every'
    ' --num_recycles value compiles its own executable and a persistent'
    ' compile cache built at one value misses at every other. With this, one'
    ' executable serves them all -- which is what makes a SHIPPED cache'
    ' useful, since a user who changes the recycle count would otherwise pay a'
    ' full compile. Prediction only: lax.fori_loop with a dynamic bound is not'
    ' reverse-differentiable, so the design path must leave this off.',
)

_STEPWISE_RECYCLES = flags.DEFINE_bool(
    'stepwise_recycles',
    False,
    'Compile ONE trunk pass and call it once per recycle from Python, instead'
    ' of a fori_loop whose trip count is baked into the graph. One executable'
    ' then serves every --num_recycles value -- which is what makes a shipped'
    ' compile cache useful -- and the embeddings are available between passes,'
    ' so a caller can show what each recycle changed. Prediction only, and NOT'
    ' bit-identical to the fused path: each pass gets an independently split'
    ' PRNG key rather than one carried through a scan.',
)

# Set to a callable(pass_index, embeddings) to observe each recycle; used by the
# notebook to draw the structure as it converges. Left None, costs nothing.
_TRUNK_CALLBACK = [None]

# How many denoise steps go in one dispatch. Frames still arrive one per step;
# this only decides how often the host is involved. 1 is the finest and the
# slowest.
_LIVE_CHUNK = [10]

_NUM_MSA = flags.DEFINE_integer(
    'num_msa',
    None,
    'MSA rows the trunk subsamples to (default 1024, keeping the query at row'
    ' 0). Featurisation always produces a fixed 16384-row buffer, so this is'
    ' about what the trunk READS, not what is downloaded. Barely a speed dial'
    ' -- 1 vs 1024 measured 0.4% at 512 tokens -- but it lowers memory, and a'
    ' different value is a different compiled executable.',
)

_AF2_NUM_MODELS = flags.DEFINE_integer(
    'af2_num_models',
    1,
    "How many of AlphaFold 2's five parameter sets to run, for --model=af2_*."
    ' Each is a separately trained model rather than a seed, so running several'
    ' and ranking them is what ColabFold does and why num_models exists there.'
    ' All five are downloaded and loaded either way; this decides how many are'
    ' USED. Ignored by the AF3-family models, which have one set of weights.',
)

_NOJIT = flags.DEFINE_bool(
    'nojit',
    False,
    'Disable JAX JIT compilation. Useful for debugging.',
)

_USE_MSA_SERVER = flags.DEFINE_bool(
    'use_msa_server',
    False,
    'Query the ColabFold/MMseqs2 server to generate MSAs for protein and RNA'
    ' chains that are missing them. Requires internet access. Results are'
    ' cached in --cache_dir so subsequent runs skip the network.',
)

_MSA_SERVER_URL = flags.DEFINE_string(
    'msa_server_url',
    'https://api.colabfold.com',
    'URL of the ColabFold MSA server.',
)

_MSA_SERVER_USER_AGENT = flags.DEFINE_string(
    'msa_server_user_agent',
    'alphafold3/1.0',
    'HTTP User-Agent string sent to the MSA server.',
)


def _bfloat16_default() -> str:
  """GlobalConfig.bfloat16: 'all' where bf16 pays, 'intermediate' where it does not.

  'intermediate' keeps the trunk and confidence head in bf16 and leaves the
  diffusion sampler in float32 -- which is what this model did for its whole
  life. 'all' extends bf16 to the sampler too, worth ~10% at 256 tokens on an
  A10 (cc 8.6).

  The boundary is compute capability 8.0 (Ampere) and it is MEASURED. Below it
  there are no bf16 tensor cores and XLA's converts cost more than the narrower
  operands save: on a real Colab T4 (sm_75), openbind0 at 59 residues / 10
  recycles / 5 samples reads 30.91 s with the sampler in f32 and 32.19 s in
  bf16 -- +4.2%, over interleaved reps on warm caches. It runs correctly either
  way, it is just slower, so a T4 gets 'intermediate'.

  The TRUNK keeps bf16 on that card regardless: its own cost there is 1.1%, and
  'none' would buy that back by doubling the [N, N, 128] pair representation's
  activation memory on the GPU with the least to spare.

  Here rather than in the model because choosing needs the device, and a model
  module should not be asking what GPU this is. AF3_SAMPLER_BF16=1 / =0 forces
  'all' / 'intermediate'.
  """
  env = os.environ.get('AF3_SAMPLER_BF16')
  if env is not None:
    return 'all' if env not in ('', '0', 'false', 'False') else 'intermediate'
  try:
    device = jax.devices()[0]
    # A TPU has no compute_capability to read, and bfloat16 is its native
    # matmul type -- 'intermediate' (the except branch's answer) would leave
    # the sampler in f32 on the one accelerator built around bf16.
    if device.platform == 'tpu':
      return 'all'
    cc = str(getattr(device, 'compute_capability', '') or '')
    major, _, minor = cc.partition('.')
    return 'all' if (int(major), int(minor or 0)) >= (8, 0) else 'intermediate'
  except Exception:
    return 'intermediate'


def make_model_config(
    *,
    flash_attention_implementation: tokamax.DotProductAttentionImplementation = 'triton',
    glu_kernel: str = 'auto',
    num_diffusion_samples: int = 5,
    num_sampling_steps: int | None = None,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
    model_name: str = 'alphafold3',
    num_msa: int | None = None,
) -> model.Model.Config:
  """Returns a model config with some defaults overridden.

  `model_name` selects the family: it lands in global_config.model, which every
  ported-family forward branch keys on, and brings that family's config shapes
  and sampler constants with it (model_registry.ModelSpec.configure).
  """
  config = model.Model.Config()
  config.global_config.flash_attention_implementation = (
      flash_attention_implementation
  )
  config.global_config.glu_kernel = glu_kernel
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  config.return_distogram = return_distogram
  config.global_config.bfloat16 = _bfloat16_default()
  model_registry.get(model_name).configure(config)
  # AFTER configure(), NOT BEFORE. The model spec sets its own sampler
  # constants -- ESMFold2's 15 steps, AF3's 200 -- so an assignment made before
  # it is silently overwritten, and a sweep at 5/15/60 steps then returns three
  # IDENTICAL structures, which is how this was caught.
  if num_sampling_steps is not None:
    config.heads.diffusion.eval.steps = num_sampling_steps
  # HOW MANY MSA ROWS THE TRUNK SEES. Featurisation always hands over a fixed
  # 16384-row buffer (pipeline.msa_crop_size) and the trunk subsamples to this,
  # keeping the query at row 0. Lowering it is the one MSA knob a user can turn.
  #
  # Measured, and worth saying before anyone reaches for it as a speed dial:
  # sweeping 1 / 256 / 1024 moved steady-state runtime by 0.4% at 512 tokens.
  # It changes MEMORY and it changes the compiled executable -- a different
  # value is a different shape, so it misses a compile cache built at another.
  #
  # ANY value is allowed, not just the notebook's ladder. Two need a guard:
  # below 1 there is no query row, and above the featurisation buffer jax does
  # NOT raise -- a gather CLAMPS out-of-range indices, so `--num_msa=20000`
  # against a 16384-row buffer would hand the trunk 3616 duplicates of the
  # last padded row and fold on quietly (checked: a[arange(3,8)] on a 5-row
  # array repeats row 4 four times).
  if num_msa is not None:
    num_msa = int(num_msa)
    from alphafold3.model.pipeline import pipeline as _model_pipeline
    buffer = _model_pipeline.WholePdbPipeline.Config().msa_crop_size
    if num_msa < 1:
      raise ValueError(
          f'--num_msa must be at least 1 -- row 0 is the query -- got {num_msa}.')
    if num_msa > buffer:
      print(f'--num_msa={num_msa} is above the {buffer}-row featurisation '
            f'buffer, so {buffer} is what the trunk can actually read; using '
            'that. Rows past the buffer would be copies of its last padded row.')
      num_msa = buffer
    config.evoformer.num_msa = num_msa

  return config


class ModelRunner:
  """Helper class to run structure prediction stages."""

  def __init__(
      self,
      config: model.Model.Config,
      device: jax.Device,
      model_dir: epath.PathLike,
      use_dropout: bool = False,
  ):
    self._model_config = config
    self._device = device
    self._model_dir = epath.Path(model_dir)
    # Traced, not baked: `_pair_dropout` computes its rate as
    # jnp.where(use_dropout, rate, 0), so OFF is an exact identity
    # (bernoulli(keep=1) is all-ones) and toggling costs no recompile.
    self._use_dropout = use_dropout
    self._autotune_result = self._load_autotune_cache()
    self._autotune_attempted = False
    self._lowercache_memo = {}
    self._jitted_apply = None

  @property
  def model_dir(self) -> epath.Path:
    return self._model_dir

  @property
  def model_name(self) -> str:
    """Which family this runner was built for (global_config.model)."""
    return self._model_config.global_config.model

  @property
  def _autotune_cache_path(self) -> str | None:
    return (
        os.path.join(_CACHE_DIR.value, 'tokamax_autotune.json')
        if _CACHE_DIR.value else None
    )

  def _load_autotune_cache(self):
    path = self._autotune_cache_path
    if path and os.path.exists(path):
      print(f'Loading tokamax autotune cache from {path}')
      with open(path) as f:
        return tokamax.AutotuningResult.load(f)
    return None

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory.

    A converted model ships a shape manifest beside its blob (written by
    converters/shapes.py). When one is present it is the graph's own statement of
    the parameter tree, so any gap in the weights is reported and filled with
    zeros here rather than surfacing as an opaque haiku error mid-forward -- or,
    worse, not surfacing at all.
    """
    loaded = params.get_model_haiku_params(model_dir=self._model_dir)
    return loaded

  @staticmethod
  def _preinit_tokamax_context() -> None:
    """Create tokamax's JAX user context BEFORE the first trace.

    tokamax builds its autotuning-cache overlay lazily, and the overlay carries
    a `jax.make_user_context(())` (ops/op.py: get_autotuning_cache_overlay_state).
    The first tokamax op to run creates it -- which happens INSIDE the first
    trace of the model. JAX includes the user context in the jit cache key, so
    the entry cached during that trace is keyed without the context while every
    later call is keyed with it: a guaranteed miss, and a full RETRACE plus
    recompile of the whole model on call 2.

    Measured on an A100 (alphafold3, 64 tokens, identical arguments both calls):

        without      call 0 62.5 s   call 1 45.3 s   call 2 2.5 s   2 traces
        with         call 0 62.5 s   call 1  2.5 s   call 2 2.5 s   1 trace

    So it costs a second cold compile on every fresh process. Invisible on
    hardware where tokamax's Pallas/Triton kernels are unavailable (an A10
    raises NotImplementedError for them and never creates the context), which is
    why this only shows up on datacentre GPUs -- exactly the ones people rent.

    Best-effort: the import path is tokamax-internal, so a version without it
    must not break inference.
    """
    try:
      from tokamax._src.ops import op as _tokamax_op

      _tokamax_op.get_autotuning_cache_overlay_state()
    except Exception:  # pylint: disable=broad-except
      pass

  def live_model(self):
    """Staged, frame-by-frame fold. See alphafold3.model.staged.

    The driver moved into the library so design code can import it without
    importing this script; what is left here is the binding of config and
    parameters that a CLI run already has.
    """
    from alphafold3.model import staged
    return staged.staged_fold(
        self._model_config, self.model_params, jit=not _NOJIT.value,
        chunk=_LIVE_CHUNK[0], use_dropout=self._use_dropout,
        on_trunk=_TRUNK_CALLBACK[0])

  def _stepwise_model(self):
    """--stepwise_recycles: live_model without the callbacks.

    It WAS a second copy of the same loop, which is how an embed-stage fix
    landed here and not in live_model -- the driver the notebook actually
    uses -- and the duplicate trunk trace it was meant to remove stayed.
    """
    run = self.live_model()
    return lambda rng_key, batch: run(rng_key, batch)

  @functools.cached_property
  def _model(
      self,
  ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""

    @hk.transform
    def forward_fn(batch, num_trunk_passes=None):
      return model.Model(self._model_config)(
          batch, use_dropout=self._use_dropout,
          num_trunk_passes_override=num_trunk_passes)

    if _STEPWISE_RECYCLES.value:
      return self._stepwise_model()

    apply_fn = forward_fn.apply
    if not _NOJIT.value:
      # No `device=`: jit's backend/device arguments are deprecated, and they
      # were redundant here -- run_inference already device_puts the batch on
      # self._device, and jax runs a computation where its committed inputs
      # are. Passing it only bought a DeprecationWarning on every run.
      apply_fn = jax.jit(apply_fn)
    # before anything is traced -- see _preinit_tokamax_context
    self._preinit_tokamax_context()
    extra = {}
    if _DYNAMIC_RECYCLES.value:
      n = model.num_trunk_passes(self._model_config.num_recycles,
                                 self._model_config.global_config.model)
      extra['num_trunk_passes'] = jnp.asarray(n, jnp.int32)
    if _LOWERCACHE_DIR.value and not _NOJIT.value:
      self._jitted_apply = apply_fn
      self._lowercache_extra = extra
      return self._lowercached_model
    return functools.partial(apply_fn, self.model_params, **extra)

  def _lowercache_key(self, rng_key, batch) -> str:
    """What a serialized executable is only valid for.

    The persistent compile cache keys on the program and the stack; this has to
    key on the same things PLUS the input signature, because a blob is an
    executable for ONE set of avals. XLA flags are in here because they change
    the program, and device_kind because an executable is built for a card.
    """
    import hashlib
    import jaxlib
    def avals(tree):
      return [(getattr(x, 'shape', None), str(getattr(x, 'dtype', type(x))))
              for x in jax.tree_util.tree_leaves(tree)]
    material = repr([
        jax.__version__, getattr(jaxlib, '__version__', '?'),
        str(getattr(jax.devices()[0], 'device_kind', '?')),
        os.environ.get('XLA_FLAGS', ''),
        os.environ.get('AF3_SAMPLER_BF16', ''),
        self.model_name, repr(self._model_config), self._use_dropout,
        avals(self.model_params), avals(rng_key), sorted(batch),
        avals([batch[k] for k in sorted(batch)]),
        # an executable is built for the kwargs it was lowered with, and
        # --dynamic_recycles passes one
        sorted(getattr(self, '_lowercache_extra', {})),
        avals(getattr(self, '_lowercache_extra', {})),
    ])
    return hashlib.sha256(material.encode()).hexdigest()[:32]

  def _lowercached_model(self, rng_key, batch):
    """The jitted apply, with trace+lowering served from disk when it can be.

    A miss lowers, compiles and WRITES the blob, so the directory populates
    itself; only the trace and lowering are skipped on a hit, and the executable
    is the same one jit would have produced (checked: a serialize round trip
    returns bit-identical values).
    """
    from jax.experimental import serialize_executable as _se
    key = self._lowercache_key(rng_key, batch)
    got = self._lowercache_memo.get(key)
    if got is None:
      d = epath.Path(_LOWERCACHE_DIR.value)
      path = d / f'{key}.bin'
      t0 = time.time()
      if path.exists():
        try:
          blob, in_tree, out_tree = pickle.loads(path.read_bytes())
          got = _se.deserialize_and_load(blob, in_tree, out_tree)
          print(f'  lowercache HIT  {path.name} in {time.time() - t0:.2f} s')
        except Exception as err:   # a blob from another stack: say so, retrace
          print(f'  lowercache blob unusable ({type(err).__name__}: {err}); '
                'lowering instead')
          got = None
      if got is None:
        lowered = self._jitted_apply.lower(
            self.model_params, rng_key, batch,
            **getattr(self, '_lowercache_extra', {}))
        got = lowered.compile()
        print(f'  lowercache MISS: lowered and compiled in '
              f'{time.time() - t0:.2f} s')
        try:
          d.mkdir(parents=True, exist_ok=True)
          payload = pickle.dumps(_se.serialize(got))
          path.write_bytes(payload)
          print(f'  lowercache WROTE {path.name} '
                f'({len(payload) / 1e6:.1f} MB)')
        except Exception as err:
          print(f'  lowercache could not be written ({type(err).__name__}: '
                f'{err}); this run is unaffected')
      self._lowercache_memo[key] = got
    return got(self.model_params, rng_key, batch,
               **getattr(self, '_lowercache_extra', {}))

  def run_inference(
      self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
  ) -> model.ModelResult:
    """Computes a forward pass of the model on a featurised example."""
    featurised_example = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
        ),
        self._device,
    )

    # ONCE PER PROCESS, and say why if it fails. On a failure this left
    # _autotune_result None, so every later inference tried again -- once per
    # seed, per bucket, per fold job -- and swallowed the reason each time:
    # silent repeated work with nothing to diagnose from. Anthropic's af3_jax
    # optimization kit reports the same thing against this fork as its FIX1
    # lever ('stock calls tokamax.autotune() before the first inference of
    # every new bucket size and the call raises on this stack').
    if (self._autotune_result is None and self._autotune_cache_path
        and not self._autotune_attempted):
      self._autotune_attempted = True
      try:
        self._autotune_result = tokamax.autotune(self._model, rng_key, featurised_example)
        os.makedirs(os.path.dirname(os.path.abspath(self._autotune_cache_path)), exist_ok=True)
        with open(self._autotune_cache_path, 'w') as f:
          self._autotune_result.dump(f)
        print(f'Tokamax autotune cache saved to {self._autotune_cache_path}')
        print('Subsequent runs will load this cache and skip autotuning.')
      except Exception as err:  # pylint: disable=broad-except
        print('TOKAMAX autotune unavailable, continuing without it: '
              f'{type(err).__name__}: {err}')

    if self._autotune_result is not None:
      with self._autotune_result:
        result = self._model(rng_key, featurised_example)
    else:
      result = self._model(rng_key, featurised_example)
    result = jax.tree.map(np.asarray, result)
    result = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        result,
    )
    result = dict(result)
    # A ported blob may carry no meta record; the identifier is provenance
    # stamped into the output, so fall back to the model's own name.
    meta = self.model_params.get('__meta__', {}).get('__identifier__')
    identifier = (np.asarray(meta).tobytes() if meta is not None
                  else self.model_name.encode())
    result['__identifier__'] = identifier
    return result

  def extract_inference_results(
      self,
      batch: features.BatchDict,
      result: model.ModelResult,
      target_name: str,
  ) -> list[model.InferenceResult]:
    """Extracts inference results from model outputs."""
    return list(
        model.Model.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )

  def extract_embeddings(
      self, result: model.ModelResult, num_tokens: int
  ) -> dict[str, np.ndarray] | None:
    """Extracts embeddings from model outputs."""
    embeddings = {}
    if 'single_embeddings' in result:
      embeddings['single_embeddings'] = result['single_embeddings'][
          :num_tokens
      ].astype(np.float16)
    if 'pair_embeddings' in result:
      embeddings['pair_embeddings'] = result['pair_embeddings'][
          :num_tokens, :num_tokens
      ].astype(np.float16)
    return embeddings or None

  def extract_distogram(
      self, result: model.ModelResult, num_tokens: int
  ) -> np.ndarray | None:
    """Extracts distogram from model outputs."""
    if 'distogram' not in result['distogram']:
      return None
    distogram = result['distogram']['distogram'][:num_tokens, :num_tokens, :]
    return distogram


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
    embeddings: The final trunk single and pair embeddings, if requested.
    distogram: The token distance histogram, if requested.
  """

  seed: int
  inference_results: Sequence[model.InferenceResult]
  full_fold_input: folding_input.Input
  embeddings: dict[str, np.ndarray] | None = None
  distogram: np.ndarray | None = None


def _precompile(model_runner, model_name, model_dir, token_counts):
  """Compile the model for each token count, then return without folding.

  A DUMMY sequence is folded through the ordinary code path. That is
  deliberate: the persistent cache key covers the whole jit configuration, so
  rebuilding the graph some other way is a way to compile something subtly
  different, and driving the real path cannot drift from what a fold does. By
  the time this runs the parameters are loaded anyway.

  Only the token count has to match the eventual input, which is what --buckets
  makes predictable: a 68-residue protein lands in the 128 bucket.
  """
  from alphafold3.constants import decoded_ccd
  from alphafold3.model import model_registry

  spec = model_registry.get(model_name)
  ccd = decoded_ccd.get_ccd()
  for n_tokens in token_counts:
    t0 = time.time()
    fold_input = folding_input.Input(
        name='precompile', rng_seeds=[0],
        chains=[folding_input.ProteinChain(
            id='A', sequence='G' * max(n_tokens - 60, 8), ptms=[],
            unpaired_msa='', paired_msa='', templates=[])])
    featurise = lambda **kw: featurisation.featurise_input(
        fold_input=fold_input, ccd=ccd, buckets=[n_tokens], **kw)
    batch = featurise()[0]
    if spec.featurise:
      n = int(np.asarray(batch['token_index']).shape[-1])
      lm_pair = None
      if spec.featurise.get('lm_pair'):
        lm_pair = np.zeros(
            (n, n, model_runner._model_config.evoformer.pair_channel),
            np.float32)
      esm = None
      if spec.featurise.get('esm'):
        esm = np.zeros((int(np.asarray(batch['is_protein']).sum()), 2560),
                       np.float32)
      batch = model_features.apply(
          batch, spec, refeaturise=featurise, model_dir=model_dir, esm=esm,
          has_msa=False, fold_input=fold_input, lm_pair=lm_pair)
    jax.block_until_ready(
        model_runner.run_inference(batch, jax.random.PRNGKey(0)))
    print(f'  compiled {model_name} for {n_tokens} tokens in '
          f'{time.time() - t0:.1f} s')
  print('Precompiled. Re-run with the same --cache_dir to fold without '
        'compiling.')


def _protein_sequences(fold_input):
  return [c.sequence for c in fold_input.chains
          if isinstance(c, folding_input.ProteinChain)]


def _resolve_esm(use_esm, fold_input, model_runner):
  """--use_esm_embeddings -> (esm2_rows, esmc_pair), either of which may be None.

  Both families fold from a language model, but they consume different things:
  chai-1 takes ESM2's last hidden state as a TOKEN feature, and ESMFold2 takes
  ESM-C's states through its own shim to become a PAIR representation. One flag
  covers both because the choice is the model's, not the user's.

  Resolved HERE rather than in main because it depends on the sequence, and one
  run can hold several fold inputs.
  """
  if not use_esm:
    return None, None
  from alphafold3.model import weights as weights_lib
  from alphafold3.model import esm

  model_name = model_runner.model_name
  sequences = _protein_sequences(fold_input)

  if model_name == 'chai1':
    print('Running the ESM2 tower for chai-1...')
    # Multi-chain is fine here: the rows are concatenated in chain order and
    # land on the batch's protein tokens in that order.
    rows = esm.embed(sequences, weights_lib.default_dir('esm2'), 'esm2')
    print(f'ESM2 embeddings {rows.shape}')
    return rows, None

  from alphafold3.model import model_registry
  if len(sequences) != 1:
    # ESM-C wraps multiple chains as [EOS, BOS]-separated with a sequence_id
    # that restricts attention within a chain. That is implemented in the tower
    # but not gated against native, so refuse rather than fold something
    # plausible-looking.
    raise ValueError(
        f'--use_esm_embeddings handles a single protein chain for ESMFold2; '
        f'this input has {len(sequences)}. Fold from an MSA instead.')
  tower = model_registry.ESMFOLD2_VARIANTS[model_name]['esmc']
  print(f'Running the {tower} tower for {model_name}...')
  # default_dir at fp32 on purpose, whatever --weights_precision says: a tower
  # is only ever int8 (converters.esm_lm.QUANT_SCHEME), so it has one directory
  # rather than one per precision.
  hidden = esm.embed(sequences[0], weights_lib.default_dir(tower),
                        'esmc', tower)
  # The shim is the model's own -- every ESMFold2 release trains one, and
  # feeding a variant another's reads corr 0.026 against native.
  shim = esm.load_shim_params(model_runner.model_dir, model_name)
  return None, esm.shim(hidden, shim)


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    *,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    fix_standalone_glycans: bool = False,
    use_esm: bool = False,
    featurise_off: Sequence[str] = (),
    cyclic: bool | Sequence[str] = False,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
  featurisation_start_time = time.time()
  # Decoded: the CCD stores every primed nucleic-acid atom name mmCIF-quoted
  # ("O5'"), and five characters do not fit AF3's four-character atom-name
  # field -- so DNA and RNA fail to featurise with an error that names
  # neither the component nor the atom. A no-op where nothing is quoted.
  ccd = decoded_ccd.get_ccd(user_ccd=fold_input.user_ccd)
  # A model that keeps a modified residue as ONE token needs that decided at
  # featurisation time, not after: it changes how many tokens there are.
  spec = model_registry.get(model_runner.model_name)
  if featurise_off:
    spec = spec.without(featurise_off)
  featurise = functools.partial(
      featurisation.featurise_input,
      fold_input=fold_input,
      buckets=buckets,
      ccd=ccd,
      flatten_non_standard_residues=not spec.featurise.get(
          'modified_as_one_token', False),
      ref_max_modified_date=ref_max_modified_date,
      conformer_max_iterations=conformer_max_iterations,
      resolve_msa_overlaps=resolve_msa_overlaps,
      fix_standalone_glycans=fix_standalone_glycans,
  )
  featurised_examples = featurise(verbose=True)
  # Some families need the INPUT built their way as well as the forward graph;
  # spec.featurise says which, and this is where those conventions land. Stock
  # AlphaFold 3, OpenFold3 and IntelliFold-2 declare none, so for them this runs
  # only if something not tied to a model was asked for -- cyclic chains.
  if featurise_off:
    print(f'Featurisation conventions switched OFF: {", ".join(featurise_off)}')
  if spec.featurise or cyclic:
    has_msa = any(
        getattr(chain, 'unpaired_msa', None) or getattr(chain, 'paired_msa', None)
        for chain in fold_input.chains
    )
    # Once, not once per seed: the examples below differ only by rng seed, and
    # `auto` runs a multi-billion-parameter tower.
    esm_embeddings, lm_pair_array = _resolve_esm(
        use_esm, fold_input, model_runner)
    featurised_examples = [
        model_features.apply(
            example, spec,
            refeaturise=lambda: featurise(verbose=False),
            model_dir=model_runner.model_dir,
            esm=esm_embeddings,
            lm_pair=lm_pair_array,
            has_msa=has_msa,
            fold_input=fold_input,
            cyclic=cyclic,
        )
        for example in featurised_examples
    ]
  print(
      f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - featurisation_start_time:.2f} seconds.'
  )
  print(
      'Running model inference and extracting output structure samples with'
      f' {len(fold_input.rng_seeds)} seed(s)...'
  )
  all_inference_start_time = time.time()
  all_inference_results = []
  # AlphaFold 2's five parameter sets are five trained models, so for af2_* the
  # iteration is over (seed, model) pairs rather than seeds alone. One set is
  # the default and reproduces the old behaviour exactly; the AF3 family has a
  # single set of weights and always takes the 1-element path.
  # AlphaFold 2 ships FIVE parameter sets -- five separately trained models,
  # not five seeds -- and until now inference used the first and ignored the
  # rest. With --af2_num_models > 1 they become the SAMPLES of a seed, which is
  # what the existing ranking already sorts and what keeps the output layout
  # (`seed-N_sample-M`) unchanged; giving each model its own ResultsForSeed
  # would have five entries claiming one seed and colliding on the directory.
  # getattr, because an AF3 runner genuinely has no such property -- one set of
  # weights -- and 1 is the right answer for it. But if af2_num_models was
  # ASKED for and the property is missing, that is a stale library rather than
  # a model with one set, and it must not be silently answered with 1: that is
  # how --af2_num_models=5 ran a single model and said nothing.
  n_sets = getattr(model_runner, 'num_param_sets', 1)
  if int(_AF2_NUM_MODELS.value) > 1 and not hasattr(model_runner,
                                                    'num_param_sets'):
    raise RuntimeError(
        '--af2_num_models > 1 needs a runner that reports num_param_sets; '
        f'{type(model_runner).__name__} does not. On Colab this means the '
        'overlay did not carry af2/inference.py -- check dev/live/overlay.txt.')
  n_models = max(1, min(int(_AF2_NUM_MODELS.value), n_sets))
  if n_models > 1:
    print(f'Using {n_models} of {n_sets} AlphaFold 2 parameter sets; each is a '
          'sample of the seed.')
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    inference_start_time = time.time()
    rng_key = jax.random.PRNGKey(seed)
    inference_results, embeddings, distogram = [], None, None
    for model_index in range(n_models):
      label = (f'seed {seed}' if n_models == 1
               else f'seed {seed}, model {model_index + 1}/{n_models}')
      print(f'Running model inference with {label}...')
      one_start = time.time()
      result = (model_runner.run_inference(example, rng_key, model_index)
                if n_models > 1 else model_runner.run_inference(example, rng_key))
      print(f'Running model inference with {label} took'
            f' {time.time() - one_start:.2f} seconds.')
      part = model_runner.extract_inference_results(
          batch=example, result=result, target_name=fold_input.name)
      inference_results.extend(part)
      if embeddings is None:
        # From the first model only: these describe the input as one model saw
        # it, and five copies of an embedding nothing ranks is just weight.
        num_tokens = len(part[0].metadata['token_chain_ids'])
        embeddings = model_runner.extract_embeddings(
            result=result, num_tokens=num_tokens)
        distogram = model_runner.extract_distogram(
            result=result, num_tokens=num_tokens)
    print(f'Extracting inference results with seed {seed} took'
          f' {time.time() - inference_start_time:.2f} seconds.')
    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
            embeddings=embeddings,
            distogram=distogram,
        )
    )
  print(
      'Running model inference and extracting output structures with'
      f' {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - all_inference_start_time:.2f} seconds.'
  )
  return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: epath.PathLike,
) -> None:
  """Writes the input JSON to the output directory."""
  output_dir = epath.Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / f'{fold_input.sanitised_name()}_data.json'
  print(f'Writing model input JSON to {path}')
  path.write_text(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: epath.PathLike,
    job_name: str,
    *,
    compress_large_output_files: bool = False,
    save_terms_of_use: bool = True,
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  output_terms = model_registry.get(_MODEL.value).output_terms()

  output_dir = epath.Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      sample_dir = output_dir / f'seed-{seed}_sample-{sample_idx}'
      sample_dir.mkdir(parents=True, exist_ok=True)
      post_processing.write_output(
          inference_result=result,
          output_dir=sample_dir,
          name=f'{job_name}_seed-{seed}_sample-{sample_idx}',
          compress=compress_large_output_files,
      )
      ranking_score = float(result.metadata['ranking_score'])
      # WHICH model produced it. The name is already in the sample's mmCIF
      # (`_software.version`, from InferenceResult.model_id), but finding out
      # that model_4 won meant opening five files. For AF3 this is the version
      # string and constant across samples, so the column is only added when
      # the samples actually differ.
      model_tag = getattr(result, 'model_id', b'') or b''
      if isinstance(model_tag, bytes):
        model_tag = model_tag.rstrip(b'\x00').decode('ascii', 'replace')
      ranking_scores.append((seed, sample_idx, ranking_score, model_tag))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

    if embeddings := results_for_seed.embeddings:
      embeddings_dir = output_dir / f'seed-{seed}_embeddings'
      embeddings_dir.mkdir(parents=True, exist_ok=True)
      post_processing.write_embeddings(
          embeddings=embeddings,
          output_dir=embeddings_dir,
          name=f'{job_name}_seed-{seed}',
      )

    if (distogram := results_for_seed.distogram) is not None:
      distogram_dir = output_dir / f'seed-{seed}_distogram'
      distogram_dir.mkdir(parents=True, exist_ok=True)
      distogram_path = distogram_dir / f'{job_name}_seed-{seed}_distogram.npz'
      with io.BytesIO() as bio:
        np.savez_compressed(bio, distogram=distogram.astype(np.float16))
        distogram_path.write_bytes(bio.getvalue())

  if max_ranking_result is not None:  # True iff ranking_scores non-empty.
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        # The output terms of use are the same for all seeds/samples.
        terms_of_use=output_terms if save_terms_of_use else None,
        name=job_name,
        compress=compress_large_output_files,
    )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    ranking_scores_csv_path = output_dir / f'{job_name}_ranking_scores.csv'
    with ranking_scores_csv_path.open('w') as f:
      writer = csv.writer(f)
      tags = {r[3] for r in ranking_scores}
      if len(tags) > 1:
        writer.writerow(['seed', 'sample', 'ranking_score', 'model'])
        writer.writerows(ranking_scores)
      else:
        # One model, so the column would be the same word five times.
        writer.writerow(['seed', 'sample', 'ranking_score'])
        writer.writerows([r[:3] for r in ranking_scores])


def replace_db_dir(
    path_with_db_dir: epath.PathLike, db_dirs: Sequence[str]
) -> str:
  """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
  path_with_db_dir = os.fspath(path_with_db_dir)
  template = string.Template(path_with_db_dir)
  if 'DB_DIR' in template.get_identifiers():
    for db_dir in db_dirs:
      path = template.substitute(DB_DIR=db_dir)
      if epath.Path(path).exists():
        return path
    raise FileNotFoundError(
        f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
    )
  if (sharded_paths := shards.get_sharded_paths(path_with_db_dir)) is not None:
    db_exists = all(epath.Path(p).exists() for p in sharded_paths)
  else:
    db_exists = epath.Path(path_with_db_dir).exists()
  if not db_exists:
    raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
  return path_with_db_dir


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    *,
    model_runner: None,
    output_dir: epath.PathLike,
    buckets: Sequence[int] | None = None,
    use_esm: bool = False,
    featurise_off: Sequence[str] = (),
    cyclic: bool | Sequence[str] = False,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    fix_standalone_glycans: bool = False,
    force_output_dir: bool = False,
    compress_large_output_files: bool = False,
    save_terms_of_use: bool = True,
) -> folding_input.Input:
  ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    *,
    model_runner: ModelRunner,
    output_dir: epath.PathLike,
    buckets: Sequence[int] | None = None,
    use_esm: bool = False,
    featurise_off: Sequence[str] = (),
    cyclic: bool | Sequence[str] = False,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    fix_standalone_glycans: bool = False,
    force_output_dir: bool = False,
    compress_large_output_files: bool = False,
    save_terms_of_use: bool = True,
) -> Sequence[ResultsForSeed]:
  ...


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    *,
    model_runner: ModelRunner | None,
    output_dir: epath.PathLike,
    buckets: Sequence[int] | None = None,
    use_esm: bool = False,
    featurise_off: Sequence[str] = (),
    cyclic: bool | Sequence[str] = False,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    fix_standalone_glycans: bool = False,
    force_output_dir: bool = False,
    compress_large_output_files: bool = False,
    save_terms_of_use: bool = True,
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input.

  Args:
    fold_input: Fold input to process.
    data_pipeline_config: Data pipeline config to use. If None, skip the data
      pipeline.
    model_runner: Model runner to use. If None, skip inference.
    output_dir: Output directory to write to.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.
    ref_max_modified_date: Optional maximum date that controls whether to allow
      use of model coordinates for a chemical component from the CCD if RDKit
      conformer generation fails and the component does not have ideal
      coordinates set. Only for components that have been released before this
      date the model coordinates can be used as a fallback.
    conformer_max_iterations: Optional override for maximum number of iterations
      to run for RDKit conformer search.
    resolve_msa_overlaps: Whether to deduplicate unpaired MSA against paired
      MSA. The default behaviour matches the method described in the AlphaFold 3
      paper. Set this to false if providing custom paired MSA using the unpaired
      MSA field to keep it exactly as is as deduplication against the paired MSA
      could break the manually crafted pairing between MSA sequences.
    fix_standalone_glycans: If True, standalone glycans are preserved when
      filter_leaving_atoms is True. This is False by default to match the
      AlphaFold 3 paper. Note that the model has been trained with the default
      setting, so setting this to True may cause non-standard behaviour of the
      model.
    force_output_dir: If True, do not create a new output directory even if the
      existing one is non-empty. Instead use the existing output directory and
      potentially overwrite existing files. If False, create a new timestamped
      output directory instead if the existing one is non-empty.
    compress_large_output_files: If True, compress large output files (mmCIF and
      confidences JSON) using zstandard.
    save_terms_of_use: If True, write the terms of use to the output directory.

  Returns:
    The processed fold input, or the inference results for each seed.

  Raises:
    ValueError: If the fold input has no chains.
  """
  print(f'\nRunning fold job {fold_input.name}...')

  if not fold_input.chains:
    raise ValueError('Fold input has no chains.')

  output_dir = epath.Path(output_dir)
  if not force_output_dir and output_dir.exists() and any(output_dir.iterdir()):
    new_output_dir = (
        output_dir.parent
        / f'{output_dir.name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    print(
        f'Output will be written in {new_output_dir} since {output_dir} is'
        ' non-empty.'
    )
    output_dir = new_output_dir
  else:
    print(f'Output will be written in {output_dir}')

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
  else:
    print('Running data pipeline...')
    fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

  if _USE_MSA_SERVER.value:
    from alphafold3.data import msa_server as _msa_server
    _msa_server.save_msas(fold_input, output_dir)

  write_fold_input_json(fold_input, output_dir)
  if model_runner is None:
    print('Skipping model inference...')
    output = fold_input
  else:
    print(
        f'Predicting 3D structure for {fold_input.name} with'
        f' {len(fold_input.rng_seeds)} seed(s)...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        buckets=buckets,
        use_esm=use_esm,
        featurise_off=featurise_off,
        cyclic=cyclic,
        ref_max_modified_date=ref_max_modified_date,
        conformer_max_iterations=conformer_max_iterations,
        resolve_msa_overlaps=resolve_msa_overlaps,
        fix_standalone_glycans=fix_standalone_glycans,
    )
    print(f'Writing outputs with {len(fold_input.rng_seeds)} seed(s)...')
    write_outputs(
        all_inference_results=all_inference_results,
        output_dir=output_dir,
        job_name=fold_input.sanitised_name(),
        compress_large_output_files=compress_large_output_files,
        save_terms_of_use=save_terms_of_use,
    )
    output = all_inference_results

  print(f'Fold job {fold_input.name} done, output written to {output_dir}\n')
  return output


def main(_):
  # Suppress tokamax noise: autotuning cache-miss spam and CPU-fallback errors
  # for gated_linear_unit (tokamax logs at ERROR but handles the fallback).
  # tokamax uses absl logging, which has its own handler that doesn't propagate
  # to the root logger — so we attach the filter to both.
  class _SuppressTokamax(logging.Filter):
    def filter(self, record):
      msg = record.getMessage()
      return 'Autotuning cache miss' not in msg and 'Failed to run implementation' not in msg
  _tokamax_filter = _SuppressTokamax()
  logging.getLogger().addFilter(_tokamax_filter)
  logging.getLogger('absl').addFilter(_tokamax_filter)
  try:
    absl_logging.get_absl_handler().addFilter(_tokamax_filter)
  except Exception:
    pass

  # Suppress int64-truncation UserWarning from featurization (expected without
  # JAX_ENABLE_X64).
  warnings.filterwarnings('ignore', message='.*int64.*')

  # Reduce absl verbosity so pipeline INFO logs (bucket sizes etc.) are hidden.
  absl_logging.set_verbosity(absl_logging.WARNING)

  if _CACHE_DIR.value is not None:
    _jax_cache = os.path.join(_CACHE_DIR.value, 'jax')
    os.makedirs(_jax_cache, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', _jax_cache)
    # Deliberately NOT lowering jax_persistent_cache_min_{entry_size_bytes,
    # compile_time_secs}. Tried on 6MRR/openbind0: it grows the cache 3.5 -> 4.1 MB
    # and changes a warm start by 0.04 s (16.35 -> 16.31), because what the
    # defaults exclude is not where the time is.
    #
    # Where it IS, measured cold vs warm across two processes with this cache:
    #   first inference   69.09 s  ->  16.35 s     (the cache is worth ~53 s)
    #   later inference    5.45 s  ->   5.44 s
    # The 10.8 s that survives is TRACING AND LOWERING, which the persistent
    # cache cannot skip -- it stores the compiled executable, but JAX still has
    # to trace and lower the graph to know which entry to fetch.

  if _JSON_PATH.value is None == _INPUT_DIR.value is None:
    raise ValueError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
    raise ValueError(
        'At least one of --run_inference or --run_data_pipeline must be'
        ' set to true.'
    )

  if _INPUT_DIR.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_dir(_INPUT_DIR.value)
  elif _JSON_PATH.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_path(_JSON_PATH.value)
  else:
    raise AssertionError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )
  # MATERIALISED so the inputs can be inspected before the model is built:
  # AlphaFold 2 has to know whether any input carries a template, because that
  # decides the graph (template embedder on) and the parameter sets (the
  # template-trained 1 and 2 rather than 1-5). Fold inputs are JSON specs, so
  # this holds specs, not features.
  fold_inputs = list(fold_inputs)

  if _OUTPUT_DIR.value is None:
    raise ValueError('Output directory must be specified with --output_dir.')

  # Make sure we can create the output directory before running anything.
  try:
    _OUTPUT_DIR.value.mkdir(parents=True, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
    raise

  jax_backend = _resolve_jax_backend() if _RUN_INFERENCE.value else None
  if _RUN_INFERENCE.value and _FLASH_ATTENTION_IMPLEMENTATION.value == 'auto':
    # Resolved HERE, after the backend is known and before the validation
    # below, so an auto run lands on a value those checks already accept.
    # `flags.FLAGS.<name> = v`, NOT `_HOLDER.value = v`: a FlagHolder's value
    # is a read-only property, so the assignment form raises AttributeError.
    # The CPU fallback below has always been written the second way, i.e. it
    # would have raised the moment it ran.
    if jax_backend == JaxBackend.CPU:
      flags.FLAGS.flash_attention_implementation = 'xla'
      print('--flash_attention_implementation=auto -> xla (no GPU)')
    else:
      from alphafold3.model.components import platform as _platform
      _picked = _platform.attention_config()
      flags.FLAGS.flash_attention_implementation = _picked['attention']
      print(f'--flash_attention_implementation=auto -> '
            f'{_picked["attention"]}: {_picked["reason"]}')
      # The GLU is a SEPARATE choice from the attention one (an A100 wants
      # Triton attention with a Pallas GLU), so the probe answers both. Only
      # fill it in when the user left --glu_kernel alone.
      if _GLU_KERNEL.value == 'auto' and _picked.get('glu'):
        flags.FLAGS.glu_kernel = _picked['glu']
        print(f'--glu_kernel=auto -> {_picked["glu"]}')
  if _RUN_INFERENCE.value:
    # Fail early on incompatible devices, but only if we're running inference.
    if jax_backend == JaxBackend.CPU:
      if _FLASH_ATTENTION_IMPLEMENTATION.value != 'xla':
        if _JAX_BACKEND.value == JaxBackend.AUTO:
          # The CPU was auto-detected rather than asked for, so pick the only
          # flash attention implementation that can run on it.
          print(
              'Setting --flash_attention_implementation=xla, required for'
              ' CPU-only inference.'
          )
          flags.FLAGS.flash_attention_implementation = 'xla'
        else:
          raise ValueError(
              'For CPU-only inference, the --flash_attention_implementation'
              ' must be set to "xla".'
          )
    elif jax_backend == JaxBackend.TPU:
      # Nothing to validate: the compute-capability rules below are CUDA's, and
      # the only attention implementation with a TPU backend is XLA, which
      # --flash_attention_implementation=auto already picks.
      pass
    elif jax_backend == JaxBackend.GPU:
      gpu_devices = jax.local_devices(backend='gpu')
      if gpu_devices:
        compute_capability = float(
            gpu_devices[_GPU_DEVICE.value].compute_capability
        )
        if compute_capability < 6.0:
          raise ValueError(
              'AlphaFold 3 requires at least GPU compute capability 6.0 (see'
              ' https://developer.nvidia.com/cuda-gpus).'
          )
        elif 7.0 <= compute_capability < 8.0:
          xla_flags = os.environ.get('XLA_FLAGS')
          required_flag = (
              '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
          )
          if not xla_flags or required_flag not in xla_flags:
            raise ValueError(
                'For devices with GPU compute capability 7.x (see'
                ' https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS'
                f' must include "{required_flag}".'
            )
          # 'volta' belongs here too: colabfold-legacy-kernels is the ONE
          # fused attention a 7.x card can run, and this check predates it.
          # Triton and cuDNN still cannot, so they still raise.
          if _FLASH_ATTENTION_IMPLEMENTATION.value not in ('xla', 'volta'):
            raise ValueError(
                'For devices with GPU compute capability 7.x (see'
                ' https://developer.nvidia.com/cuda-gpus) the'
                ' --flash_attention_implementation must be "xla", or "volta"'
                ' with colabfold-legacy-kernels installed.'
            )
    else:
      raise ValueError(f'Unsupported JAX backend: {jax_backend}')

  max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)
  if _RUN_DATA_PIPELINE.value:
    expand_path = lambda x: replace_db_dir(x, DB_DIR.value)
    data_pipeline_config = pipeline.DataPipelineConfig(
        jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,  # pyrefly: ignore[bad-argument-type]
        nhmmer_binary_path=_NHMMER_BINARY_PATH.value,  # pyrefly: ignore[bad-argument-type]
        hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,  # pyrefly: ignore[bad-argument-type]
        hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,  # pyrefly: ignore[bad-argument-type]
        hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,  # pyrefly: ignore[bad-argument-type]
        small_bfd_database_path=expand_path(_SMALL_BFD_DATABASE_PATH.value),
        small_bfd_z_value=_SMALL_BFD_Z_VALUE.value,
        mgnify_database_path=expand_path(_MGNIFY_DATABASE_PATH.value),
        mgnify_z_value=_MGNIFY_Z_VALUE.value,
        uniprot_cluster_annot_database_path=expand_path(
            _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
        ),
        uniprot_cluster_annot_z_value=_UNIPROT_CLUSTER_ANNOT_Z_VALUE.value,
        uniref90_database_path=expand_path(_UNIREF90_DATABASE_PATH.value),
        uniref90_z_value=_UNIREF90_Z_VALUE.value,
        ntrna_database_path=expand_path(_NTRNA_DATABASE_PATH.value),
        ntrna_z_value=_NTRNA_Z_VALUE.value,
        rfam_database_path=expand_path(_RFAM_DATABASE_PATH.value),
        rfam_z_value=_RFAM_Z_VALUE.value,
        rna_central_database_path=expand_path(_RNA_CENTRAL_DATABASE_PATH.value),
        rna_central_z_value=_RNA_CENTRAL_Z_VALUE.value,
        pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
        seqres_database_path=expand_path(_SEQRES_DATABASE_PATH.value),
        jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
        jackhmmer_max_parallel_shards=_JACKHMMER_MAX_PARALLEL_SHARDS.value,
        nhmmer_n_cpu=_NHMMER_N_CPU.value,
        nhmmer_max_parallel_shards=_NHMMER_MAX_PARALLEL_SHARDS.value,
        max_template_date=max_template_date,
    )
  else:
    data_pipeline_config = None

  model_name = _MODEL.value
  # chain ids -> asym_ids, which are 1-based and assigned in chain order.
  cyclic = _CYCLIC.value or False
  if cyclic and [c.lower() for c in cyclic] == ['all']:
    cyclic = True
  use_esm = _USE_ESM_EMBEDDINGS.value
  wants_esm = model_name == 'chai1' or model_name in model_config.ESMFOLD2_FAMILY
  if use_esm and not wants_esm:
    raise ValueError(
        f'--use_esm_embeddings is for chai-1 and the ESMFold2 family; '
        f'{model_name} does not read a language model')
  if not use_esm and wants_esm:
    # Silent until now, and the cost is not subtle for either family: see the
    # flag's help.
    print('WARNING: %s is running WITHOUT its language model, which is part of'
          ' the model rather than an extra; see --use_esm_embeddings.'
          % model_name)
  if MODEL_DIR.value is not None:
    model_dir = MODEL_DIR.value
  elif model_name == 'alphafold3':
    model_dir = _DEFAULT_MODEL_DIR
  else:
    model_dir = None  # resolved by ensure_weights below

  # The terms-of-use notice is DeepMind's, about DeepMind's parameters. It does
  # not apply when running someone else's model through this graph.
  if model_name == 'alphafold3':
    notice = textwrap.wrap(
        'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
        ' parameters are only available under terms of use provided at'
        ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
        ' If you do not agree to these terms and are using AlphaFold 3 derived'
        ' model parameters, cancel execution of AlphaFold 3 inference with'
        ' CTRL-C, and do not use the model parameters.',
        break_long_words=False,
        break_on_hyphens=False,
        width=80,
    )
    print('\n' + '\n'.join(notice) + '\n')

  if _RUN_INFERENCE.value:
    devices = jax.local_devices(backend=jax_backend)
    if jax_backend == JaxBackend.CPU:
      device = devices[0]
      print(f'Found local CPU devices: {devices}, using device 0: {device}')
    elif jax_backend == JaxBackend.TPU:
      device = devices[0]
      print(f'Found local TPU devices: {devices}, using device 0: {device}')
    elif jax_backend == JaxBackend.GPU:
      print(
          f'Found local GPU devices: {devices}, using device '
          f'{_GPU_DEVICE.value}: {devices[_GPU_DEVICE.value]}'
      )
      device = devices[_GPU_DEVICE.value]
    else:
      raise ValueError(f'Unsupported JAX backend: {jax_backend}')

    spec = model_registry.get(model_name)
    if spec.engine == 'af2':
      # AlphaFold 2's parameters are DeepMind's own release under CC BY 4.0, so
      # they are fetched FROM SOURCE rather than republished here -- point
      # --model_dir at your own copy and this is a no-op.
      from alphafold3.af2 import inference as af2_inference

      model_dir = weights.ensure_af2_params(
          model_dir, download=_DOWNLOAD_WEIGHTS.value)
      # Templates need the template-trained parameter sets and the template
      # embedder switched on, so this is decided by the INPUT rather than a flag.
      af2_templates = any(getattr(c, 'templates', None)
                          for fi in fold_inputs for c in fi.chains)
      print('Building AlphaFold 2 from scratch...')
      model_runner = af2_inference.AF2ModelRunner(
          spec, device=device, model_dir=model_dir,
          num_recycles=_NUM_RECYCLES.value,
          use_templates=af2_templates,
          use_dropout=_DROPOUT.value,
      )
    else:
      # Idempotent after the first run: a directory that already holds a blob is
      # left alone. AlphaFold 3's own parameters are never fetched.
      model_dir = weights.ensure_weights(
          model_name, model_dir, download=_DOWNLOAD_WEIGHTS.value,
          precision=_WEIGHTS_PRECISION.value)

      print('Building model from scratch...')
      model_runner = ModelRunner(
          config=make_model_config(
              flash_attention_implementation=typing.cast(
                  tokamax.DotProductAttentionImplementation,
                  _FLASH_ATTENTION_IMPLEMENTATION.value,
              ),
              glu_kernel=_GLU_KERNEL.value,
              num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
              num_sampling_steps=_NUM_SAMPLING_STEPS.value,
              num_recycles=_NUM_RECYCLES.value,
              return_embeddings=_SAVE_EMBEDDINGS.value,
              return_distogram=_SAVE_DISTOGRAM.value,
              model_name=model_name,
              num_msa=_NUM_MSA.value,
          ),
          device=device,
          model_dir=model_dir,
          use_dropout=_DROPOUT.value,
      )
    # Check we can load the model parameters before launching anything.
    print('Checking that model parameters can be loaded...')
    _ = model_runner.model_params
    if _PRECOMPILE.value:
      _precompile(model_runner, model_name, model_dir,
                  [int(n) for n in _PRECOMPILE.value])
      return 0
  else:
    model_runner = None

  num_fold_inputs = 0
  for fold_input in fold_inputs:
    if _NUM_SEEDS.value is not None:
      print(f'Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds')
      fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
    if _USE_MSA_SERVER.value:
      from alphafold3.data import msa_server as _msa_server
      fold_input = _msa_server.fill_missing_msas(
          fold_input,
          host_url=_MSA_SERVER_URL.value,
          user_agent=_MSA_SERVER_USER_AGENT.value,
      )
    process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=data_pipeline_config,
        model_runner=model_runner,
        output_dir=_OUTPUT_DIR.value / fold_input.sanitised_name(),
        buckets=None
        if _NOJIT.value
        else tuple(int(bucket) for bucket in _BUCKETS.value),
        use_esm=use_esm,
        featurise_off=_FEATURISE_OFF.value,
        cyclic=cyclic,
        ref_max_modified_date=max_template_date,
        conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
        resolve_msa_overlaps=_RESOLVE_MSA_OVERLAPS.value,
        fix_standalone_glycans=_FIX_STANDALONE_GLYCANS.value,
        force_output_dir=_FORCE_OUTPUT_DIR.value,
        compress_large_output_files=_COMPRESS_LARGE_OUTPUT_FILES.value,
        save_terms_of_use=_SAVE_TERMS_OF_USE.value,
    )
    num_fold_inputs += 1

  print(f'Done running {num_fold_inputs} fold jobs.')


def run():
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)


if __name__ == '__main__':
  run()
