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

"""An implementation of the inference pipeline of AlphaFold 3."""

# libcifpp refuses to load without a components.cif, and the FULL dictionary is
# 518 MB raw / 120 MB zipped -- 92% of a wheel that is otherwise 10 MB. So a
# minimal one (the standard residues) ships at constants/libcifpp and is pointed
# at HERE, before anything imports alphafold3.cpp and the extension loads.
#
# Set LIBCIFPP_DATA_DIR yourself to use the full dictionary; `setdefault` means
# an existing value always wins. Anything beyond the standard residues comes
# from files.rcsb.org via `constants.ccd_fetch`, or from the input's userCCD.
import os as _os

_bundled = _os.path.join(_os.path.dirname(__file__), 'constants', 'libcifpp')
if _os.path.exists(_os.path.join(_bundled, 'components.cif')):
  _os.environ.setdefault('LIBCIFPP_DATA_DIR', _bundled)
del _os, _bundled
