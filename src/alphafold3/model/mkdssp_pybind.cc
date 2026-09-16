// Copyright 2024 DeepMind Technologies Limited
//
// AlphaFold 3 source code is licensed under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with the
// License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// To request access to the AlphaFold 3 model parameters, follow the process set
// out at https://github.com/google-deepmind/alphafold3. You may only use these
// if received directly from Google. Use is subject to terms of use available at
// https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

#include "alphafold3/model/mkdssp_pybind.h"

#include <filesystem>

#include <cif++/file.hpp>
#include <cif++/pdb.hpp>
#include <dssp.hpp>
#include <sstream>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace alphafold3 {
namespace py = pybind11;

void RegisterModuleMkdssp(pybind11::module m) {
  // THE CHECK IS DEFERRED TO THE CALL, not done at registration.
  //
  // This used to `throw` here when no components.cif could be found, which
  // failed the import of the WHOLE `alphafold3.cpp` module -- including
  // `cif_dict`, which featurisation needs -- because DSSP's registration could
  // not find a dictionary. DSSP is used in exactly one place
  // (`confidences.predicted_disorder`, the AlphaFold-RSA disorder metric) and
  // nothing in the fold path touches it, so an absent dictionary made the
  // package unusable for a reason that had nothing to do with what was being
  // run. `pip install alphafold3-colabfold` then died with
  // "Could not find the libcifpp components.cif file" on `from
  // alphafold3.common import folding_input`.
  //
  // So: look for the dictionary, remember whether it was found, and raise only
  // if someone actually calls get_dssp.
  bool have_data = getenv("LIBCIFPP_DATA_DIR") != nullptr;
  if (!have_data) {
    py::module site = py::module::import("site");
    py::list paths = py::cast<py::list>(site.attr("getsitepackages")());
    // Find the first path that contains the libcifpp components.cif file.
    for (const auto& py_path : paths) {
      auto path_str =
          std::filesystem::path(py::cast<absl::string_view>(py_path)) /
          "share/libcifpp/components.cif";
      if (std::filesystem::exists(path_str)) {
        setenv("LIBCIFPP_DATA_DIR", path_str.parent_path().c_str(), 0);
        have_data = true;
        break;
      }
    }
  }
  m.def(
      "get_dssp",
      [have_data](absl::string_view mmcif, int model_no,
         int min_poly_proline_stretch_length,
         bool calculate_surface_accessibility) {
        if (!have_data) {
          throw py::value_error(
              "get_dssp needs libcifpp's components.cif, which was not found. "
              "Set LIBCIFPP_DATA_DIR to a directory containing it (the wwPDB "
              "CCD, ~518 MB). Only DSSP-derived outputs need it -- folding "
              "does not.");
        }
        cif::file cif_file(mmcif.data(), mmcif.size());
        dssp result(cif_file.front(), model_no, min_poly_proline_stretch_length,
                    calculate_surface_accessibility);
        std::stringstream sstream;
        result.write_legacy_output(sstream);
        return sstream.str();
      },
      py::arg("mmcif"), py::arg("model_no") = 1,
      py::arg("min_poly_proline_stretch_length") = 3,
      py::arg("calculate_surface_accessibility") = false,
      py::doc("Gets secondary structure from an mmCIF file."));
}
}  // namespace alphafold3
