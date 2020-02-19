// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef JXL_DEC_PARAMS_H_
#define JXL_DEC_PARAMS_H_

// Parameters and flags that govern JXL decompression.

#include <stddef.h>
#include <stdint.h>

#include <limits>

#include "jxl/base/override.h"

namespace jxl {

struct BrunsliDecoderOptions {
  bool fix_dc_staircase = false;
  bool gaborish = false;
};

struct DecompressParams {
  // If true, checks at the end of decoding that all of the compressed data
  // was consumed by the decoder.
  bool check_decompressed_size = true;

  // If true, skip dequant and iDCT and decode to JPEG (only if possible)
  bool keep_dct = false;

  // These cannot be kOn because they need encoder support.
  Override preview = Override::kDefault;
  Override noise = Override::kDefault;
  Override adaptive_reconstruction = Override::kDefault;

  // How many passes to decode at most. By default, decode everything.
  uint32_t max_passes = std::numeric_limits<uint32_t>::max();
  // Alternatively, one can specify the maximum tolerable downscaling factor
  // with respect to the full size of the image. By default, nothing less than
  // the full size is requested.
  size_t max_downsampling = 1;

  BrunsliDecoderOptions brunsli;
};

}  // namespace jxl

#endif  // JXL_DEC_PARAMS_H_