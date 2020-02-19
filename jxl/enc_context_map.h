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

#ifndef JXL_ENC_CONTEXT_MAP_H_
#define JXL_ENC_CONTEXT_MAP_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "jxl/enc_bit_writer.h"

namespace jxl {

// Context map uses uint8_t.
constexpr size_t kMaxClusters = 256;
// Max limit is 255 because encoding assumes numbers < 255
// More clusters can help compression, but makes encode/decode somewhat slower
static const size_t kClustersLimit = 128;

// Encodes the given context map to the bit stream. The number of different
// histogram ids is given by num_histograms.
void EncodeContextMap(const std::vector<uint8_t>& context_map,
                      size_t num_histograms,
                      const BitWriter::Allotment& allotment, BitWriter* writer);

}  // namespace jxl

#endif  // JXL_ENC_CONTEXT_MAP_H_