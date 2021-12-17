#pragma once

#include <vector>

#include "lib/jxl/base/padded_bytes.h"

namespace research {

struct EncodedImages {
  // data に含まれる画像のインデックス
  std::vector<size_t> image_indices;
  jxl::PaddedBytes data;
  // 書き込まれたビット数
  size_t n_bits;
};

}  // namespace research
