#pragma once

#include "fields.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/modular/modular_image.h"

namespace research {

class ClusterFileReader {
 public:
  ClusterFileReader(uint32_t width, uint32_t height, uint32_t n_channel,
                    jxl::Span<const uint8_t> data);

  const ClusterHeader& header() const noexcept { return header_; }

  size_t n_images() const noexcept { return header_.pointers.size(); }

  std::vector<jxl::Image> ReadAll();

  jxl::Image Read(size_t idx);

 private:
  uint32_t width_;
  uint32_t height_;
  uint32_t n_channel_;
  // ヘッダーを含まないバイト列
  jxl::Span<const uint8_t> data_;
  ClusterHeader header_;
};

}  // namespace research
