#pragma once

#include "fields.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/modular/modular_image.h"

namespace research {

class ClusterFileReader {
 public:
  ClusterFileReader(uint32_t width, uint32_t height, uint32_t n_channel,
                    size_t max_refs, jxl::Span<const uint8_t> data);

  const ClusterHeader& header() const noexcept { return header_; }

  size_t n_images() const noexcept { return header_.pointers.size(); }

  jxl::Status ReadAll(std::vector<jxl::Image>& out_images);

  jxl::Status Read(size_t idx, jxl::Image& out_image);

 private:
  uint32_t width_;
  uint32_t height_;
  jxl::MultiOptions multi_options_;
  // ヘッダーを含まないバイト列
  jxl::Span<const uint8_t> data_;
  ClusterHeader header_;
};

}  // namespace research
