#pragma once

#include "common_cluster.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/modular/modular_image.h"

namespace research {

struct DecodingOptions {
  uint32_t width;
  uint32_t height;
  jxl::MultiOptions multi_options;
  int refchan;
  bool flif_enabled;
  int flif_additional_props;
};

class ClusterFileReader {
 public:
  ClusterFileReader(const DecodingOptions& options,
                    jxl::Span<const uint8_t> data);

  const ClusterHeader& header() const noexcept { return header_; }

  size_t n_images() const noexcept { return header_.pointers.size(); }

  jxl::Status ReadAll(std::vector<jxl::Image>& out_images);

  jxl::Status Read(size_t idx, jxl::Image& out_image);

 private:
  const DecodingOptions& options_;
  // ヘッダーを含まないバイト列
  jxl::Span<const uint8_t> data_;
  ClusterHeader header_;
};

}  // namespace research
