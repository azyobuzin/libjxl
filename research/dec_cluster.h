#pragma once

#include "common_cluster.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/modular/modular_image.h"

namespace research {

struct DecodingOptions {
  uint32_t width;
  uint32_t height;
  uint32_t n_channel;
  jxl::ParentReferenceType reference_type;
  bool flif_enabled;
  int flif_additional_props;
};

class ClusterFileReader {
 public:
  ClusterFileReader(const DecodingOptions& options,
                    jxl::Span<const uint8_t> data);

  const ClusterHeader& header() const noexcept { return header_; }

  uint32_t n_images() const noexcept {
    return static_cast<uint32_t>(pointers_.size());
  }

  jxl::Status ReadAll(std::vector<jxl::Image>& out_images);

  jxl::Status Read(uint32_t idx, jxl::Image& out_image);

 private:
  const DecodingOptions& options_;
  // ヘッダーを含まないバイト列
  jxl::Span<const uint8_t> data_;
  ClusterHeader header_;
  std::vector<uint32_t> pointers_;
  std::vector<std::vector<uint32_t>> references_;
};

void DecodeClusterPointers(jxl::BitReader& reader,
                           std::vector<uint32_t>& pointers);

void DecodeReferences(jxl::BitReader& reader,
                      std::vector<uint32_t>& references);

}  // namespace research
