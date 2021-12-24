#pragma once

#include <vector>

#include "lib/jxl/fields.h"
#include "lib/jxl/modular/transform/transform.h"

namespace research {

struct CombinedImageInfo : public jxl::Fields {
  CombinedImageInfo(uint32_t width, uint32_t height, uint32_t n_channel,
                    bool flif_enabled)
      : width_(width),
        height_(height),
        n_channel_(n_channel),
        flif_enabled_(flif_enabled) {
    jxl::Bundle::Init(this);
  }

#if JXL_IS_DEBUG_BUILD
  const char* Name() const override { return "research::CombinedImageInfo"; }
#endif

  jxl::Status VisitFields(jxl::Visitor* JXL_RESTRICT visitor) override;

  uint32_t n_images;
  uint32_t n_bytes;
  uint32_t n_flif_bytes;

 private:
  uint32_t width_;
  uint32_t height_;
  uint32_t n_channel_;
  bool flif_enabled_;
};

struct ClusterHeader : public jxl::Fields {
  ClusterHeader(uint32_t width, uint32_t height, uint32_t n_channel,
                bool flif_enabled)
      : width_(width),
        height_(height),
        n_channel_(n_channel),
        flif_enabled_(flif_enabled) {
    jxl::Bundle::Init(this);
  }

#if JXL_IS_DEBUG_BUILD
  const char* Name() const override { return "research::ClusterHeader"; }
#endif

  jxl::Status VisitFields(jxl::Visitor* JXL_RESTRICT visitor) override;

  std::vector<CombinedImageInfo> combined_images;

  // 元のi番目の画像は、combined_imagesの何番目の画像か
  std::vector<uint32_t> pointers;

 private:
  uint32_t width_;
  uint32_t height_;
  uint32_t n_channel_;
  bool flif_enabled_;
};

}  // namespace research
