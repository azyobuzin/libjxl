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

  JXL_FIELDS_NAME(research::CombinedImageInfo)

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
        flif_enabled_(flif_enabled) {}

  JXL_FIELDS_NAME(research::ClusterHeader)

  jxl::Status VisitFields(jxl::Visitor* JXL_RESTRICT visitor) override;

  std::vector<CombinedImageInfo> combined_images;

 private:
  uint32_t width_;
  uint32_t height_;
  uint32_t n_channel_;
  bool flif_enabled_;
};

struct IndexFields : public jxl::Fields {
  IndexFields() { jxl::Bundle::Init(this); }

  JXL_FIELDS_NAME(research::IndexFields)

  jxl::Status VisitFields(jxl::Visitor* JXL_RESTRICT visitor) override;

  uint32_t width;
  uint32_t height;
  uint32_t n_channel;
  uint32_t n_clusters;

  // i番目の画像がどのクラスタにあるか
  std::vector<uint32_t> assignments;
};

}  // namespace research
