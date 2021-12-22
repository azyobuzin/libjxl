#pragma once

#include <vector>

#include "lib/jxl/fields.h"
#include "lib/jxl/modular/transform/transform.h"

namespace research {

// パレット変換情報を CombinedImage ごとに持つ
// enc_without_headerの実験結果から、パレット変換はあまり意味がなさそうなので、使わない方向で
struct CombinedImageHeader : public jxl::Fields {
  CombinedImageHeader() { jxl::Bundle::Init(this); }

#if JXL_IS_DEBUG_BUILD
  const char* Name() const override { return "research::CombinedImageHeader"; }
#endif

  jxl::Status VisitFields(jxl::Visitor* JXL_RESTRICT visitor) override;

  std::vector<jxl::Transform> transforms;
};

// 全体の情報を格納するために使う予定
struct ImageInfo : public jxl::Fields {
  ImageInfo() { jxl::Bundle::Init(this); }

#if JXL_IS_DEBUG_BUILD
  const char* Name() const override { return "research::ImageInfo"; }
#endif

  jxl::Status VisitFields(jxl::Visitor* JXL_RESTRICT visitor) override;

  uint32_t width;
  uint32_t height;
  uint32_t n_channel;
};

struct CombinedImageInfo : public jxl::Fields {
  CombinedImageInfo(uint32_t width, uint32_t height, uint32_t n_channel)
      : width_(width), height_(height), n_channel_(n_channel) {
    jxl::Bundle::Init(this);
  }

#if JXL_IS_DEBUG_BUILD
  const char* Name() const override { return "research::CombinedImageInfo"; }
#endif

  jxl::Status VisitFields(jxl::Visitor* JXL_RESTRICT visitor) override;

  uint32_t n_images;
  uint32_t n_bytes;

 private:
  uint32_t width_;
  uint32_t height_;
  uint32_t n_channel_;
};

struct ClusterHeader : public jxl::Fields {
  ClusterHeader(uint32_t width, uint32_t height, uint32_t n_channel)
      : width_(width), height_(height), n_channel_(n_channel) {
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
};

}  // namespace research
