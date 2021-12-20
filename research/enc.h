#pragma once

#include <optional>
#include <vector>

#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/modular/transform/transform.h"

namespace research {

struct EncodedImages {
  // data に含まれる画像のインデックス
  std::vector<size_t> image_indices;
  std::vector<std::shared_ptr<const jxl::Image>> included_images;
  jxl::PaddedBytes data;
  // 書き込まれたビット数
  size_t n_bits;
};

struct ImagesHeader : public jxl::Fields {
  ImagesHeader() { jxl::Bundle::Init(this); }

  const char* Name() const override { return "research::ImagesHeader"; }

  jxl::Status VisitFields(jxl::Visitor* JXL_RESTRICT visitor) override {
    uint32_t num_transforms = static_cast<uint32_t>(transforms.size());
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->U32(jxl::Val(0), jxl::Val(1), jxl::BitsOffset(4, 2),
                     jxl::BitsOffset(8, 18), 0, &num_transforms));
    if (visitor->IsReading()) transforms.resize(num_transforms);
    for (size_t i = 0; i < num_transforms; i++) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&transforms[i]));
    }
    return true;
  }

  std::vector<jxl::Transform> transforms;
};

}  // namespace research
