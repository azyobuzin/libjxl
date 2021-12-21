#include "fields.h"

#include <limits>
#include <vector>

namespace research {

namespace {

constexpr int bit_width(unsigned int x) noexcept {
  // https://github.com/boostorg/core/blob/df3b9827cfc9f38080c8d90af048f1f20c449c36/include/boost/core/bit.hpp#L449-L453
  int countl =
      x ? __builtin_clz(x) : std::numeric_limits<unsigned int>().digits;
  return std::numeric_limits<unsigned int>().digits - countl;
}

}  // namespace

jxl::Status CombinedImageHeader::VisitFields(
    jxl::Visitor* JXL_RESTRICT visitor) {
  // https://github.com/libjxl/libjxl/blob/1d62c5fc07ec2bcb4baed46cc4b6e4611c714602/lib/jxl/modular/encoding/encoding.h#L35-L42
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

jxl::Status ImageInfo::VisitFields(jxl::Visitor* JXL_RESTRICT visitor) {
  // https://github.com/libjxl/libjxl/blob/1d62c5fc07ec2bcb4baed46cc4b6e4611c714602/lib/jxl/frame_header.cc#L259-L260
  const jxl::U32Enc enc(
      jxl::BitsOffset(8, 1), jxl::BitsOffset(11, 1 + (1 << 8)),
      jxl::BitsOffset(14, 1 + (1 << 8) + (1 << 11)),
      jxl::BitsOffset(30, 1 + (1 << 8) + (1 << 11) + (1 << 14)));
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(enc, 1, &width));
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(enc, 1, &height));
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(jxl::BitsOffset(1, 1), jxl::Val(3),
                                         jxl::Val(4), jxl::BitsOffset(2, 5), 1,
                                         &n_channel));
  return true;
}

jxl::Status CombinedImageInfo::VisitFields(jxl::Visitor* JXL_RESTRICT visitor) {
  // TODO(research): ビット割り当ては検討の余地あり
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
      jxl::Val(1), jxl::BitsOffset(4, 1), jxl::BitsOffset(8, 1 + (1 << 4)),
      jxl::BitsOffset(14, 1 + (1 << 4) + (1 << 8)), 1, &n_images));

  // 3～6bpp であると想定する
  uint64_t total_px =
      static_cast<uint64_t>(width_) * height_ * n_channel_ * n_images;
  uint32_t total_bytes_3bpp = total_px * 3 / 8;
  uint32_t total_bytes_6bpp = total_px * 6 / 8;
  JXL_ASSERT(total_bytes_3bpp > 1);
  JXL_ASSERT(total_bytes_3bpp < total_bytes_6bpp);
  int lower_bits = bit_width(total_bytes_3bpp) - 1;
  int range_bits = bit_width(total_bytes_6bpp - (2 + (1u << lower_bits)));
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
      jxl::BitsOffset(lower_bits, 1),  // 3bpp 未満のケース
      jxl::BitsOffset(range_bits, 1 + (1 << lower_bits)),  // 3～6bpp
      // 以下、外した場合
      jxl::BitsOffset(24, 1), jxl::BitsOffset(30, 1 + (1 << 24)), 1, &n_bytes));

  if (!visitor->IsReading() &&
      (n_images < 1u + (1u << lower_bits) ||
       n_images >= 1u + (1u << lower_bits) + (1u << range_bits)))
    JXL_WARNING("n_bytes (%" PRIu32 ") is not in expected range", n_bytes);

  return true;
}

jxl::Status ClusterHeader::VisitFields(jxl::Visitor* JXL_RESTRICT visitor) {
  uint32_t n_combined_images = static_cast<uint32_t>(combined_images.size());
  // TODO(research): ビット割り当ては検討の余地あり
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
      jxl::Val(1), jxl::BitsOffset(4, 1), jxl::BitsOffset(8, 1 + (1 << 4)),
      jxl::BitsOffset(14, 1 + (1 << 4) + (1 << 8)), 0, &n_combined_images));

  if (visitor->IsReading()) {
    combined_images.resize(n_combined_images,
                           CombinedImageInfo(width_, height_, n_channel_));
  }

  for (size_t i = 0; i < n_combined_images; i++) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&combined_images[i]));
  }

  uint32_t n_images = 0;
  for (const auto& x : combined_images) n_images += x.n_images;

  if (n_images <= 0) return false;

  // 値域を小さくするため、未使用のインデックスのインデックスを書き込む
  std::vector<uint32_t> index_map(n_images);
  for (uint32_t i = 0; i < n_images; i++) index_map[i] = i;

  if (visitor->IsReading()) {
    pointers.resize(n_images);

    for (uint32_t i = 0; i < n_images - 1; i++) {
      uint32_t mapped_idx = 0;
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->Bits(bit_width(index_map.size() - 1), 0, &mapped_idx));
      pointers[i] = index_map.at(mapped_idx);
      index_map.erase(index_map.begin() + mapped_idx);
    }

    JXL_ASSERT(index_map.size() == 1);
    pointers[n_images - 1] = index_map[0];
  } else {
    JXL_ASSERT(pointers.size() == n_images);

    for (uint32_t i = 0; i < n_images - 1; i++) {
      auto mapped_idx_iter =
          std::find(index_map.begin(), index_map.end(), pointers[i]);
      JXL_CHECK(mapped_idx_iter != index_map.end());
      uint32_t mapped_idx = mapped_idx_iter - index_map.begin();
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->Bits(bit_width(index_map.size() - 1), 0, &mapped_idx));
      index_map.erase(mapped_idx_iter);
    }

    JXL_ASSERT(index_map.size() == 1);
    JXL_CHECK(pointers[n_images - 1] == index_map[0]);
  }

  return true;
}

}  // namespace research
