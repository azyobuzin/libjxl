#include "common_cluster.h"

#include <limits>
#include <vector>

using namespace jxl;

namespace research {

namespace {

constexpr int bit_width(unsigned int x) noexcept {
  // https://github.com/boostorg/core/blob/df3b9827cfc9f38080c8d90af048f1f20c449c36/include/boost/core/bit.hpp#L449-L453
  int countl = x ? __builtin_clz(x) : std::numeric_limits<unsigned int>::digits;
  return std::numeric_limits<unsigned int>::digits - countl;
}

U32Enc U32EncForNBytes(uint64_t total_px, uint32_t* lower_bound,
                       uint32_t* upper_bound) {
  // 3～6bpp であると想定する
  uint32_t total_bytes_3bpp = total_px * 3 / 8;
  uint32_t total_bytes_6bpp = total_px * 6 / 8;
  JXL_ASSERT(total_bytes_3bpp > 1);
  JXL_ASSERT(total_bytes_3bpp < total_bytes_6bpp);
  int lower_bits = bit_width(total_bytes_3bpp) - 1;
  int range_bits = bit_width(total_bytes_6bpp - (2 + (1u << lower_bits)));
  if (lower_bound) *lower_bound = 1u + (1u << lower_bits);
  if (upper_bound) *upper_bound = 1u + (1u << lower_bits) + (1u << range_bits);
  return U32Enc(BitsOffset(lower_bits, 1),  // 3bpp 未満のケース
                BitsOffset(range_bits, 1 + (1 << lower_bits)),  // 3～6bpp
                // 以下、外した場合
                BitsOffset(24, 1), BitsOffset(30, 1 + (1 << 24)));
}

}  // namespace

Status CombinedImageInfo::VisitFields(Visitor* JXL_RESTRICT visitor) {
  // TODO(research): ビット割り当ては検討の余地あり
  JXL_RETURN_IF_ERROR(
      visitor->U32(Val(1), BitsOffset(4, 1), BitsOffset(8, 1 + (1 << 4)),
                   BitsOffset(14, 1 + (1 << 4) + (1 << 8)), 1, &n_images));

  uint32_t lower_bound, upper_bound;
  U32Enc nb_enc =
      U32EncForNBytes(static_cast<uint64_t>(width_) * height_ *
                          (flif_enabled_ ? 1 : n_channel_) * n_images,
                      &lower_bound, &upper_bound);
  JXL_RETURN_IF_ERROR(visitor->U32(nb_enc, lower_bound, &n_bytes));

  if (!visitor->IsReading() &&
      (n_bytes < lower_bound || n_bytes >= upper_bound)) {
    JXL_WARNING("n_bytes (%" PRIu32 ") is not in expected range [%" PRIu32
                " - %" PRIu32 ")",
                n_bytes, lower_bound, upper_bound);
  }

  if (flif_enabled_) {
    nb_enc =
        U32EncForNBytes(static_cast<uint64_t>(width_) * height_ * 2 * n_images,
                        &lower_bound, &upper_bound);
    JXL_RETURN_IF_ERROR(visitor->U32(nb_enc, lower_bound, &n_flif_bytes));

    if (!visitor->IsReading() &&
        (n_flif_bytes < lower_bound || n_flif_bytes >= upper_bound)) {
      JXL_WARNING("n_flif_bytes (%" PRIu32
                  ") is not in expected range [%" PRIu32 " - %" PRIu32 ")",
                  n_flif_bytes, lower_bound, upper_bound);
    }
  }

  return true;
}

Status ClusterHeader::VisitFields(Visitor* JXL_RESTRICT visitor) {
  uint32_t n_combined_images = static_cast<uint32_t>(combined_images.size());
  // TODO(research): ビット割り当ては検討の余地あり
  U32Enc enc(Val(1), BitsOffset(4, 1), BitsOffset(8, 1 + (1 << 4)),
             BitsOffset(14, 1 + (1 << 4) + (1 << 8)));
  JXL_RETURN_IF_ERROR(visitor->U32(enc, 0, &n_combined_images));

  if (visitor->IsReading()) {
    combined_images.resize(
        n_combined_images,
        CombinedImageInfo(width_, height_, n_channel_, flif_enabled_));
  }

  for (size_t i = 0; i < n_combined_images; i++) {
    JXL_RETURN_IF_ERROR(visitor->VisitNested(&combined_images[i]));
  }

  return true;
}

Status IndexFields::VisitFields(Visitor* JXL_RESTRICT visitor) {
  // https://github.com/libjxl/libjxl/blob/1d62c5fc07ec2bcb4baed46cc4b6e4611c714602/lib/jxl/frame_header.cc#L259-L260
  const U32Enc enc(BitsOffset(8, 1), BitsOffset(11, 1 + (1 << 8)),
                   BitsOffset(14, 1 + (1 << 8) + (1 << 11)),
                   BitsOffset(30, 1 + (1 << 8) + (1 << 11) + (1 << 14)));
  JXL_RETURN_IF_ERROR(visitor->U32(enc, 1, &width));
  JXL_RETURN_IF_ERROR(visitor->U32(enc, 1, &height));

  bool is_color = n_channel == 3;
  JXL_RETURN_IF_ERROR(visitor->Bool(false, &is_color));
  n_channel = is_color ? 3 : 1;

  JXL_RETURN_IF_ERROR(visitor->U32(enc, 1, &n_clusters));

  uint32_t n_images = assignments.size();
  JXL_RETURN_IF_ERROR(visitor->U32(enc, 0, &n_images));

  if (visitor->IsReading()) assignments.resize(n_images);

  // TODO(research): もっと効率よくならない？
  size_t ic_bits = bit_width(n_clusters - 1);

  for (uint32_t i = 0; i < n_images; i++) {
    JXL_RETURN_IF_ERROR(visitor->Bits(ic_bits, 0, &assignments[i]));
  }

  return true;
}

}  // namespace research
