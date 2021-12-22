#include "dec_jxl_multi.h"

#include <fmt/core.h>
#include <tbb/parallel_for.h>

#include "lib/jxl/dec_ans.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/encoding.h"

using namespace jxl;

namespace research {

namespace {

// 8bit しか想定しないことにする
constexpr int kBitdepth = 8;

Image DecodeCombinedImage(uint32_t width, uint32_t height, uint32_t n_channel,
                          uint32_t n_images, Span<const uint8_t> data) {
  BitReader reader(data);

  // 決定木
  Tree tree;
  size_t tree_size_limit =
      std::min(static_cast<size_t>(1 << 22),
               1024 + static_cast<size_t>(width) * height * n_channel / 16);
  JXL_CHECK(DecodeTree(&reader, &tree, tree_size_limit));

  // 画像のヒストグラム
  ANSCode code;
  std::vector<uint8_t> context_map;
  JXL_CHECK(
      DecodeHistograms(&reader, (tree.size() + 1) / 2, &code, &context_map));

  // 画像
  Image ci(width, height, kBitdepth, n_channel * n_images);
  GroupHeader header;
  header.use_global_tree = true;
  ModularOptions options;
  DecodingRect dr = {"research::DecodeCombinedImage", 0, 0, 0};
  JXL_CHECK(ModularGenericDecompress(&reader, ci, &header, 0, &options, &dr,
                                     false, &tree, &code, &context_map, false));
  return ci;
}

Image ImageFromCombined(const Image& combined_image, uint32_t n_channel,
                        uint32_t idx) {
  Image result(combined_image.w, combined_image.h, combined_image.bitdepth,
               n_channel);
  for (uint32_t c = 0; c < n_channel; c++) {
    CopyImageTo(combined_image.channel.at(idx * n_channel + c).plane,
                &result.channel[c].plane);
  }
  return result;
}

}  // namespace

ClusterFileReader::ClusterFileReader(uint32_t width, uint32_t height,
                                     uint32_t n_channel,
                                     Span<const uint8_t> data)
    : width_(width),
      height_(height),
      n_channel_(n_channel),
      header_(width, height, n_channel) {
  BitReader reader(data);
  JXL_CHECK(Bundle::Read(&reader, &header_));
  JXL_CHECK(reader.JumpToByteBoundary());
  data_ = reader.GetSpan();
}

std::vector<Image> ClusterFileReader::ReadAll() {
  std::vector<Image> results(n_images());

  std::vector<std::pair<size_t, size_t>> accum_idx_bytes;
  accum_idx_bytes.reserve(header_.combined_images.size());
  accum_idx_bytes.emplace_back();
  for (size_t i = 1; i < header_.combined_images.size(); i++) {
    const auto& prev = accum_idx_bytes.back();
    accum_idx_bytes.emplace_back(
        prev.first + header_.combined_images[i].n_images,
        prev.second + header_.combined_images[i].n_bytes);
  }

  tbb::parallel_for(size_t(0), accum_idx_bytes.size(), [&](size_t i) {
    const auto& ci_info = header_.combined_images[i];
    auto [result_idx, offset] = accum_idx_bytes[i];
    Span<const uint8_t> span(data_.data() + offset, ci_info.n_bytes);
    Image combined_image = DecodeCombinedImage(width_, height_, n_channel_,
                                               ci_info.n_images, span);
    for (size_t j = 0; j < ci_info.n_images; j++)
      results.at(result_idx + j) =
          ImageFromCombined(combined_image, n_channel_, j);
  });

  return results;
}

Image ClusterFileReader::Read(size_t idx) {
  size_t accum_idx = 0, accum_bytes = 0;

  for (const auto& ci_info : header_.combined_images) {
    if (idx >= accum_idx + ci_info.n_images) {
      accum_idx += ci_info.n_images;
      accum_bytes += ci_info.n_bytes;
      continue;
    }

    // Found
    Span<const uint8_t> span(data_.data() + accum_bytes, ci_info.n_bytes);
    Image combined_image = DecodeCombinedImage(width_, height_, n_channel_,
                                               ci_info.n_images, span);
    return ImageFromCombined(combined_image, n_channel_, idx - accum_idx);
  }

  throw std::out_of_range(fmt::format("idx {} >= n_images {}", idx, accum_idx));
}

}  // namespace research
