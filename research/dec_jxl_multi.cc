#include "dec_jxl_multi.h"

#include <fmt/core.h>
#include <tbb/parallel_for.h>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/encoding.h"

using namespace jxl;

namespace research {

// modular/encoding/encoding.cc で定義
jxl::Status ModularDecodeMulti(
    jxl::BitReader *br, jxl::Image &image, jxl::GroupHeader &header,
    size_t group_id, jxl::ModularOptions *options, const jxl::Tree *global_tree,
    const jxl::ANSCode *global_code, const std::vector<uint8_t> *global_ctx_map,
    const jxl::DecodingRect *rect, const jxl::MultiOptions &multi_options);

namespace {

// 8bit しか想定しないことにする
constexpr int kBitdepth = 8;

Status DecodeCombinedImage(uint32_t width, uint32_t height, uint32_t n_images,
                           const MultiOptions &multi_options,
                           Span<const uint8_t> data, Image &out_image) {
  BitReader reader(data);

  // 決定木
  Tree tree;
  size_t tree_size_limit =
      std::min(static_cast<size_t>(1 << 22),
               1024 + static_cast<size_t>(width) * height *
                          multi_options.channel_per_image / 16);
  Status status =
      JXL_STATUS(DecodeTree(&reader, &tree, tree_size_limit), "DecodeTree");
  if (!status) {
    reader.Close();
    return status;
  }

  // 画像のヒストグラム
  ANSCode code;
  std::vector<uint8_t> context_map;
  status = JXL_STATUS(
      DecodeHistograms(&reader, (tree.size() + 1) / 2, &code, &context_map),
      "DecodeHistograms");
  if (!status) {
    reader.Close();
    return status;
  }

  // 画像
  out_image = Image(width, height, kBitdepth,
                    multi_options.channel_per_image * n_images);
  GroupHeader header;
  header.use_global_tree = true;
  ModularOptions options;
  DecodingRect dr = {"research::DecodeCombinedImage", 0, 0, 0};
  status = JXL_STATUS(
      ModularDecodeMulti(&reader, out_image, header, 0, &options, &tree, &code,
                         &context_map, &dr, multi_options),
      "ModularDecodeMulti");
  if (!status) {
    reader.Close();
    return status;
  }

  JXL_RETURN_IF_ERROR(reader.Close());
  return true;
}

Image ImageFromCombined(const Image &combined_image, uint32_t n_channel,
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
                                     uint32_t n_channel, size_t max_refs,
                                     Span<const uint8_t> data)
    : width_(width), height_(height), header_(width, height, n_channel) {
  multi_options_.channel_per_image = n_channel;
  multi_options_.max_refs = max_refs;

  BitReader reader(data);
  JXL_CHECK(Bundle::Read(&reader, &header_));
  JXL_CHECK(reader.JumpToByteBoundary());
  JXL_CHECK(reader.Close());
  data_ = reader.GetSpan();
}

Status ClusterFileReader::ReadAll(std::vector<Image> &out_images) {
  // 画像列のインデックスから、本来のインデックスを求められるようにする
  std::vector<size_t> reverse_pointer(n_images());
  for (size_t i = 0; i < n_images(); i++)
    reverse_pointer.at(header_.pointers[i]) = i;

  // CombinedImageのオフセットを求める
  std::vector<std::pair<size_t, size_t>> accum_idx_bytes;
  accum_idx_bytes.reserve(header_.combined_images.size());
  accum_idx_bytes.emplace_back();
  for (size_t i = 1; i < header_.combined_images.size(); i++) {
    const auto &prev = accum_idx_bytes.back();
    accum_idx_bytes.emplace_back(
        prev.first + header_.combined_images[i].n_images,
        prev.second + header_.combined_images[i].n_bytes);
  }

  std::atomic<Status> status = Status(true);

  tbb::parallel_for(size_t(0), accum_idx_bytes.size(), [&](size_t i) {
    const auto &ci_info = header_.combined_images[i];
    auto [idx_offset, bytes_offset] = accum_idx_bytes[i];
    Span<const uint8_t> span(data_.data() + bytes_offset, ci_info.n_bytes);
    Image combined_image;
    Status decode_status =
        JXL_STATUS(DecodeCombinedImage(width_, height_, ci_info.n_images,
                                       multi_options_, span, combined_image),
                   "failed to decode %" PRIuS, i);
    if (decode_status) {
      for (size_t j = 0; j < ci_info.n_images; j++) {
        out_images[reverse_pointer.at(idx_offset + j)] = ImageFromCombined(
            combined_image, multi_options_.channel_per_image, j);
      }
    } else {
      // 失敗を記録
      while (true) {
        Status old_status = status.load();
        if (old_status.IsFatalError()) return;
        if (!old_status && !decode_status.IsFatalError()) return;
        if (status.compare_exchange_weak(old_status, decode_status)) return;
      }
    }
  });

  return status;
}

Status ClusterFileReader::Read(size_t idx, Image &out_image) {
  idx = header_.pointers.at(idx);

  size_t accum_idx = 0, accum_bytes = 0;

  for (const auto &ci_info : header_.combined_images) {
    if (idx >= accum_idx + ci_info.n_images) {
      accum_idx += ci_info.n_images;
      accum_bytes += ci_info.n_bytes;
      continue;
    }

    // Found
    Span<const uint8_t> span(data_.data() + accum_bytes, ci_info.n_bytes);
    Image combined_image;
    Status status = DecodeCombinedImage(width_, height_, ci_info.n_images,
                                        multi_options_, span, combined_image);
    if (status) {
      out_image = ImageFromCombined(
          combined_image, multi_options_.channel_per_image, idx - accum_idx);
    }
    return status;
  }

  throw std::out_of_range(fmt::format("idx {} >= n_images {}", idx, accum_idx));
}

}  // namespace research
