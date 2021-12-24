#include "dec_cluster.h"

#include <fmt/core.h>
#include <tbb/parallel_for.h>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/encoding.h"

using namespace jxl;

namespace research {

// modular/encoding/encoding.cc で定義
Status ModularDecodeMulti(BitReader *br, Image &image, size_t group_id,
                          ModularOptions *options, const Tree *global_tree,
                          const ANSCode *global_code,
                          const std::vector<uint8_t> *global_ctx_map,
                          const DecodingRect *rect,
                          const MultiOptions &multi_options);

namespace {

// 8bit しか想定しないことにする
constexpr int kBitdepth = 8;

Status DecodeCombinedImage(const DecodingOptions &decoding_options,
                           uint32_t n_images, Span<const uint8_t> data,
                           Image &out_image) {
  BitReader reader(data);

  // 決定木
  Tree tree;
  size_t tree_size_limit = std::min(
      static_cast<size_t>(1 << 22),
      1024 + static_cast<size_t>(decoding_options.width) *
                 decoding_options.height *
                 decoding_options.multi_options.channel_per_image / 16);
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
  out_image =
      Image(decoding_options.width, decoding_options.height, kBitdepth,
            decoding_options.multi_options.channel_per_image * n_images);
  ModularOptions options;
  options.max_properties = decoding_options.refchan;
  DecodingRect dr = {"research::DecodeCombinedImage", 0, 0, 0};
  status = JXL_STATUS(
      ModularDecodeMulti(&reader, out_image, 0, &options, &tree, &code,
                         &context_map, &dr, decoding_options.multi_options),
      "ModularDecodeMulti");
  if (!status) {
    reader.Close();
    return status;
  }

  if (!reader.JumpToByteBoundary() ||
      reader.TotalBitsConsumed() != data.size() * kBitsPerByte) {
    reader.Close();
    return JXL_STATUS(false, "読み残しがあります");
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

ClusterFileReader::ClusterFileReader(const DecodingOptions &options,
                                     Span<const uint8_t> data)
    : options_(options),
      header_(options.width, options.height,
              options.multi_options.channel_per_image, options.flif_enabled) {
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
  out_images.resize(n_images());

  tbb::parallel_for(size_t(0), accum_idx_bytes.size(), [&](size_t i) {
    const auto &ci_info = header_.combined_images[i];
    auto [idx_offset, bytes_offset] = accum_idx_bytes[i];
    Span<const uint8_t> span(data_.data() + bytes_offset, ci_info.n_bytes);
    Image combined_image;
    Status decode_status = JXL_STATUS(
        DecodeCombinedImage(options_, ci_info.n_images, span, combined_image),
        "failed to decode %" PRIuS, i);
    if (decode_status) {
      for (size_t j = 0; j < ci_info.n_images; j++) {
        out_images[reverse_pointer.at(idx_offset + j)] = ImageFromCombined(
            combined_image, options_.multi_options.channel_per_image, j);
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
    Status status =
        DecodeCombinedImage(options_, ci_info.n_images, span, combined_image);
    if (status) {
      out_image = ImageFromCombined(combined_image,
                                    options_.multi_options.channel_per_image,
                                    idx - accum_idx);
    }
    return status;
  }

  throw std::out_of_range(fmt::format("idx {} >= n_images {}", idx, accum_idx));
}

}  // namespace research
