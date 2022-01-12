#include "dec_cluster.h"

#include <fmt/core.h>
#include <gmpxx.h>
#include <tbb/parallel_for.h>

#include "dec_flif.h"
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
                           const std::vector<uint32_t> *references,
                           Span<const uint8_t> jxl_data,
                           Span<const uint8_t> flif_data,
                           std::vector<Image> &out_images) {
  // FLIF がある場合は、Yチャネルのみ JPEG XL になっている
  MultiOptions multi_options = {decoding_options.n_channel,
                                decoding_options.reference_type, references};
  if (decoding_options.flif_enabled) multi_options.channel_per_image = 1;

  BitReader reader(jxl_data);

  // 決定木
  Tree tree;
  // 巨大決定木が生まれるので、制限を緩くしておく
  size_t tree_size_limit = std::numeric_limits<int32_t>().max();
  Status status =
      JXL_STATUS(DecodeTree(&reader, &tree, tree_size_limit), "DecodeTree");
  if (!status) {
    [[maybe_unused]] auto ignored = reader.Close();
    return status;
  }

  // 画像のヒストグラム
  ANSCode code;
  std::vector<uint8_t> context_map;
  status = JXL_STATUS(
      DecodeHistograms(&reader, (tree.size() + 1) / 2, &code, &context_map),
      "DecodeHistograms");
  if (!status) {
    [[maybe_unused]] auto ignored = reader.Close();
    return status;
  }

  // 画像
  Image ci(decoding_options.width, decoding_options.height, kBitdepth,
           multi_options.channel_per_image * out_images.size());
  ModularOptions options;
  DecodingRect dr = {"research::DecodeCombinedImage", 0, 0, 0};
  status = JXL_STATUS(ModularDecodeMulti(&reader, ci, 0, &options, &tree, &code,
                                         &context_map, &dr, multi_options),
                      "ModularDecodeMulti");
  if (!status) {
    [[maybe_unused]] auto ignored = reader.Close();
    return status;
  }

  if (!reader.JumpToByteBoundary() ||
      reader.TotalBitsConsumed() != jxl_data.size() * kBitsPerByte) {
    [[maybe_unused]] auto ignored = reader.Close();
    return JXL_STATUS(false, "読み残しがあります");
  }

  JXL_RETURN_IF_ERROR(reader.Close());

  // チャネルを切り出す
  for (size_t i = 0; i < out_images.size(); i++) {
    auto &dst = out_images[i];
    dst = Image(ci.w, ci.h, ci.bitdepth, multi_options.channel_per_image);

    for (uint32_t c = 0; c < multi_options.channel_per_image; c++) {
      CopyImageTo(ci.channel.at(i * multi_options.channel_per_image + c).plane,
                  &dst.channel.at(c).plane);
    }
  }

  // FLIF のデコード
  if (decoding_options.flif_enabled) {
    JXL_RETURN_IF_ERROR(DecodeColorSignalWithFlif(
        out_images, flif_data, decoding_options.flif_additional_props));
  }

  return true;
}

mpz_class ReadMpz(jxl::BitReader &reader, const mpz_class &max_state) {
  mpz_class state = 0;
  size_t n_bits = mpz_sizeinbase(max_state.get_mpz_t(), 2);
  mp_bitcnt_t shift = 0;
  while (shift <= n_bits) {
    mp_bitcnt_t bits_left = n_bits - shift;
    if (bits_left >= 32) {
      state |= mpz_class(reader.ReadFixedBits<32>()) << shift;
      shift += 32;
    } else {
      state |= mpz_class(reader.ReadBits(bits_left)) << shift;
      break;
    }
  }
  return state;
}

}  // namespace

ClusterFileReader::ClusterFileReader(const DecodingOptions &options,
                                     Span<const uint8_t> data)
    : options_(options),
      header_(options.width, options.height, options.n_channel,
              options.flif_enabled) {
  BitReader reader(data);
  JXL_CHECK(Bundle::Read(&reader, &header_));

  uint32_t n_images = 0;
  for (const auto &x : header_.combined_images) n_images += x.n_images;

  pointers_.resize(n_images);
  DecodeClusterPointers(reader, pointers_);

  if (options.reference_type != kParentReferenceNone) {
    references_.resize(header_.combined_images.size());
    for (size_t i = 0; i < header_.combined_images.size(); i++) {
      references_[i].resize(header_.combined_images[i].n_images - 1);
      DecodeReferences(reader, references_[i]);
    }
  }

  JXL_CHECK(reader.JumpToByteBoundary());
  JXL_CHECK(reader.Close());
  data_ = reader.GetSpan();
}

Status ClusterFileReader::ReadAll(std::vector<Image> &out_images) {
  // 画像列のインデックスから、本来のインデックスを求められるようにする
  std::vector<size_t> reverse_pointer(n_images());
  for (size_t i = 0; i < n_images(); i++) reverse_pointer.at(pointers_[i]) = i;

  // CombinedImageのオフセットを求める
  std::vector<std::pair<size_t, size_t>> accum_idx_bytes;
  accum_idx_bytes.reserve(header_.combined_images.size());
  accum_idx_bytes.emplace_back();
  for (size_t i = 1; i < header_.combined_images.size(); i++) {
    const auto &prev = accum_idx_bytes.back();
    const auto &ci_info = header_.combined_images[i - 1];
    accum_idx_bytes.emplace_back(
        prev.first + ci_info.n_images,
        prev.second + ci_info.n_bytes +
            (options_.flif_enabled ? ci_info.n_flif_bytes : 0));
  }

  std::atomic<Status> status = Status(true);
  out_images.resize(n_images());

  tbb::parallel_for(size_t(0), accum_idx_bytes.size(), [&](size_t i) {
    const auto &ci_info = header_.combined_images[i];
    auto [idx_offset, bytes_offset] = accum_idx_bytes[i];

    const std::vector<uint32_t> *references =
        options_.reference_type == kParentReferenceNone ? nullptr
                                                        : &references_.at(i);
    Span<const uint8_t> jxl_span(data_.data() + bytes_offset, ci_info.n_bytes);
    Span<const uint8_t> flif_span(data_.data() + bytes_offset + ci_info.n_bytes,
                                  ci_info.n_flif_bytes);

    std::vector<Image> images(ci_info.n_images);
    Status decode_status = JXL_STATUS(
        DecodeCombinedImage(options_, references, jxl_span, flif_span, images),
        "failed to decode %" PRIuS, i);

    if (decode_status) {
      for (size_t j = 0; j < ci_info.n_images; j++) {
        out_images[reverse_pointer.at(idx_offset + j)] =
            std::move(images.at(j));
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

Status ClusterFileReader::Read(uint32_t idx, Image &out_image) {
  idx = pointers_.at(idx);

  uint32_t accum_idx = 0, accum_bytes = 0;

  for (size_t i = 0; i < header_.combined_images.size(); i++) {
    const auto &ci_info = header_.combined_images[i];

    if (idx >= accum_idx + ci_info.n_images) {
      accum_idx += ci_info.n_images;
      accum_bytes += ci_info.n_bytes;
      if (options_.flif_enabled) accum_bytes += ci_info.n_flif_bytes;
      continue;
    }

    // Found
    const std::vector<uint32_t> *references =
        options_.reference_type == kParentReferenceNone ? nullptr
                                                        : &references_.at(i);
    Span<const uint8_t> jxl_span(data_.data() + accum_bytes, ci_info.n_bytes);
    Span<const uint8_t> flif_span(data_.data() + accum_bytes + ci_info.n_bytes,
                                  ci_info.n_flif_bytes);
    std::vector<Image> images(ci_info.n_images);
    Status status =
        DecodeCombinedImage(options_, references, jxl_span, flif_span, images);
    if (status) {
      out_image = std::move(images.at(idx - accum_idx));
    }
    return status;
  }

  throw std::out_of_range(fmt::format("idx {} >= n_images {}", idx, accum_idx));
}

void DecodeClusterPointers(BitReader &reader, std::vector<uint32_t> &pointers) {
  const uint32_t n_images = static_cast<uint32_t>(pointers.size());
  if (n_images == 0) return;

  pointers[n_images - 1] = 0;

  if (n_images == 1) return;

  // 理論上の最大値
  mpz_class max_state = 0;
  for (uint32_t i = 0; i < n_images - 1; i++) {
    max_state *= n_images - i;
    max_state += n_images - i - 1;
  }

  mpz_class state = ReadMpz(reader, max_state);

  for (uint32_t i = 2; i <= n_images; i++) {
    // 商が次の状態、余りが値
    pointers[n_images - i] =
        mpz_tdiv_q_ui(state.get_mpz_t(), state.get_mpz_t(), i);
  }

  std::vector<uint32_t> index_map(n_images);
  for (uint32_t i = 0; i < n_images; i++) index_map[i] = i;

  // pointers を実際の値にマップする
  for (uint32_t &p : pointers) {
    JXL_ASSERT(p < index_map.size());
    auto it = index_map.begin() + p;
    p = *it;
    index_map.erase(it);
  }
}

void DecodeReferences(BitReader &reader, std::vector<uint32_t> &references) {
  const uint32_t n_refs = references.size();
  if (n_refs == 0) return;

  references[0] = 0;

  if (n_refs == 1) return;

  mpz_class max_state = 0;
  for (uint32_t i = 1; i < n_refs; i++) {
    max_state *= i + 1;
    max_state += i;
  }

  mpz_class state = ReadMpz(reader, max_state);

  for (uint32_t i = n_refs - 1; i >= 1; i--) {
    // 商が次の状態、余りが値
    references[i] = mpz_tdiv_q_ui(state.get_mpz_t(), state.get_mpz_t(), i + 1);
  }
}

}  // namespace research
