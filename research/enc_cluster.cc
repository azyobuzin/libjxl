#include "enc_cluster.h"

#include "common_cluster.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"

using namespace jxl;

namespace jxl {

float EstimateWPCost(const Image &img, size_t i);

}

namespace research {

namespace {

// splitting_heuristics_properties に max_properties と max_refs を反映する
void ApplyPropertiesOption(ModularOptions &options,
                           const MultiOptions &multi_options) {
  // channel_per_image == 0 → JPEG XL のデフォルト通りにやる
  // max_refs == 0 → 明示的に画像を参照しないので、 max_properties に制限を設けない
  if (multi_options.max_refs > 0 && multi_options.channel_per_image > 0) {
    options.max_properties =
        std::min(options.max_properties,
                 static_cast<int>(multi_options.channel_per_image) - 1);
  }

  uint32_t n_ref_channels =
      options.max_properties +
      multi_options.max_refs * multi_options.channel_per_image;
  for (uint32_t i = 0; i < n_ref_channels * 4; i++) {
    uint32_t prop = kNumNonrefProperties + i;
    if (std::find(options.splitting_heuristics_properties.begin(),
                  options.splitting_heuristics_properties.end(),
                  prop) == options.splitting_heuristics_properties.end())
      options.splitting_heuristics_properties.push_back(prop);
  }
}

}  // namespace

int FindBestWPMode(const Image &image) {
  // https://github.com/libjxl/libjxl/blob/3d077b281fa65eab595447ae38ba9efc385ba03e/lib/jxl/enc_modular.cc#L1375-L1383
  float best_cost = std::numeric_limits<float>::max();
  int wp_mode = 0;
  for (size_t i = 0; i < 5; i++) {
    float cost = EstimateWPCost(image, i);
    if (cost < best_cost) {
      best_cost = cost;
      wp_mode = i;
    }
  }
  return wp_mode;
}

CombinedImage::CombinedImage(std::shared_ptr<const Image> image,
                             size_t n_images)
    : n_images(n_images) {
  JXL_CHECK(image);  // image must be not null
  JXL_CHECK((image->channel.size() - image->nb_meta_channels) % n_images == 0);
  this->image = std::move(image);
}

CombinedImage CombineImage(std::shared_ptr<const Image> image) {
  return {std::move(image), 1};
}

CombinedImage CombineImage(
    const std::vector<std::shared_ptr<const Image>> &images) {
  JXL_CHECK(images.size() > 0);

  if (images.size() == 1) return CombineImage(images[0]);

  const Image &first_image = *images[0];

  // すべての画像が同じ条件であることを確認
  for (auto &x : images) {
    JXL_CHECK(x->nb_meta_channels == 0);
    JXL_CHECK(x->w == first_image.w && x->h == first_image.h);
    JXL_CHECK(x->channel.size() == first_image.channel.size());
    for (auto &c : x->channel)
      JXL_CHECK(c.w == first_image.w && c.h == first_image.h && c.hshift == 0 &&
                c.vshift == 0);
  }

  auto image = std::make_shared<Image>(
      first_image.w, first_image.h, first_image.bitdepth,
      first_image.channel.size() * images.size());

  for (size_t i = 0; i < images.size(); i++) {
    for (size_t j = 0; j < first_image.channel.size(); j++) {
      const Channel &src = images[i]->channel.at(j);
      Channel &dst = image->channel.at(i * first_image.channel.size() + j);
      CopyImageTo(src.plane, &dst.plane);
    }
  }

  return {std::move(image), images.size()};
}

// modular/encoding/enc_encoding.cc で定義。
// ModularGenericCompress に multi_options を追加。
Status ModularEncodeMulti(
    const Image &image, const ModularOptions &options,
    const MultiOptions &multi_options, BitWriter *writer,
    AuxOut *aux_out = nullptr, size_t layer = 0, size_t group_id = 0,
    TreeSamples *tree_samples = nullptr, size_t *total_pixels = nullptr,
    const Tree *tree = nullptr, GroupHeader *header = nullptr,
    std::vector<Token> *tokens = nullptr, size_t *width = nullptr);

Tree LearnTree(BitWriter &writer, const CombinedImage &ci,
               ModularOptions &options, size_t max_refs) {
  const Image &image = *ci.image;
  MultiOptions multi_options{
      (image.channel.size() - image.nb_meta_channels) / ci.n_images,
      std::min(max_refs, ci.n_images - 1)};
  ApplyPropertiesOption(options, multi_options);
  options.wp_mode = FindBestWPMode(image);

  TreeSamples tree_samples;
  if (!tree_samples.SetPredictor(options.predictor, options.wp_tree_mode))
    JXL_ABORT("SetPredictor failed");

  if (!tree_samples.SetProperties(options.splitting_heuristics_properties,
                                  options.wp_tree_mode))
    JXL_ABORT("SetProperty failed");

  std::vector<pixel_type> pixel_samples;
  std::vector<pixel_type> diff_samples;
  std::vector<uint32_t> group_pixel_count;
  std::vector<uint32_t> channel_pixel_count;
  CollectPixelSamples(image, options, 0, group_pixel_count, channel_pixel_count,
                      pixel_samples, diff_samples);

  StaticPropRange range;
  range[0] = {{0, static_cast<uint32_t>(image.channel.size())}};
  range[1] = {{0, 1}};  // group id
  std::vector<ModularMultiplierInfo> multiplier_info;
  tree_samples.PreQuantizeProperties(range, multiplier_info, group_pixel_count,
                                     channel_pixel_count, pixel_samples,
                                     diff_samples, options.max_property_values);

  size_t total_pixels = 0;
  JXL_CHECK(ModularEncodeMulti(image, options, multi_options, nullptr, nullptr,
                               0, 0, &tree_samples, &total_pixels));

  Tree tree = LearnTree(std::move(tree_samples), total_pixels, options,
                        multiplier_info, range);

  std::vector<std::vector<Token>> tokens(1);
  Tree decoded_tree;
  TokenizeTree(tree, &tokens[0], &decoded_tree);

  HistogramParams params;
  params.lz77_method = HistogramParams::LZ77Method::kOptimal;
  EntropyEncodingData code;
  std::vector<uint8_t> context_map;
  BuildAndEncodeHistograms(params, kNumTreeContexts, tokens, &code,
                           &context_map, &writer, kLayerModularTree, nullptr);
  WriteTokens(tokens[0], code, context_map, &writer, kLayerModularTree,
              nullptr);

  return decoded_tree;
}

void EncodeImages(jxl::BitWriter &writer, const CombinedImage &ci,
                  const jxl::ModularOptions &options_in, size_t max_refs,
                  const jxl::Tree &tree) {
  const Image &image = *ci.image;
  MultiOptions multi_options{
      (image.channel.size() - image.nb_meta_channels) / ci.n_images,
      std::min(max_refs, ci.n_images - 1)};
  ModularOptions options = options_in;
  ApplyPropertiesOption(options, multi_options);

  GroupHeader group_header;
  std::vector<std::vector<Token>> tokens(1);
  std::vector<size_t> image_widths(1);
  JXL_CHECK(ModularEncodeMulti(image, options, multi_options, nullptr, nullptr,
                               0, 0, nullptr, nullptr, &tree, &group_header,
                               &tokens[0], &image_widths[0]));

  HistogramParams params;
  // TODO(research):
  // LZ77で時間がかかり、オフにしても決定木比較に影響ないならば、kNoneにしたい
  params.lz77_method = HistogramParams::LZ77Method::kOptimal;
  params.image_widths = std::move(image_widths);
  EntropyEncodingData code;
  std::vector<uint8_t> context_map;
  BuildAndEncodeHistograms(params, (tree.size() + 1) / 2, tokens, &code,
                           &context_map, &writer, 0, nullptr);

  JXL_CHECK(Bundle::Write(group_header, &writer, kLayerHeader, nullptr));
  WriteTokens(tokens[0], code, context_map, &writer, 0, nullptr);
}

void PackToClusterFile(const std::vector<EncodedCombinedImage> &combined_images,
                       std::ostream &dst) {
  JXL_CHECK(combined_images.size() > 0);

  // 1枚目からデータを収集
  uint32_t width, height, n_channel;
  bool flif_enabled;
  {
    const auto &first_image = combined_images[0].included_images.at(0);
    width = first_image->w;
    height = first_image->h;
    n_channel = first_image->channel.size() - first_image->nb_meta_channels;
    flif_enabled = !combined_images[0].flif_data.empty();
  }

  // ヘッダー作成
  ClusterHeader header(width, height, n_channel, flif_enabled);
  size_t n_images = 0;

  header.combined_images.reserve(combined_images.size());
  for (const auto &ci : combined_images) {
    auto &ci_info = header.combined_images.emplace_back(
        width, height, n_channel, flif_enabled);
    JXL_ASSERT(ci.image_indices.size() == ci.included_images.size());
    n_images += ci.image_indices.size();
    ci_info.n_images = static_cast<uint32_t>(ci.image_indices.size());
    ci_info.n_bytes = static_cast<uint32_t>(ci.data.size());
    ci_info.n_flif_bytes = static_cast<uint32_t>(ci.flif_data.size());
  }

  header.pointers.resize(n_images);
  size_t ptr_idx = 0;
  for (const auto &ci : combined_images) {
    for (auto idx : ci.image_indices) header.pointers.at(idx) = ptr_idx++;
  }

  JXL_ASSERT(ptr_idx == n_images);

  BitWriter header_writer;
  JXL_CHECK(Bundle::Write(header, &header_writer, 0, nullptr));
  header_writer.ZeroPadToByte();

  Span<const uint8_t> header_span = header_writer.GetSpan();
  dst.write(reinterpret_cast<const char *>(header_span.data()),
            header_span.size());

  // 画像を書き込む
  for (const auto &ci : combined_images) {
    dst.write(reinterpret_cast<const char *>(ci.data.data()), ci.data.size());

    if (flif_enabled) {
      dst.write(reinterpret_cast<const char *>(ci.flif_data.data()),
                ci.flif_data.size());
    }
  }
}

}  // namespace research
