#include "enc_jxl_multi.h"

#include "lib/jxl/image_ops.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"

using namespace jxl;

namespace research {

CombinedImage::CombinedImage(Image image, size_t n_images)
    : n_images(n_images) {
  JXL_CHECK((image.channel.size() - image.nb_meta_channels) % n_images == 0);
  this->image = std::move(image);
}

CombinedImage CombineImage(Image &&image) {
  return {std::forward<Image>(image), 1};
}

CombinedImage CombineImage(const std::vector<std::shared_ptr<const Image>> &images) {
  JXL_CHECK(images.size() > 0);
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

  Image image(first_image.w, first_image.h, first_image.bitdepth,
              first_image.channel.size() * images.size());

  for (size_t i = 0; i < images.size(); i++) {
    for (size_t j = 0; j < first_image.channel.size(); j++) {
      const Channel &src = images[i]->channel.at(j);
      Channel &dst = image.channel.at(i * first_image.channel.size() + j);
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
               const ModularOptions &options, size_t max_refs) {
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
  CollectPixelSamples(ci.image, options, 0, group_pixel_count,
                      channel_pixel_count, pixel_samples, diff_samples);

  StaticPropRange range;
  range[0] = {{0, static_cast<uint32_t>(ci.image.channel.size())}};
  range[1] = {{0, 1}};  // group id
  std::vector<ModularMultiplierInfo> multiplier_info;
  tree_samples.PreQuantizeProperties(range, multiplier_info, group_pixel_count,
                                     channel_pixel_count, pixel_samples,
                                     diff_samples, options.max_property_values);

  size_t total_pixels = 0;
  JXL_CHECK(ModularEncodeMulti(
      ci.image, options,
      {(ci.image.channel.size() - ci.image.nb_meta_channels) / ci.n_images,
       std::min(max_refs, ci.n_images - 1)},
      nullptr, nullptr, 0, 0, &tree_samples, &total_pixels));

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
                  const jxl::ModularOptions &options, size_t max_refs,
                  const jxl::Tree &tree) {
  std::vector<std::vector<Token>> tokens(1);
  std::vector<size_t> image_widths(1);
  JXL_CHECK(ModularEncodeMulti(
      ci.image, options,
      {(ci.image.channel.size() - ci.image.nb_meta_channels) / ci.n_images,
       std::min(max_refs, ci.n_images - 1)},
      nullptr, nullptr, 0, 0, nullptr, nullptr, &tree, nullptr, &tokens[0],
      &image_widths[0]));

  HistogramParams params;
  // TODO(research):
  // LZ77で時間がかかり、オフにしても決定木比較に影響ないならば、kNoneにしたい
  params.lz77_method = HistogramParams::LZ77Method::kOptimal;
  params.image_widths = std::move(image_widths);
  EntropyEncodingData code;
  std::vector<uint8_t> context_map;
  BuildAndEncodeHistograms(params, (tree.size() + 1) / 2, tokens, &code,
                           &context_map, &writer, 0, nullptr);
  WriteTokens(tokens[0], code, context_map, &writer, 0, nullptr);
}

}  // namespace research
