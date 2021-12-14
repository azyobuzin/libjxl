#include "prop_extract.h"

#include <fmt/core.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <random>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/modular/encoding/enc_debug_tree.h"
#include "lib/jxl/modular/encoding/enc_ma.h"

using namespace jxl;

namespace research {

namespace {

struct CollectSamplesBody {
  ImagesProvider &images;
  const ModularOptions &options;

  std::vector<uint32_t> group_pixel_count;
  std::vector<uint32_t> channel_pixel_count;
  std::vector<std::vector<jxl::pixel_type>> pixel_samples;
  std::vector<std::vector<jxl::pixel_type>> diff_samples;

  CollectSamplesBody(ImagesProvider &images,
                     const ModularOptions &options) noexcept
      : images(images), options(options) {}

  CollectSamplesBody(const CollectSamplesBody &other, tbb::split) noexcept
      : images(other.images), options(other.options) {}

  void operator()(const tbb::blocked_range<size_t> &range) {
    auto &ps = pixel_samples.emplace_back();
    auto &ds = diff_samples.emplace_back();

    for (auto i = range.begin(); i < range.end(); i++) {
      auto image = images.get(i);

      // すべてが有効なチャンネルであると仮定する（パレット変換をしていない）
      JXL_ASSERT(image.nb_meta_channels == 0);

      CollectPixelSamples(image, options, 0, group_pixel_count,
                          channel_pixel_count, ps, ds);
    }
  }

  void join(CollectSamplesBody &rhs) {
    // Merge group_pixel_count
    if (group_pixel_count.size() < rhs.group_pixel_count.size())
      group_pixel_count.resize(rhs.group_pixel_count.size());
    for (size_t i = 0; i < rhs.group_pixel_count.size(); i++)
      group_pixel_count[i] += rhs.group_pixel_count[i];

    // Merge channel_pixel_count
    if (channel_pixel_count.size() < rhs.channel_pixel_count.size())
      channel_pixel_count.resize(rhs.channel_pixel_count.size());
    for (size_t i = 0; i < rhs.channel_pixel_count.size(); i++)
      channel_pixel_count[i] += rhs.channel_pixel_count[i];

    // Merge pixel_samples
    pixel_samples.reserve(pixel_samples.size() + rhs.pixel_samples.size());
    std::move(rhs.pixel_samples.begin(), rhs.pixel_samples.end(),
              std::back_inserter(pixel_samples));

    // Merge diff_samples
    diff_samples.reserve(diff_samples.size() + rhs.diff_samples.size());
    std::move(rhs.diff_samples.begin(), rhs.diff_samples.end(),
              std::back_inserter(diff_samples));
  }
};

template <typename T>
void flatten(const std::vector<std::vector<T>> &src, std::vector<T> &dst) {
  size_t total = 0;
  for (const std::vector<T> &x : src) total += x.size();
  dst.reserve(dst.size() + total);

  for (const std::vector<T> &x : src)
    dst.insert(dst.end(), x.cbegin(), x.cend());
}

}  // namespace

SamplesForQuantization CollectSamplesForQuantization(
    ImagesProvider &images, const ModularOptions &options) {
  CollectSamplesBody body(images, options);
  tbb::parallel_reduce(tbb::blocked_range<size_t>(0, images.size(), 8), body);

  SamplesForQuantization result;
  result.group_pixel_count = std::move(body.group_pixel_count);
  result.channel_pixel_count = std::move(body.channel_pixel_count);
  flatten(body.pixel_samples, result.pixel_samples);
  flatten(body.diff_samples, result.diff_samples);
  return result;
}

void InitializeTreeSamples(TreeSamples &tree_samples,
                           const std::vector<uint32_t> &props_to_use,
                           size_t max_property_values,
                           SamplesForQuantization &samples_for_quantization) {
  if (!tree_samples.SetPredictor(Predictor::Gradient,
                                 ModularOptions::TreeMode::kNoWP))
    JXL_ABORT("SetPredictor failed");

  if (!tree_samples.SetProperties(props_to_use,
                                  ModularOptions::TreeMode::kNoWP))
    JXL_ABORT("SetProperty failed");

  std::vector<ModularMultiplierInfo> dummy_multiplier_info;
  StaticPropRange range;
  range[0] = {{0, static_cast<uint32_t>(
                      samples_for_quantization.channel_pixel_count.size())}};
  range[1] = {{0, 1}};

  tree_samples.PreQuantizeProperties(
      range, dummy_multiplier_info, samples_for_quantization.group_pixel_count,
      samples_for_quantization.channel_pixel_count,
      samples_for_quantization.pixel_samples,
      samples_for_quantization.diff_samples, max_property_values);
}

BlockPropertyDistributions ExtractPropertiesFromBlock(
    const Image &image, const Rect &block, const ModularOptions &options,
    const TreeSamples &quantizer) {
  JXL_ASSERT(options.nb_repeats > 0 && options.nb_repeats <= 1);
  bool use_all_pixels = options.nb_repeats >= 1;
  size_t n_pixels_to_sample =
      use_all_pixels ? block.xsize() * block.ysize()
                     : static_cast<size_t>(block.xsize() * block.ysize() *
                                           options.nb_repeats);
  JXL_ASSERT(n_pixels_to_sample > 0);

  // 座標をサンプリングするために、座標リストをつくる
  // FIXME: 無駄なメモリすぎる
  typedef std::pair<size_t, size_t> PointYX;
  std::vector<PointYX> points(block.ysize() * block.xsize());
  for (size_t y = 0; y < block.ysize(); y++) {
    for (size_t x = 0; x < block.xsize(); x++) {
      points.at(y * block.xsize() + x) = {block.y0() + y, block.x0() + x};
    }
  }

  std::random_device seed_gen;  // TODO: シードを固定できるようにする
  std::mt19937 engine{seed_gen()};
  std::vector<PointYX> sampling_points(n_pixels_to_sample);
  if (use_all_pixels) sampling_points = points;

  TreeSamples tree_samples = quantizer;  // Copy

  std::vector<std::vector<uint8_t>> quantized_values(
      tree_samples.NumProperties());
  for (std::vector<uint8_t> &v : quantized_values)
    v.reserve(n_pixels_to_sample *
              (image.channel.size() - image.nb_meta_channels));

  // GatherTreeData をパクっていく

  Properties properties(kNumNonrefProperties +
                        kExtraPropsPerChannel * options.max_properties);

  for (size_t i = image.nb_meta_channels; i < image.channel.size(); i++) {
    const Channel &channel = image.channel[i];
    if (!channel.w || !channel.h) {
      JXL_ABORT("empty channel %" PRIuS, i);
    }
    if (channel.w > options.max_chan_size ||
        channel.h > options.max_chan_size) {
      break;
    }

    // 使用するピクセルを選択
    if (!use_all_pixels) {
      std::sample(points.cbegin(), points.cend(), sampling_points.begin(),
                  n_pixels_to_sample, engine);
      std::sort(sampling_points.begin(), sampling_points.end());
    }
    auto next_px = sampling_points.cbegin();

    tree_samples.PrepareForSamples(n_pixels_to_sample);

    std::array<pixel_type, kNumStaticProperties> static_props = {
        {static_cast<pixel_type>(i), 0}};
    const intptr_t onerow = channel.plane.PixelsPerRow();
    Channel references(properties.size() - kNumNonrefProperties, channel.w);

    // 使わないけど、 PredictLearn の呼び出しに必要なので
    weighted::Header wp_header;
    weighted::State wp_state(wp_header, channel.w, channel.h);

    while (next_px != sampling_points.cend()) {
      size_t y = next_px->first;
      const pixel_type *p = channel.Row(y);
      PrecomputeReferences(channel, y, image, i, &references);
      InitPropsRow(&properties, static_props, y);

      for (; next_px != sampling_points.cend() && next_px->first == y;
           ++next_px) {
        size_t x = next_px->second;
        PredictLearn(&properties, channel.w, p + x, onerow, x, y,
                     Predictor::Gradient, references, &wp_state);
        for (size_t prop_idx = 0; prop_idx < tree_samples.NumProperties();
             prop_idx++) {
          // 内部では uint8_t なので、キャストできる
          auto qv = static_cast<uint8_t>(tree_samples.QuantizeProperty(
              prop_idx, properties[tree_samples.PropertyFromIndex(prop_idx)]));
          quantized_values[prop_idx].push_back(qv);
        }
      }
    }
  }

  // プロパティ値の平均と標準偏差を求める
  BlockPropertyDistributions results(tree_samples.NumProperties());
  for (size_t prop_idx = 0; prop_idx < tree_samples.NumProperties();
       prop_idx++) {
    std::vector<uint8_t> &values = quantized_values[prop_idx];
    uint64_t sum = 0;
    for (auto v : values) sum += v;
    float mean = static_cast<float>(sum) / values.size();
    float variance = 0;
    for (auto v : values) variance += (v - mean) * (v - mean);
    variance /= values.size();
    results[prop_idx] = {mean, sqrtf(variance)};
  }

  return results;
}

std::vector<Rect> SplitRect(size_t w, size_t h, size_t split) {
  std::vector<Rect> rects = {{0, 0, w, h}};
  std::vector<Rect> tmp_rects;

  for (size_t i = 0; i < split; i++) {
    tmp_rects.clear();
    tmp_rects.reserve(rects.size() * 2);

    if (i % 2 == 0) {
      // 縦を分割（水平）
      for (Rect &r : rects) {
        size_t half = r.ysize() / 2;
        JXL_ASSERT(half > 0);
        tmp_rects.emplace_back(r.x0(), r.y0(), r.xsize(), half);
        tmp_rects.emplace_back(r.x0(), r.y0() + half, r.xsize(),
                               r.ysize() - half);
      }
    } else {
      // 横を分割（垂直）
      for (Rect &r : rects) {
        size_t half = r.xsize() / 2;
        JXL_ASSERT(half > 0);
        tmp_rects.emplace_back(r.x0(), r.y0(), half, r.ysize());
        tmp_rects.emplace_back(r.x0() + half, r.y0(), r.xsize() - half,
                               r.ysize());
      }
    }

    std::swap(rects, tmp_rects);
  }

  return rects;
}

ImagePropertyVector ExtractPropertiesFromImage(
    const jxl::Image &image, size_t split, const jxl::ModularOptions &options,
    const jxl::TreeSamples &quantizer,
    std::vector<std::string> *out_vector_descriptions) {
  const auto rects = SplitRect(image.w, image.h, split);
  const size_t veclen_per_block = quantizer.NumProperties() * 2;
  ImagePropertyVector result(rects.size() * veclen_per_block);

  tbb::parallel_for(size_t(0), rects.size(), [&](size_t block_idx) {
    auto block_result =
        ExtractPropertiesFromBlock(image, rects[block_idx], options, quantizer);
    JXL_ASSERT(block_result.size() == quantizer.NumProperties());

    size_t i = block_idx * veclen_per_block;
    for (auto &x : block_result) {
      result[i++] = x.mean;
      result[i++] = x.stdev;
    }
    JXL_ASSERT(i <= result.size());
    JXL_ASSERT(i == (block_idx + 1) * veclen_per_block);
  });

  if (out_vector_descriptions != nullptr) {
    for (size_t block_idx = 0; block_idx < rects.size(); block_idx++) {
      for (size_t prop_idx = 0; prop_idx < quantizer.NumProperties();
           prop_idx++) {
        std::string prop_name =
            jxl::PropertyName(quantizer.PropertyFromIndex(prop_idx));
        out_vector_descriptions->push_back(
            fmt::format("block{:02} {} mean", block_idx, prop_name));
        out_vector_descriptions->push_back(
            fmt::format("block{:02} {} stdev", block_idx, prop_name));
      }
    }
  }

  return result;
}

}  // namespace research
