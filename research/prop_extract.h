// 画像からプロパティ値をサンプリングする

#pragma once

#include <mlpack/core.hpp>
#include <string>
#include <vector>

#include "images_provider.h"
#include "lib/jxl/modular/encoding/enc_ma.h"

namespace research {

struct ValueDistribution {
  double mean;
  double stdev;
};

typedef std::vector<ValueDistribution> BlockPropertyDistributions;

// BlockPropertyDistributions を展開したもの
typedef arma::vec ImagePropertyVector;

struct SamplesForQuantization {
  std::vector<uint32_t> group_pixel_count;
  std::vector<uint32_t> channel_pixel_count;
  std::vector<jxl::pixel_type> pixel_samples;
  std::vector<jxl::pixel_type> diff_samples;
};

// gradient, W-NW, NW-N, N-NE, N-NN (from splitting_heuristics_properties)
const uint32_t kPropsToUse[] = {9, 10, 11, 12, 13};

// プロパティ値の量子化に使用するヒストグラムの材料を収集
SamplesForQuantization CollectSamplesForQuantization(
    ImagesProvider &images_provider, const jxl::ModularOptions &options);

// TreeSamples をプロパティ値の量子化に使えるように初期化する
void InitializeTreeSamples(jxl::TreeSamples &tree_samples,
                           const std::vector<uint32_t> &props_to_use,
                           size_t max_property_values,
                           SamplesForQuantization &samples_for_quantization);

// 画像の一部からその特徴を抽出する
BlockPropertyDistributions ExtractPropertiesFromBlock(
    const jxl::Image &image, const jxl::Rect &block,
    const jxl::ModularOptions &options, const jxl::TreeSamples &quantizer);

// 画像を split 回分割して、画像の特徴を抽出し、ベクトルとして並べる
ImagePropertyVector ExtractPropertiesFromImage(
    const jxl::Image &image, size_t split, const jxl::ModularOptions &options,
    const jxl::TreeSamples &quantizer,
    std::vector<std::string> *out_vector_descriptions =
        nullptr /**< [out] \c nullptr でなければ、結果の各要素の説明を出力する */);

// ExtractPropertiesFromImage をすべての画像に対して適用した行列を作成する
void CreatePropertyMatrix(ImagesProvider &images, size_t split,
                          const jxl::ModularOptions &options,
                          const jxl::TreeSamples &quantizer,
                          arma::mat &out_mat);

}  // namespace research
