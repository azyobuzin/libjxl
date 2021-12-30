// クラスタリングから圧縮まで全部やる

#include <fmt/core.h>
#include <tbb/parallel_for.h>

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <set>

#include "cocbo.h"
#include "common_cluster.h"
#include "enc_brute_force.h"
#include "prop_extract.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace research;
using namespace mlpack::kmeans;

namespace {

arma::Row<size_t> ClusterImages(size_t split, float fraction,
                                const std::string& method, size_t k, int margin,
                                ImagesProvider& images) {
  jxl::ModularOptions options{.nb_repeats = fraction};
  std::vector<uint32_t> props_to_use(std::cbegin(kPropsToUse),
                                     std::cend(kPropsToUse));
  jxl::TreeSamples tree_samples;

  // 量子化方法を決定するために適当なサンプリング
  SamplesForQuantization samples_for_quantization =
      CollectSamplesForQuantization(images, options);
  InitializeTreeSamples(tree_samples, props_to_use, options.max_property_values,
                        samples_for_quantization);

  const size_t n_rows =
      (size_t(2) << split /* 2^split * 2 */) * props_to_use.size();
  arma::mat prop_mat(n_rows, images.size(), arma::fill::none);

  // 特徴量を prop_mat に代入していく
  tbb::parallel_for(size_t(0), images.size(), [&](size_t i) {
    auto img = images.get(i);
    auto result =
        ExtractPropertiesFromImage(img, split, options, tree_samples, nullptr);
    JXL_ASSERT(result.size() == n_rows);
    auto col = prop_mat.col(i);
    std::copy(result.begin(), result.end(), col.begin());
  });

  // クラスタリング
  arma::Row<size_t> assignments;
  if (method == "kmeans") {
    KMeans<> model;
    model.Cluster(prop_mat, k, assignments);
  } else if (method == "cocbo") {
    ClusterWithCocbo(prop_mat, k, std::max(static_cast<int>(k) - margin, 0),
                     k + 1 + margin, assignments);
  } else {
    JXL_ABORT("method is invalid");
  }

  JXL_ASSERT(assignments.size() == images.size());

  return assignments;
}

void WriteIndexFile(uint32_t width, uint32_t height, uint32_t n_channel,
                    uint32_t n_clusters, const arma::Row<size_t>& assignments,
                    const fs::path& out_dir) {
  IndexFields fields;
  fields.width = width;
  fields.height = height;
  fields.n_channel = n_channel;
  fields.n_clusters = n_clusters;
  fields.assignments.resize(assignments.size());
  std::copy(assignments.cbegin(), assignments.cend(),
            fields.assignments.begin());

  jxl::BitWriter writer;
  JXL_CHECK(jxl::Bundle::Write(fields, &writer, 0, nullptr));
  writer.ZeroPadToByte();
  jxl::Span<const uint8_t> span = writer.GetSpan();

  fs::path out_path = out_dir / "index.bin";
  FILE* fp = fopen(out_path.c_str(), "wb");
  bool failed =
      fwrite(span.data(), sizeof(uint8_t), span.size(), fp) != span.size();
  failed |= fclose(fp) != 0;

  if (failed) JXL_ABORT("Failed to write %s", out_path.c_str());
}

}  // namespace

int main(int argc, char* argv[]) {
  po::options_description pos_ops;
  pos_ops.add_options()(
      "image-file",
      po::value<std::vector<std::string>>()->multitoken()->required());

  po::positional_options_description pos_desc;
  pos_desc.add("image-file", -1);

  po::options_description ops_desc;
  // clang-format off
  ops_desc.add_options()
    // クラスタリング
    ("split", po::value<uint16_t>()->default_value(2), "画像を何回分割するか")
    ("clustering", po::value<std::string>()->default_value("cocbo"), "kmeans or cocbo")
    ("k", po::value<uint16_t>()->default_value(2), "kmeansの場合はクラスタ数。cocboの場合はクラスタあたりの画像数")
    ("margin", po::value<uint16_t>()->default_value(2), "(cocbo) kのマージン")
    // クラスタリング / エンコード
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    // エンコード
    ("refchan", po::value<uint16_t>()->default_value(0), "画像内のチャンネル参照数")
    ("max-refs", po::value<size_t>()->default_value(1), "画像の参照数")
    ("flif", po::bool_switch(), "色チャネルをFLIFで符号化")
    ("flif-learn", po::value<int>()->default_value(2), "FLIF学習回数")
    ("out-dir", po::value<fs::path>()->required(), "圧縮結果の出力先ディレクトリ");
  // clang-format on

  po::options_description all_desc;
  all_desc.add(pos_ops).add(ops_desc);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(all_desc)
                  .positional(pos_desc)
                  .run(),
              vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl
              << std::endl
              << "Usage: enc_all [OPTIONS] IMAGE_FILE..." << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  const size_t split = vm["split"].as<uint16_t>();
  const float fraction = vm["fraction"].as<float>();
  const std::string& method = vm["clustering"].as<std::string>();
  const size_t k = vm["k"].as<uint16_t>();
  const int margin = vm["margin"].as<uint16_t>();
  const bool flif_enabled = vm["flif"].as<bool>();

  const std::vector<std::string>& paths =
      vm["image-file"].as<std::vector<std::string>>();
  FileImagesProvider images(paths);
  images.ycocg = true;
  images.only_first_channel = flif_enabled;

  // クラスタリング
  std::cerr << "Clustering" << std::endl;
  arma::Row<size_t> assignments =
      ClusterImages(split, fraction, method, k, margin, images);
  size_t n_clusters =
      *std::max_element(assignments.cbegin(), assignments.cend()) + 1;

  int refchan = vm["refchan"].as<uint16_t>();
  size_t max_refs = vm["max-refs"].as<size_t>();
  int flif_learn_repeats = vm["flif-learn"].as<int>();
  const fs::path& out_dir = vm["out-dir"].as<fs::path>();

  fs::create_directories(out_dir);

  // Tortoise相当
  jxl::ModularOptions options{
      .nb_repeats = fraction,
      .max_properties = refchan,
      .splitting_heuristics_properties = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3,
                                          4, 5, 6, 7, 8},
      .max_property_values = 256,
      .predictor = jxl::Predictor::Variable};
  EncodingOptions encoding_options{max_refs, flif_enabled, flif_learn_repeats};

  std::atomic_size_t n_completed_clusters = 0;
  ConsoleProgressReporter progress("Encoding");
  std::atomic_bool failed = false;

  // クラスタごとに圧縮
  tbb::parallel_for(size_t(0), n_clusters, [&](size_t cluster_idx) {
    std::vector<std::string> cluster_inputs;
    for (size_t i = 0; i < assignments.size(); i++) {
      if (assignments[i] == cluster_idx) cluster_inputs.push_back(paths.at(i));
    }

    FileImagesProvider cluster_images(std::move(cluster_inputs));
    cluster_images.ycocg = true;

    auto tree = CreateMstWithDifferentTree(cluster_images, options, nullptr);
    auto results = EncodeWithBruteForce(cluster_images, tree, options,
                                        encoding_options, nullptr);

    fs::path out_path = out_dir / fmt::format("cluster{}.bin", cluster_idx);
    std::ofstream dst(out_path, std::ios_base::out | std::ios_base::binary);
    if (dst) {
      PackToClusterFile(results, dst);
      dst.flush();
    }

    if (!dst) {
      std::cerr << "Failed to write " << out_path.c_str() << std::endl;
      failed = true;
    }

    progress.report(++n_completed_clusters, n_clusters);
  });

  images.ycocg = false;
  images.only_first_channel = false;
  auto first_image = images.get(0);
  WriteIndexFile(first_image.w, first_image.h,
                 first_image.channel.size() - first_image.nb_meta_channels,
                 n_clusters, assignments, out_dir);

  return 0;
}
