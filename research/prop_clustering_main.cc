// 画像からプロパティを抽出し、それをk-meansでクラスタリングする

#include <fmt/core.h>
#include <tbb/parallel_for.h>

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <set>
#include <unordered_map>

#include "cocbo.h"
#include "prop_extract.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace research;
using namespace mlpack::kmeans;

int main(int argc, char *argv[]) {
  po::options_description pos_ops;
  pos_ops.add_options()(
      "image-file",
      po::value<std::vector<std::string>>()->multitoken()->required());

  po::positional_options_description pos_desc;
  pos_desc.add("image-file", -1);

  po::options_description ops_desc;
  // clang-format off
  ops_desc.add_options()
    ("split", po::value<uint16_t>()->default_value(2), "画像を何回分割するか")
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    ("y-only", po::bool_switch(), "Yチャネルのみを利用する")
    ("method", po::value<std::string>()->default_value("kmeans"), "kmeans or cocbo")
    ("k", po::value<uint16_t>()->default_value(2), "kmeansの場合はクラスタ数。cocboの場合はクラスタあたりの画像数")
    ("margin", po::value<uint16_t>()->default_value(2), "(cocbo) kのマージン")
    ("copy-to", po::value<fs::path>(), "クラスタリングされた画像をディレクトリにコピーする");
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
  } catch (const po::error &e) {
    std::cerr << e.what() << std::endl
              << std::endl
              << "Usage: prop_kmeans [OPTIONS] IMAGE_FILE..." << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  const size_t split = vm["split"].as<uint16_t>();
  const float fraction = vm["fraction"].as<float>();
  const std::string &method = vm["method"].as<std::string>();
  const size_t k = vm["k"].as<uint16_t>();
  const int margin = vm["margin"].as<uint16_t>();

  const std::vector<std::string> &paths =
      vm["image-file"].as<std::vector<std::string>>();
  FileImagesProvider images(paths);
  images.ycocg = true;
  images.only_first_channel = vm["y-only"].as<bool>();

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
  arma::mat prop_mat(n_rows, paths.size(), arma::fill::none);

  // 特徴量を prop_mat に代入していく
  tbb::parallel_for(size_t(0), paths.size(), [&](size_t i) {
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
    ClusterWithCocbo(prop_mat, k, std::max(static_cast<int>(k) - margin, 1),
                     k + 1 + margin, assignments);
  } else {
    std::cerr << "method is invalid" << std::endl;
    return 1;
  }

  JXL_ASSERT(assignments.size() == paths.size());

  std::set<size_t> available_clusters;
  for (size_t i = 0; i < assignments.size(); i++)
    available_clusters.insert(assignments[i]);

  for (size_t i : available_clusters) {
    std::cout << "=== Cluster " << i << " ===" << std::endl;
    for (size_t j = 0; j < paths.size(); j++) {
      if (assignments[j] == i) std::cout << paths[j] << std::endl;
    }
    std::cout << std::endl;
  }

  if (!vm["copy-to"].empty()) {
    // 出力先ディレクトリ作成
    const fs::path &dst_dir = vm["copy-to"].as<fs::path>();
    std::unordered_map<size_t, fs::path> dst_dirs;
    dst_dirs.reserve(k);
    for (size_t i : available_clusters) {
      const auto p =
          dst_dirs.emplace(i, dst_dir / fmt::format("cluster{:02}", i));
      JXL_ASSERT(p.second);
      fs::create_directories(p.first->second);
    }

    // クラスタごとのディレクトリにコピー
    for (size_t i = 0; i < paths.size(); i++) {
      fs::path src = paths[i];
      fs::copy(src, dst_dirs.at(assignments[i]) / src.filename(),
               fs::copy_options::overwrite_existing);
    }
  }

  return 0;
}
