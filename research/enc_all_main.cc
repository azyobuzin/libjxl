// クラスタリングから圧縮まで全部やる

#include <fmt/core.h>
#include <tbb/parallel_for.h>

#include <boost/program_options.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mlpack/core.hpp>
#include <set>

#include "common_cluster.h"
#include "enc_all.h"
#include "enc_brute_force.h"
#include "prop_extract.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace std::chrono;
using namespace research;

namespace {

template <typename CreateGraphFunction>
std::vector<EncodedCombinedImage> EncodeImages(
    ImagesProvider& images, const jxl::ModularOptions& options,
    const EncodingOptions& encoding_options, bool use_brute_force,
    CreateGraphFunction create_graph) {
  auto tree = ComputeMstFromGraph(create_graph());
  return use_brute_force ? EncodeWithBruteForce<>(images, tree, options,
                                                  encoding_options, nullptr)
                         : EncodeWithCombineAll<>(images, tree, options,
                                                  encoding_options, nullptr);
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
    ("random", po::bool_switch(), "乱数シードをランダムに設定する")
    // クラスタリング / エンコード
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    // エンコード
    ("cost", po::value<std::string>()->default_value("tree"), "MSTに使用するコスト tree: JPEG XL決定木入れ替え, y: Yチャネル, props: JPEG XLプロパティ, random")
    ("refchan", po::value<uint16_t>()->default_value(0), "画像内のチャンネル参照数")
    ("parent-ref", po::value<int>()->default_value(4), "0: 参照なし, 1: 親の同チャネル参照, 2: 親の全チャネル参照, 3: 前フレーム同チャネル参照, 4: 前フレーム全チャネル参照")
    ("speed", po::value<uint16_t>()->default_value(1), "1: tortoise, 2: kitten, 3: squirrel")
    ("flif", po::bool_switch(), "色チャネルをFLIFで符号化")
    ("flif-learn", po::value<int>()->default_value(2), "FLIF学習回数")
    ("enc-method", po::value<std::string>()->default_value("brute-force"), "brute-force or combine-all")
    ("out-dir", po::value<fs::path>()->required(), "圧縮結果の出力先ディレクトリ")
    ("time", po::bool_switch(), "時間計測する");
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

  if (vm["random"].as<bool>()) {
    mlpack::math::RandomSeed(static_cast<size_t>(std::time(nullptr)));
  }

  const size_t split = vm["split"].as<uint16_t>();
  const float fraction = vm["fraction"].as<float>();
  const std::string& cluster_method = vm["clustering"].as<std::string>();
  const size_t k = vm["k"].as<uint16_t>();
  const int margin = vm["margin"].as<uint16_t>();
  const std::string& cost = vm["cost"].as<std::string>();
  const bool flif_enabled = vm["flif"].as<bool>();
  const std::string& enc_method = vm["enc-method"].as<std::string>();
  const bool measure_time = vm["time"].as<bool>();

  bool use_brute_force = false;
  if (enc_method == "brute-force") {
    use_brute_force = true;
  } else if (enc_method != "combine-all") {
    JXL_ABORT("Invalid enc-method '%s'", enc_method.c_str());
  }

  const std::vector<std::string>& paths =
      vm["image-file"].as<std::vector<std::string>>();
  FileImagesProvider images(paths);
  images.ycocg = true;
  images.only_first_channel = flif_enabled;

  // クラスタリング
  std::cerr << "Clustering" << std::endl;
  auto clustering_start = steady_clock::now();
  arma::Row<size_t> assignments =
      ClusterImages(split, fraction, cluster_method, k, margin, images);
  size_t n_clusters =
      *std::max_element(assignments.cbegin(), assignments.cend()) + 1;

  if (measure_time) {
    auto clustering_end = steady_clock::now();
    std::cout << "Clustering Time: "
              << duration<double>(clustering_end - clustering_start).count()
              << " s" << std::endl;
  }

  const int refchan = vm["refchan"].as<uint16_t>();
  const jxl::ParentReferenceType parent_ref =
      static_cast<jxl::ParentReferenceType>(vm["parent-ref"].as<int>());
  const int flif_learn_repeats = vm["flif-learn"].as<int>();
  const fs::path& out_dir = vm["out-dir"].as<fs::path>();

  fs::create_directories(out_dir);

  auto encoding_start = steady_clock::now();

  // Tortoise相当
  jxl::ModularOptions options{
      .nb_repeats = fraction,
      .max_properties = refchan,
      .splitting_heuristics_properties = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3,
                                          4, 5, 6, 7, 8},
      .max_property_values = 256,
      .predictor = jxl::Predictor::Variable};

  auto speed = static_cast<jxl::SpeedTier>(vm["speed"].as<uint16_t>());
  if (speed >= jxl::SpeedTier::kSquirrel) {
    options.splitting_heuristics_properties.resize(8);
    options.max_property_values = 32;
  } else if (speed >= jxl::SpeedTier::kKitten) {
    options.splitting_heuristics_properties.resize(10);
    options.max_property_values = 64;
  }

  EncodingOptions encoding_options{parent_ref, flif_enabled,
                                   flif_learn_repeats};

  std::atomic_size_t n_completed_clusters = 0;
  ConsoleProgressReporter progress("Encoding");
  std::atomic_bool failed = false;

  std::function<std::vector<EncodedCombinedImage>(ImagesProvider&)>
      encode_cluster;
  if (cost == "tree") {
    encode_cluster = [&](ImagesProvider& cluster_images) {
      return EncodeImages(cluster_images, options, encoding_options,
                          use_brute_force, [&]() {
                            return CreateGraphWithDifferentTree(
                                cluster_images, options, nullptr);
                          });
    };
  } else if (cost == "y") {
    encode_cluster = [&](ImagesProvider& cluster_images) {
      return EncodeImages(cluster_images, options, encoding_options,
                          use_brute_force, [&]() {
                            return CreateGraphWithYDistance(
                                cluster_images, kSelfCostJxl, options, nullptr);
                          });
    };
  } else if (cost == "props") {
    encode_cluster = [&](ImagesProvider& cluster_images) {
      return EncodeImages(
          cluster_images, options, encoding_options, use_brute_force, [&]() {
            return CreateGraphWithPropsDistance(cluster_images, kSelfCostJxl,
                                                split, fraction, options,
                                                nullptr);
          });
    };
  } else if (cost == "random") {
    encode_cluster = [&](ImagesProvider& cluster_images) {
      return EncodeImages(cluster_images, options, encoding_options,
                          use_brute_force, [&]() {
                            return CreateGraphWithRandomCost(
                                cluster_images, kSelfCostJxl, options, nullptr);
                          });
    };
  } else {
    JXL_ABORT("Invalid cost '%s'", cost.c_str());
  }

  // クラスタごとに圧縮
  tbb::parallel_for(size_t(0), n_clusters, [&](size_t cluster_idx) {
    std::vector<std::string> cluster_inputs;
    for (size_t i = 0; i < assignments.size(); i++) {
      if (assignments[i] == cluster_idx) cluster_inputs.push_back(paths.at(i));
    }

    if (cluster_inputs.empty()) {
      // クラスタに属する画像がないならば、読み取られることはないので ClusterFile は出力しなくて良い
    } else {
      FileImagesProvider cluster_images(std::move(cluster_inputs));
      cluster_images.ycocg = true;

      std::vector<EncodedCombinedImage> results =
          encode_cluster(cluster_images);

      fs::path out_path = out_dir / fmt::format("cluster{}.bin", cluster_idx);
      std::ofstream dst(out_path, std::ios_base::out | std::ios_base::binary);
      if (dst) {
        PackToClusterFile(results, parent_ref, dst);
        dst.flush();
      }

      if (!dst) {
        std::cerr << "Failed to write " << out_path.c_str() << std::endl;
        failed = true;
      }
    }

    progress.report(++n_completed_clusters, n_clusters);
  });

  images.ycocg = false;
  images.only_first_channel = false;
  auto first_image = images.get(0);
  WriteIndexFile(first_image.w, first_image.h,
                 first_image.channel.size() - first_image.nb_meta_channels,
                 n_clusters, assignments, out_dir);

  if (measure_time) {
    auto encoding_end = steady_clock::now();
    std::cout << "Encoding Time: "
              << duration<double>(encoding_end - encoding_start).count() << " s"
              << std::endl;
  }

  return 0;
}
