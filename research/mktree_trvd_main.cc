// treevideo 形式の MST を出力する

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <tbb/parallel_for.h>

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>
#include <mlpack/core.hpp>
#include <stack>

#include "common_cluster.h"
#include "cost_graph.h"
#include "enc_all.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace std::chrono;
using namespace research;

namespace {

template <typename CreateGraphFunction>
std::function<void(ImagesProvider& images, std::string& out_line)>
MakeComputeMstFunction(CreateGraphFunction create_graph) {
  return [f = std::move(create_graph)](ImagesProvider& images,
                                       std::string& out_line) {
    if (images.size() == 0) {
      out_line.clear();
      return;
    }

    out_line = "Edge: root";

    auto tree = ComputeMstFromGraph(f(images));
    std::stack<int32_t> stack;
    stack.push(tree.root);

    while (!stack.empty()) {
      const auto& node = tree.nodes.at(stack.top());
      stack.pop();

      fmt::format_to(std::back_inserter(out_line), " -> {}",
                     images.get_label(node.image_idx));

      // コストの小さい順を得る
      std::vector<std::pair<
          typename std::remove_reference<decltype(node.self_cost)>::type,
          size_t>>
          costs;
      costs.reserve(node.children.size());
      for (const auto& edge : node.children)
        costs.emplace_back(edge.cost, edge.target);
      std::sort(costs.begin(), costs.end());

      for (const auto& [cost, child_idx] : costs) stack.push(child_idx);
    }

    out_line += '\n';
  };
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
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    ("clustering", po::value<std::string>()->default_value("cocbo"), "kmeans or cocbo")
    ("k", po::value<uint16_t>()->default_value(2), "kmeansの場合はクラスタ数。cocboの場合はクラスタあたりの画像数")
    ("margin", po::value<uint16_t>()->default_value(2), "(cocbo) kのマージン")
    ("random", po::bool_switch(), "乱数シードをランダムに設定する")
    ("cost", po::value<std::string>()->default_value("tree"), "MSTに使用するコスト tree: JPEG XL決定木入れ替え, y: Yチャネル, props: JPEG XLプロパティ, random")
    ("refchan", po::value<uint16_t>()->default_value(0), "画像内のチャンネル参照数")
    ("out", po::value<fs::path>()->required(), "出力先テキストファイル");
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
              << "Usage: mktree_trvd_main [OPTIONS] IMAGE_FILE..." << std::endl
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
  const int refchan = vm["refchan"].as<uint16_t>();

  const std::vector<std::string>& paths =
      vm["image-file"].as<std::vector<std::string>>();
  FileImagesProvider images(paths);
  images.ycocg = true;

  // クラスタリング
  std::cerr << "Clustering" << std::endl;
  arma::Row<size_t> assignments =
      ClusterImages(split, fraction, cluster_method, k, margin, images);
  size_t n_clusters =
      *std::max_element(assignments.cbegin(), assignments.cend()) + 1;

  // Tortoise相当
  jxl::ModularOptions options{
      .nb_repeats = fraction,
      .max_properties = refchan,
      .splitting_heuristics_properties = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3,
                                          4, 5, 6, 7, 8},
      .max_property_values = 256,
      .predictor = jxl::Predictor::Variable};

  std::atomic_size_t n_completed_clusters = 0;
  ConsoleProgressReporter progress("Computing MST");

  std::function<void(ImagesProvider&, std::string&)> compute_mst;
  if (cost == "tree") {
    compute_mst = MakeComputeMstFunction([&](ImagesProvider& cluster_images) {
      return CreateGraphWithDifferentTree(cluster_images, options, nullptr);
    });
  } else if (cost == "y") {
    compute_mst = MakeComputeMstFunction([&](ImagesProvider& cluster_images) {
      return CreateGraphWithYDistance(cluster_images, kSelfCostFlif, options,
                                      nullptr);
    });
  } else if (cost == "props") {
    compute_mst = MakeComputeMstFunction([&](ImagesProvider& cluster_images) {
      return CreateGraphWithPropsDistance(cluster_images, kSelfCostFlif, split,
                                          fraction, options, nullptr);
    });
  } else if (cost == "random") {
    compute_mst = MakeComputeMstFunction([&](ImagesProvider& cluster_images) {
      return CreateGraphWithRandomCost(cluster_images, kSelfCostFlif, options,
                                       nullptr);
    });
  } else {
    JXL_ABORT("Invalid cost '%s'", cost.c_str());
  }

  // クラスタごとに MST を求める
  std::vector<std::string> edge_lines(n_clusters);

  tbb::parallel_for(size_t(0), n_clusters, [&](size_t cluster_idx) {
    std::vector<std::string> cluster_inputs;
    for (size_t i = 0; i < assignments.size(); i++) {
      if (assignments[i] == cluster_idx) cluster_inputs.push_back(paths.at(i));
    }

    if (!cluster_inputs.empty()) {
      FileImagesProvider cluster_images(std::move(cluster_inputs));
      cluster_images.ycocg = true;

      compute_mst(cluster_images, edge_lines[cluster_idx]);
    }

    progress.report(++n_completed_clusters, n_clusters);
  });

  const auto& output_path = vm["out"].as<fs::path>();
  auto output_base = fs::weakly_canonical(output_path).parent_path();
  std::ofstream out_stream(output_path);
  if (!out_stream) {
    std::cerr << "Failed to open " << output_path << std::endl;
    return 2;
  }

  for (size_t i = 0; i < paths.size(); i++) {
    auto relative_path = fs::relative(paths[i], output_base);
    fmt::print(out_stream, "Node: {}, {}\n", images.get_label(i),
               relative_path.c_str());
  }

  for (const auto& line : edge_lines) out_stream << line;

  out_stream.close();
  if (!out_stream) {
    std::cerr << "I/O error" << std::endl;
    return 2;
  }

  return 0;
}
