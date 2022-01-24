#include <boost/graph/graphviz.hpp>
#include <boost/program_options.hpp>
#include <iostream>

#include "cost_graph_util.h"

namespace po = boost::program_options;

using namespace research;

namespace {

struct ImageVertexLabelWriter {
  ImagesProvider &images;

  void operator()(std::ostream &out, size_t vertex_idx) const {
    out << "[label=" << boost::escape_dot_string(images.get_label(vertex_idx))
        << "]";
  }
};

template <typename F>
auto WithProgress(F f) {
  ConsoleProgressReporter progress("Working");
  return f(static_cast<ProgressReporter *>(&progress));
}

template <typename Cost>
void PrintDot(const BidirectionalCostGraphResult<Cost> &gr,
              ImagesProvider &images, bool mst) {
  if (mst) {
    auto tree = ComputeMstFromGraph(gr);
    PrintImageTreeDot(std::cout, tree, &images);
  } else {
    auto &graph = gr.graph;
    boost::write_graphviz(
        std::cout, graph, ImageVertexLabelWriter{images},
        boost::make_label_writer(get(boost::edge_weight_t(), graph)));
  }
}

}  // namespace

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
    ("split", po::value<uint16_t>()->default_value(2), "画像を何回分割するか（cost = props-* のみ）")
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    ("y-only", po::bool_switch(), "Yチャネルのみを利用する")
    ("cost", po::value<std::string>()->default_value("tree"), "tree: JPEG XL決定木入れ替え, y-jxl: Yチャネル（自己コスト JPEG XL）, y-flif: Yチャネル（自己コスト FLIF）, props-jxl: JPEG XLプロパティ（自己コスト JPEG XL）, props-flif: JPEG XLプロパティ（自己コスト FLIF）, random-jxl, random-flif")
    ("mst", po::bool_switch(), "MSTを求める");
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
              << "Usage: cost_graph [OPTIONS] IMAGE_FILE..." << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  const std::vector<std::string> &paths =
      vm["image-file"].as<std::vector<std::string>>();
  FileImagesProvider images(paths);
  images.ycocg = true;
  images.only_first_channel = vm["y-only"].as<bool>();

  const size_t split = vm["split"].as<uint16_t>();
  const float fraction = vm["fraction"].as<float>();

  // Tortoise相当
  jxl::ModularOptions options{
      .nb_repeats = fraction,
      .splitting_heuristics_properties = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3,
                                          4, 5, 6, 7, 8},
      .max_property_values = 256,
      .predictor = jxl::Predictor::Variable};

  const std::string &cost = vm["cost"].as<std::string>();
  bool mst = vm["mst"].as<bool>();

  if (cost == "tree") {
    PrintDot(WithProgress([&](ProgressReporter *progress) {
               return CreateGraphWithDifferentTree(images, options, progress);
             }),
             images, mst);
  } else if (cost == "y-jxl") {
    PrintDot(WithProgress([&](ProgressReporter *progress) {
               return CreateGraphWithYDistance(images, kSelfCostJxl, options,
                                               progress);
             }),
             images, mst);
  } else if (cost == "y-flif") {
    PrintDot(WithProgress([&](ProgressReporter *progress) {
               return CreateGraphWithYDistance(images, kSelfCostFlif, options,
                                               progress);
             }),
             images, mst);
  } else if (cost == "props-jxl") {
    PrintDot(WithProgress([&](ProgressReporter *progress) {
               return CreateGraphWithPropsDistance(images, kSelfCostJxl, split,
                                                   fraction, options, progress);
             }),
             images, mst);
  } else if (cost == "props-flif") {
    PrintDot(WithProgress([&](ProgressReporter *progress) {
               return CreateGraphWithPropsDistance(images, kSelfCostFlif, split,
                                                   fraction, options, progress);
             }),
             images, mst);
  } else if (cost == "random-jxl") {
    PrintDot(WithProgress([&](ProgressReporter *progress) {
               return CreateGraphWithRandomCost(images, kSelfCostJxl, options,
                                                progress);
             }),
             images, mst);
  } else if (cost == "random-flif") {
    PrintDot(WithProgress([&](ProgressReporter *progress) {
               return CreateGraphWithRandomCost(images, kSelfCostFlif, options,
                                                progress);
             }),
             images, mst);
  } else {
    std::cerr << "Invalid cost '" << cost << "'\n";
    return 1;
  }

  return 0;
}
