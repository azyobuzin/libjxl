#include <boost/graph/graphviz.hpp>
#include <boost/program_options.hpp>
#include <iostream>

#include "cost_graph_util.h"

namespace po = boost::program_options;

using namespace research;

namespace {

BidirectionalCostGraph<int64_t> CreateGraph(
    ImagesProvider &images, const jxl::ModularOptions &options) {
  ConsoleProgressReporter progress("Working");
  return CreateGraphWithDifferentTree(images, options, &progress).graph;
}

std::shared_ptr<ImageTree<int64_t>> CreateTree(
    ImagesProvider &images, const jxl::ModularOptions &options) {
  ConsoleProgressReporter progress("Working");
  return CreateMstWithDifferentTree(images, options, &progress);
}

struct ImageVertexLabelWriter {
  ImagesProvider &images;

  void operator()(std::ostream &out, size_t vertex_idx) const {
    out << "[label=" << boost::escape_dot_string(images.get_label(vertex_idx))
        << "]";
  }
};

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
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    ("y-only", po::bool_switch(), "Yチャネルのみを利用する")
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
              << "Usage: cost_graph_enc [OPTIONS] IMAGE_FILE..." << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  const std::vector<std::string> &paths =
      vm["image-file"].as<std::vector<std::string>>();
  FileImagesProvider images(paths);
  images.ycocg = true;
  images.only_first_channel = vm["y-only"].as<bool>();

  // Tortoise相当
  jxl::ModularOptions options{
      .nb_repeats = vm["fraction"].as<float>(),
      .splitting_heuristics_properties = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3,
                                          4, 5, 6, 7, 8},
      .max_property_values = 256,
      .predictor = jxl::Predictor::Variable};

  if (vm["mst"].as<bool>()) {
    auto tree = CreateTree(images, options);
    PrintImageTreeDot(std::cout, tree, &images);
  } else {
    auto graph = CreateGraph(images, options);
    boost::write_graphviz(
        std::cout, graph, ImageVertexLabelWriter{images},
        boost::make_label_writer(get(boost::edge_weight_t(), graph)));
  }

  return 0;
}
