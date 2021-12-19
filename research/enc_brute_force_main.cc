#include <fmt/core.h>

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>

#include "cost_graph_util.h"
#include "enc_brute_force.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace research;

namespace {

std::shared_ptr<ImageTree<size_t>> CreateTree(
    ImagesProvider &images, const jxl::ModularOptions &options) {
  ConsoleProgressReporter progress("Computing MST");
  return CreateMstWithDifferentTree(images, options, &progress);
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
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    ("y-only", po::bool_switch(), "Yチャネルのみを利用する")
    ("max-refs", po::value<size_t>()->default_value(1), "画像の参照数")
    ("out-dir", po::value<fs::path>(), "圧縮結果の出力先");
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
              << "Usage: enc_brute_force [OPTIONS] IMAGE_FILE..." << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  // MST生成まで cost_graph_enc_main.cc と同じ
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

  auto tree = CreateTree(images, options);

  std::vector<EncodedImages> results;
  {
    ConsoleProgressReporter progress("Encoding");
    results = EncodeWithBruteForce(images, tree, options,
                                   vm["max-refs"].as<size_t>(), &progress);
  }

  for (const auto &x : results) {
    std::cout << "images: ";
    bool is_first = true;
    for (size_t image_idx : x.image_indices) {
      if (is_first)
        is_first = false;
      else
        std::cout << ", ";

      std::cout << images.get_label(image_idx);
    }

    std::cout << std::endl << "bits: " << x.n_bits << std::endl << std::endl;
  }

  if (!vm["out-dir"].empty()) {
    const auto &out_dir = vm["out-dir"].as<fs::path>();
    for (size_t i = 0; i < results.size(); i++) {
      const auto &data = results[i].data;
      fs::path p = out_dir / fmt::format("{}.bin", i);
      FILE *fp = fopen(p.c_str(), "wb");
      if (!fp) {
        std::cerr << "Failed to open " << p.string() << std::endl;
        return 1;
      }
      fwrite(data.data(), 1, data.size(), fp);
      fclose(fp);
    }
  }

  return 0;
}
