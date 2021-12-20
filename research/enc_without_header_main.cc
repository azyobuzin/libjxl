// ヘッダーなしのJPEG XLに変換する ベースライン検証用

#include <fmt/core.h>
#include <tbb/parallel_for.h>

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>

#include "enc_jxl_multi.h"
#include "images_provider.h"
#include "progress.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace research;

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
    ("out-dir", po::value<fs::path>()->required(), "圧縮結果の出力先");
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

  const auto &out_dir = vm["out-dir"].as<fs::path>();
  fs::create_directories(out_dir);

  std::atomic_bool failed = false;
  std::atomic_size_t n_completed = 0;
  ConsoleProgressReporter progress("Encoding");

  tbb::parallel_for(size_t(0), images.size(), [&](size_t i) {
    // エンコード
    CombinedImage image = CombineImage(images.get(i));
    jxl::BitWriter writer;
    jxl::Tree tree = LearnTree(writer, image, options, 0);
    EncodeImages(writer, image, options, 0, tree);
    writer.ZeroPadToByte();

    // ファイルに出力
    auto span = writer.GetSpan();
    fs::path p = out_dir / fmt::format("{}.bin", i);
    FILE *fp = fopen(p.c_str(), "wb");
    if (!fp) {
      std::cerr << "Failed to open " << p.string() << std::endl;
      failed = true;
      return;
    }
    fwrite(span.data(), 1, span.size(), fp);
    fclose(fp);

    progress.report(++n_completed, images.size());
  });

  return failed ? 1 : 0;
}
