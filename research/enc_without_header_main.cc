// ヘッダーなしのJPEG XLに変換する ベースライン検証用

#include <fmt/core.h>
#include <tbb/parallel_for.h>

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>

#include "enc_jxl_multi.h"
#include "fields.h"
#include "images_provider.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/modular/transform/enc_transform.h"
#include "lib/jxl/modular/transform/transform.h"
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
    ("refchan", po::value<uint16_t>()->default_value(0), "画像内のチャンネル参照数")
    ("palette", po::bool_switch(), "パレット変換を有効化する")
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
      .max_properties = vm["refchan"].as<uint16_t>(),
      .splitting_heuristics_properties = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3,
                                          4, 5, 6, 7, 8},
      .max_property_values = 256,
      .predictor = jxl::Predictor::Variable};

  const auto &out_dir = vm["out-dir"].as<fs::path>();
  fs::create_directories(out_dir);

  const bool use_palette = vm["palette"].as<bool>();
  std::atomic_bool failed = false;
  std::atomic_size_t n_completed = 0;
  ConsoleProgressReporter progress("Encoding");

  tbb::parallel_for(size_t(0), images.size(), [&](size_t i) {
    // エンコード
    CombinedImage image = CombineImage(images.get(i));
    jxl::BitWriter writer;

    if (use_palette) {
      CombinedImageHeader header;
      jxl::CompressParams cparams;
      cparams.SetLossless();

      // Global palette
      jxl::Transform global_palette(jxl::TransformId::kPalette);
      global_palette.begin_c = image.image.nb_meta_channels;
      global_palette.num_c =
          image.image.channel.size() - image.image.nb_meta_channels;
      global_palette.nb_colors =
          std::min((int)(image.image.w * image.image.h / 8),
                   std::abs(cparams.palette_colors));
      global_palette.ordered_palette = cparams.palette_colors >= 0;
      global_palette.lossy_palette = false;
      if (jxl::TransformForward(global_palette, image.image,
                                jxl::weighted::Header(), nullptr)) {
        header.transforms.push_back(std::move(global_palette));
        std::cerr << images.get_label(i) << " use global palette" << std::endl;
      }

      // Local channel palette
      JXL_ASSERT(cparams.channel_colors_percent > 0);
      for (size_t i = image.image.nb_meta_channels;
           i < image.image.channel.size(); i++) {
        int min, max;
        jxl::compute_minmax(image.image.channel[i], &min, &max);
        int colors = max - min + 1;
        jxl::Transform local_palette(jxl::TransformId::kPalette);
        local_palette.begin_c = i;
        local_palette.num_c = 1;
        local_palette.nb_colors =
            std::min((int)(image.image.w * image.image.h * 0.8),
                     (int)(cparams.channel_colors_percent / 100. * colors));
        if (jxl::TransformForward(local_palette, image.image,
                                  jxl::weighted::Header(), nullptr)) {
          header.transforms.push_back(std::move(local_palette));
          std::cerr << images.get_label(i) << " use local palette (channel "
                    << (i - image.image.nb_meta_channels) << ")" << std::endl;
        }
      }

      JXL_CHECK(jxl::Bundle::Write(header, &writer, 0, nullptr));
    }

    jxl::Tree tree = LearnTree(writer, image, options, 0);
    EncodeImages(writer, image, options, 0, tree);
    writer.ZeroPadToByte();

    // ファイルに出力
    auto span = writer.GetSpan();
    fs::path p = out_dir / fmt::format("{}.bin", i);
    FILE *fp = fopen(p.c_str(), "wb");
    if (fp) {
      if (fwrite(span.data(), 1, span.size(), fp) != span.size()) {
        std::cerr << "Failed to write " << p.string() << std::endl;
        failed = true;
      }
      fclose(fp);
    } else {
      std::cerr << "Failed to open " << p.string() << std::endl;
      failed = true;
    }

    progress.report(++n_completed, images.size());
  });

  return failed ? 1 : 0;
}
