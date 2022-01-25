// ヘッダーなしのJPEG XLに変換する ベースライン検証用

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <tbb/parallel_for.h>

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>

#include "common_cluster.h"
#include "enc_cluster.h"
#include "images_provider.h"
#include "jxl_parallel.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/modular/transform/enc_transform.h"
#include "lib/jxl/modular/transform/transform.h"
#include "progress.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace research;

namespace jxl {

bool do_transform(Image& image, const Transform& tr,
                  const weighted::Header& wp_header,
                  jxl::ThreadPool* pool = nullptr);

float EstimateCost(const Image& img);

}  // namespace jxl

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
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    ("y-only", po::bool_switch(), "Yチャネルのみを利用する")
    ("refchan", po::value<uint16_t>()->default_value(0), "画像内のチャンネル参照数")
    ("palette", po::bool_switch(), "パレット変換を有効化する")
    ("rct", po::bool_switch(), "色変換をすべて試す")
    ("speed", po::value<uint16_t>()->default_value(1), "1: tortoise, 2: kitten, 3: squirrel")
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
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl
              << std::endl
              << "Usage: enc_without_header [OPTIONS] IMAGE_FILE..."
              << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  const std::vector<std::string>& paths =
      vm["image-file"].as<std::vector<std::string>>();
  const bool try_all_rct = vm["rct"].as<bool>();
  FileImagesProvider images(paths);
  images.only_first_channel = vm["y-only"].as<bool>();
  images.ycocg = images.only_first_channel || !try_all_rct;

  // Tortoise相当
  jxl::ModularOptions options{
      .nb_repeats = vm["fraction"].as<float>(),
      .max_properties = vm["refchan"].as<uint16_t>(),
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

  const auto& out_dir = vm["out-dir"].as<fs::path>();
  fs::create_directories(out_dir);

  const bool use_palette = vm["palette"].as<bool>();
  std::atomic_bool failed = false;
  std::atomic_size_t n_completed = 0;
  ConsoleProgressReporter progress("Encoding");

  tbb::parallel_for(size_t(0), images.size(), [&](size_t i) {
    // エンコード
    auto image_ptr = std::make_shared<jxl::Image>(images.get(i));
    jxl::Image& image = *image_ptr;

    jxl::BitWriter writer;
    jxl::ThreadPool pool(TbbParallelRunner, nullptr);

    if (use_palette) {
      jxl::CompressParams cparams;
      cparams.SetLossless();

      // Global palette
      jxl::Transform global_palette(jxl::TransformId::kPalette);
      global_palette.begin_c = image.nb_meta_channels;
      global_palette.num_c = image.channel.size() - image.nb_meta_channels;
      global_palette.nb_colors = std::min((int)(image.w * image.h / 8),
                                          std::abs(cparams.palette_colors));
      global_palette.ordered_palette = cparams.palette_colors >= 0;
      global_palette.lossy_palette = false;
      if (jxl::TransformForward(global_palette, image, {}, &pool)) {
        image.transform.push_back(std::move(global_palette));
        std::cerr << images.get_label(i) << " use global palette" << std::endl;
      }

      // Local channel palette
      JXL_ASSERT(cparams.channel_colors_percent > 0);
      for (size_t i = image.nb_meta_channels; i < image.channel.size(); i++) {
        size_t real_chan = i - image.nb_meta_channels;
        int min, max;
        jxl::compute_minmax(image.channel[i], &min, &max);
        int colors = max - min + 1;
        jxl::Transform local_palette(jxl::TransformId::kPalette);
        local_palette.begin_c = i;
        local_palette.num_c = 1;
        local_palette.nb_colors =
            std::min((int)(image.w * image.h * 0.8),
                     (int)(cparams.channel_colors_percent / 100. * colors));
        if (jxl::do_transform(image, local_palette, {}, &pool)) {
          image.transform.push_back(std::move(local_palette));
          if (JXL_DEBUG_V_LEVEL >= 2) {
            std::cerr << images.get_label(i) << " use local palette (channel "
                      << real_chan << ")" << std::endl;
          }
        }
      }
    }

    if (!images.ycocg) {
      // すべてのRCTを試す
      // https://github.com/libjxl/libjxl/blob/3d077b281fa65eab595447ae38ba9efc385ba03e/lib/jxl/enc_modular.cc#L1303-L1361
      jxl::Transform sg(jxl::TransformId::kRCT);
      sg.begin_c = image.nb_meta_channels;
      float best_cost = std::numeric_limits<float>::max();
      size_t best_rct = 0;
      for (int i : {0 * 7 + 0, 0 * 7 + 6, 0 * 7 + 5, 1 * 7 + 3, 3 * 7 + 5,
                    5 * 7 + 5, 1 * 7 + 5, 2 * 7 + 5, 1 * 7 + 1, 0 * 7 + 4,
                    1 * 7 + 2, 2 * 7 + 1, 2 * 7 + 2, 2 * 7 + 3, 4 * 7 + 4,
                    4 * 7 + 5, 0 * 7 + 2, 0 * 7 + 1, 0 * 7 + 3}) {
        sg.rct_type = i;
        if (jxl::do_transform(image, sg, {}, &pool)) {
          float cost = jxl::EstimateCost(image);
          if (cost < best_cost) {
            best_rct = i;
            best_cost = cost;
          }
          jxl::Transform t = image.transform.back();
          JXL_CHECK(t.Inverse(image, {}, &pool));
          image.transform.pop_back();
        }
      }

      sg.rct_type = best_rct;
      if (jxl::do_transform(image, sg, {}, &pool)) {
        if (JXL_DEBUG_V_LEVEL >= 2) {
          if (best_rct % 7 == 6) {
            fmt::print(std::cerr, "{} use YCoCg, permutation {}\n",
                       images.get_label(i), best_rct / 7);
          } else {
            fmt::print(std::cerr, "{} use RCT {}, {}, {}\n",
                       images.get_label(i), best_rct / 7, (best_rct % 7) >> 1,
                       (best_rct % 7) & 1);
          }
        }
      }
    }

    jxl::ModularOptions local_options = options;
    local_options.wp_mode = FindBestWPMode(image);

    CombinedImage ci = CombineImage(std::move(image_ptr));
    jxl::Tree tree =
        LearnTree(writer, ci, local_options, jxl::kParentReferenceNone);
    EncodeImages(writer, ci, local_options, jxl::kParentReferenceNone, tree);
    writer.ZeroPadToByte();

    // ファイルに出力
    auto span = writer.GetSpan();
    fs::path p = out_dir / fmt::format("{}.bin", i);
    FILE* fp = fopen(p.c_str(), "wb");
    if (fp) {
      if (fwrite(span.data(), 1, span.size(), fp) != span.size()) {
        std::cerr << "Failed to write " << p.c_str() << std::endl;
        failed = true;
      }
      fclose(fp);
    } else {
      std::cerr << "Failed to open " << p.c_str() << std::endl;
      failed = true;
    }

    progress.report(++n_completed, images.size());
  });

  return failed ? 1 : 0;
}
