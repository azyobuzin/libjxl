#include <fmt/core.h>
#include <fmt/ostream.h>
#include <tbb/parallel_for.h>

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

#include "cost_graph_util.h"
#include "dec_cluster.h"
#include "enc_brute_force.h"
#include "lib/jxl/base/printf_macros.h"

namespace po = boost::program_options;

using namespace research;

namespace {

template <typename CreateGraphFunction>
std::vector<EncodedCombinedImage> EncodeImages(
    ImagesProvider &images, const jxl::ModularOptions &options,
    const EncodingOptions &encoding_options, bool use_brute_force,
    CreateGraphFunction create_graph) {
  decltype(ComputeMstFromGraph(
      create_graph(static_cast<ProgressReporter *>(nullptr)))) tree;
  {
    ConsoleProgressReporter progress("Computing MST");
    auto gr = create_graph(static_cast<ProgressReporter *>(&progress));
    tree = ComputeMstFromGraph(gr);
  }

  ConsoleProgressReporter progress("Encoding");
  return use_brute_force ? EncodeWithBruteForce<>(images, tree, options,
                                                  encoding_options, &progress)
                         : EncodeWithCombineAll<>(images, tree, options,
                                                  encoding_options, &progress);
}

std::tuple<uint32_t, uint32_t, uint32_t> GetImageInfo(ImagesProvider &images) {
  jxl::Image img = images.get(0);
  return std::make_tuple(
      static_cast<uint32_t>(img.w), static_cast<uint32_t>(img.h),
      static_cast<uint32_t>(img.channel.size() - img.nb_meta_channels));
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
    ("cost", po::value<std::string>()->default_value("tree"), "MSTに使用するコスト tree: JPEG XL決定木入れ替え, y: Yチャネル, props: JPEG XLプロパティ")
    ("split", po::value<uint16_t>()->default_value(2), "画像を何回分割するか（cost = props-* のみ）")
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    ("refchan", po::value<uint16_t>()->default_value(0), "画像内のチャンネル参照数")
    ("max-refs", po::value<size_t>()->default_value(1), "画像の参照数")
    ("flif", po::bool_switch(), "色チャネルをFLIFで符号化")
    ("flif-learn", po::value<int>()->default_value(2), "FLIF学習回数")
    ("enc-method", po::value<std::string>()->default_value("brute-force"), "brute-force or combine-all")
    ("out", po::value<std::string>(), "圧縮結果の出力先ファイルパス")
    ("verify", po::bool_switch(), "エンコード結果をデコードして、一致するかを確認する");
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
              << "Usage: enc_cluster [OPTIONS] IMAGE_FILE..." << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  // MST生成まで cost_graph_enc_main.cc と同じ
  const std::vector<std::string> &paths =
      vm["image-file"].as<std::vector<std::string>>();
  FileImagesProvider images(paths);
  images.ycocg = true;

  const std::string &cost = vm["cost"].as<std::string>();
  const size_t split = vm["split"].as<uint16_t>();
  const float fraction = vm["fraction"].as<float>();
  int refchan = vm["refchan"].as<uint16_t>();
  size_t max_refs = vm["max-refs"].as<size_t>();
  bool flif_enabled = vm["flif"].as<bool>();
  int flif_learn_repeats = vm["flif-learn"].as<int>();

  const std::string &enc_method = vm["enc-method"].as<std::string>();
  bool use_brute_force = false;
  if (enc_method == "brute-force") {
    use_brute_force = true;
  } else if (enc_method != "combine-all") {
    JXL_ABORT("Invalid enc-method '%s'", enc_method.c_str());
  }

  // Tortoise相当
  jxl::ModularOptions options{
      .nb_repeats = fraction,
      .max_properties = refchan,
      .splitting_heuristics_properties = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3,
                                          4, 5, 6, 7, 8},
      .max_property_values = 256,
      .predictor = jxl::Predictor::Variable};

  EncodingOptions encoding_options{max_refs, flif_enabled, flif_learn_repeats};

  std::vector<EncodedCombinedImage> results;

  if (cost == "tree") {
    results = EncodeImages(images, options, encoding_options, use_brute_force,
                           [&](ProgressReporter *progress) {
                             return CreateGraphWithDifferentTree(
                                 images, options, progress);
                           });
  } else if (cost == "y") {
    results = EncodeImages(images, options, encoding_options, use_brute_force,
                           [&](ProgressReporter *progress) {
                             return CreateGraphWithYDistance(
                                 images, kSelfCostJxl, options, progress);
                           });
  } else if (cost == "props") {
    results = EncodeImages(images, options, encoding_options, use_brute_force,
                           [&](ProgressReporter *progress) {
                             return CreateGraphWithPropsDistance(
                                 images, kSelfCostJxl, split, fraction, options,
                                 progress);
                           });
  } else {
    JXL_ABORT("Invalid cost '%s'", cost.c_str());
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

    std::cout << std::endl
              << "bytes: " << x.n_bytes() << std::endl
              << std::endl;
  }

  if (!vm["out"].empty()) {
    const auto &out_path = vm["out"].as<std::string>();
    std::ofstream dst(out_path, std::ios_base::out | std::ios_base::binary);
    if (!dst) {
      std::cerr << "Failed to open " << out_path.c_str() << std::endl;
      return 1;
    }

    PackToClusterFile(results, dst);
    dst.flush();

    if (!dst) {
      std::cerr << "Failed to write " << out_path.c_str() << std::endl;
      return 1;
    }
  }

  if (vm["verify"].as<bool>()) {
    ConsoleProgressReporter progress("Verifying");

    std::vector<jxl::Image> decoded_images;
    {
      auto [width, height, n_channel] = GetImageInfo(images);

      // ClusterFile形式に変換
      std::ostringstream oss(std::ios_base::out | std::ios_base::binary);
      PackToClusterFile(results, oss);

      if (!oss) {
        std::cerr << "Failed to write to buffer" << std::endl;
        return 1;
      }

      // ClusterFile形式から読み出す
      std::string buf = oss.str();
      jxl::Span<const uint8_t> span(buf);
      DecodingOptions options{
          width,   height,       {n_channel, max_refs},
          refchan, flif_enabled, /*flif_additional_props=*/0};
      ClusterFileReader reader(options, span);
      if (!reader.ReadAll(decoded_images)) {
        std::cerr << "Failed to decode images" << std::endl;
        return 1;
      }
    }

    if (decoded_images.size() != images.size()) {
      fmt::print(std::cerr, "decoded_images.size ({}) != images.size ({})\n",
                 decoded_images.size(), images.size());
      return 1;
    }

    images.ycocg = false;

    std::atomic_size_t n_completed = 0;
    std::atomic_bool mismatch = false;

    tbb::parallel_for(size_t(0), decoded_images.size(), [&](size_t i) {
      auto &decoded_image = decoded_images[i];
      const auto expected_image = images.get(i);

      if (decoded_image.channel.size() - decoded_image.nb_meta_channels == 3) {
        // YCoCg → RGB
        auto &ycocg_transform =
            decoded_image.transform.emplace_back(jxl::TransformId::kRCT);
        ycocg_transform.rct_type = 6;
        ycocg_transform.begin_c = decoded_image.nb_meta_channels;
        decoded_image.undo_transforms({}, nullptr);
      }

      if (decoded_image.channel.size() != expected_image.channel.size()) {
        fmt::print(std::cerr,
                   "{} ({}): channel mismatch (actual: {}, expected: {})\n",
                   images.get_label(i), i, decoded_image.channel.size(),
                   expected_image.channel.size());
        mismatch = true;
      } else {
        for (size_t chan = 0; chan < decoded_image.channel.size(); chan++) {
          const auto &decoded_chan = decoded_image.channel[chan];
          const auto &expected_chan = expected_image.channel[chan];

          if ((decoded_chan.w != expected_chan.w) ||
              (decoded_chan.h != expected_chan.h) ||
              (decoded_chan.hshift != expected_chan.hshift) ||
              (decoded_chan.vshift != expected_chan.vshift)) {
            fmt::print(std::cerr,
                       "{} ({}): size mismatch at channel {} (actual: {}<<{} x "
                       "{}<<{}, expected: {}<<{} x {}<<{})\n",
                       images.get_label(i), i, chan, decoded_chan.w,
                       decoded_chan.hshift, decoded_chan.h, decoded_chan.vshift,
                       expected_chan.w, expected_chan.hshift, expected_chan.h,
                       expected_chan.vshift);
            mismatch = true;
            continue;
          }

          for (size_t y = 0; y < decoded_chan.h; y++) {
            const jxl::pixel_type *decoded_row = decoded_chan.Row(y);
            const jxl::pixel_type *expected_row = expected_chan.Row(y);
            for (size_t x = 0; x < decoded_chan.w; x++) {
              if (decoded_row[x] != expected_row[x]) {
                fmt::print(std::cerr,
                           "{} ({}): pixel ({}, {}) mismatch at channel {} "
                           "(actual: {}, expected: {})\n",
                           images.get_label(i), i, x, y, chan, decoded_row[x],
                           expected_row[x]);
                mismatch = true;
              }
            }
          }
        }
      }

      progress.report(++n_completed, decoded_images.size());
    });

    if (mismatch) return 1;
  }

  return 0;
}
