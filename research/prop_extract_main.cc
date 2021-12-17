#include <boost/program_options.hpp>
#include <iostream>

#include "prop_extract.h"

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
    ("split", po::value<uint16_t>()->default_value(2), "画像を何回分割するか")
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    ("y-only", po::bool_switch(), "Yチャネルのみを利用する")
    ("csv", po::bool_switch(), "結果をCSV形式で出力する");
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
              << "Usage: prop_extract [OPTIONS] IMAGE_FILE..." << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  size_t split = vm["split"].as<uint16_t>();
  float fraction = vm["fraction"].as<float>();
  bool csv = vm["csv"].as<bool>();

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

  std::vector<std::string> descriptions;

  // 各画像のプロパティから特徴量を求めて、出力する
  if (csv) {
    bool is_first = true;

    for (size_t i = 0; i < paths.size(); i++) {
      auto img = images.get(i);
      auto result =
          ExtractPropertiesFromImage(img, split, options, tree_samples,
                                     is_first ? &descriptions : nullptr);

      if (is_first) {
        std::cout << "path";
        for (const auto &desc : descriptions) std::cout << "," << desc;
        std::cout << std::endl;
        is_first = false;
      }

      std::cout << "\"" << paths[i] << "\"";
      for (const auto &x : result) std::cout << "," << x;
      std::cout << std::endl;
    }
  } else {
    for (size_t i = 0; i < paths.size(); i++) {
      std::cout << paths[i] << std::endl;
      auto img = images.get(i);

      descriptions.clear();
      auto result = ExtractPropertiesFromImage(img, split, options,
                                               tree_samples, &descriptions);

      for (size_t j = 0; j < result.size(); j++) {
        std::cout << descriptions[j] << "\t" << result[j] << std::endl;
      }

      std::cout << std::endl;
    }
  }

  return 0;
}
