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
  ops_desc.add_options()("split", po::value<uint16_t>()->default_value(2),
                         "画像を何回分割するか");
  ops_desc.add_options()("fraction", po::value<float>()->default_value(.5f),
                         "サンプリングする画素の割合 (0, 1]");

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

  const std::vector<std::string> &paths =
      vm["image-file"].as<std::vector<std::string>>();
  FileImagesProvider images(paths);
  jxl::ModularOptions options{.nb_repeats = vm["fraction"].as<float>()};
  // gradient, W-NW, NW-N, N-NE, N-NN (from splitting_heuristics_properties)
  std::vector<uint32_t> props_to_use = {9, 10, 11, 12, 13};
  jxl::TreeSamples tree_samples;

  SamplesForQuantization samples_for_quantization =
      CollectSamplesForQuantization(images, options);
  InitializeTreeSamples(tree_samples, props_to_use, options.max_properties,
                        samples_for_quantization);

  images.reset();

  std::vector<std::string> descriptions;

  for (const auto &path : paths) {
    std::cout << path << std::endl;
    auto img = images.next().value();

    descriptions.clear();
    auto result = ExtractPropertiesFromImage(
        img, vm["split"].as<uint16_t>(), options, tree_samples, &descriptions);

    for (size_t i = 0; i < result.size(); i++) {
      std::cout << descriptions[i] << "\t" << result[i] << std::endl;
    }

    std::cout << std::endl;
  }

  return 0;
}
