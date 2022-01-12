#include <fmt/core.h>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mlpack/core.hpp>
#include <random>

#include "common_cluster.h"
#include "dec_cluster.h"
#include "enc_all.h"
#include "enc_brute_force.h"
#include "lib/jxl/base/compiler_specific.h"
#include "prop_extract.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace std::chrono;
using namespace research;

namespace {

template <class RandomEngine>
jxl::Image DecodeOneImage(const fs::path& input_dir,
                          jxl::ParentReferenceType parent_reference,
                          RandomEngine& engine) {
  IndexFields index;
  {
    boost::iostreams::mapped_file_source index_file(input_dir / "index.bin");
    jxl::Span<const uint8_t> index_span(index_file);
    jxl::BitReader reader(index_span);
    JXL_CHECK(jxl::Bundle::Read(&reader, &index));
    JXL_CHECK(reader.Close());
  }

  size_t img_idx = std::uniform_int_distribution<size_t>(
      0, index.assignments.size() - 1)(engine);
  JXL_ASSERT(img_idx < index.assignments.size());
  uint32_t cluster_idx = index.assignments[img_idx];

  // クラスタ内でのインデックスを求める
  uint32_t idx_in_cluster = 0;
  for (uint32_t i = 0; i < img_idx; i++) {
    if (index.assignments[i] == cluster_idx) idx_in_cluster++;
  }

  DecodingOptions options{index.width,      index.height, index.n_channel,
                          parent_reference, false,        0};

  boost::iostreams::mapped_file_source cluster_file(
      input_dir / fmt::format("cluster{}.bin", cluster_idx));
  jxl::Span<const uint8_t> cluster_span(cluster_file);
  ClusterFileReader cluster_reader(options, cluster_span);
  jxl::Image result;
  JXL_CHECK(cluster_reader.Read(idx_in_cluster, result));
  return result;
}

}  // namespace

int main(int argc, char* argv[]) {
  po::options_description pos_ops;
  pos_ops.add_options()("input-dir", po::value<fs::path>()->required());

  po::positional_options_description pos_desc;
  pos_desc.add("input-dir", -1);

  po::options_description ops_desc;
  // clang-format off
  ops_desc.add_options()
    ("parent-ref", po::value<int>()->default_value(2), "0: 参照なし, 1: 親の同チャネル参照, 2: 親の全チャネル参照")
    ("flif", po::bool_switch(), "色チャネルをFLIFで符号化")
    ("iter", po::value<uint32_t>()->default_value(1000), "デコードする画像数");
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
              << "Usage: benchmark_random_decode [OPTIONS] INDEX_DIRECTORY..."
              << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  const fs::path& input_dir = vm["input-dir"].as<fs::path>();
  const jxl::ParentReferenceType parent_ref =
      static_cast<jxl::ParentReferenceType>(vm["parent-ref"].as<int>());
  const bool flif_enabled = vm["flif"].as<bool>();
  if (flif_enabled) JXL_ABORT("not implemented");
  // TODO(research): FLIF対応

  uint32_t iter = vm["iter"].as<uint32_t>();
  arma::vec durations(iter);

  std::random_device seed_gen;
  std::mt19937 rng(seed_gen());

  for (uint32_t i = 0; i < iter; i++) {
    auto start = steady_clock::now();
    DecodeOneImage(input_dir, parent_ref, rng);
    auto end = steady_clock::now();
    durations[i] = duration<double, std::milli>(end - start).count();
  }

  std::cout << "mean: " << arma::mean(durations) << " ms\n"
            << "stddev: " << arma::stddev(durations) << " ms\n"
            << "total: " << arma::sum(durations) << " ms\n";

  return 0;
}
