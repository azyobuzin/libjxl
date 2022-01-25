// enc_all でエンコードしたファイルの決定木を集計する

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>
#include <set>
#include <vector>

#include "common_cluster.h"
#include "dec_cluster.h"
#include "fmt/color.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_debug_tree.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace jxl;
using namespace research;

namespace {

IndexFields ReadIndex(const fs::path& input_dir) {
  boost::iostreams::mapped_file_source index_file(input_dir / "index.bin");
  Span<const uint8_t> index_span(index_file);
  BitReader reader(index_span);
  IndexFields index;
  JXL_CHECK(Bundle::Read(&reader, &index));
  JXL_CHECK(reader.Close());
  return index;
}

constexpr size_t kMaxPropertyCount =
    kNumNonrefProperties + 3 * kExtraPropsPerChannel;

}  // namespace

int main(int argc, char* argv[]) {
  po::options_description pos_ops;
  pos_ops.add_options()("input-dir", po::value<fs::path>()->required());

  po::positional_options_description pos_desc;
  pos_desc.add("input-dir", -1);

  po::options_description ops_desc;
  // clang-format off
  ops_desc.add_options()
    ("parent-ref", po::value<int>()->default_value(4), "0: 参照なし, 1: 親の同チャネル参照, 2: 親の全チャネル参照, 3: 前フレーム同チャネル参照, 4: 前フレーム全チャネル参照")
    ("flif", po::bool_switch(), "色チャネルをFLIFで符号化");
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
              << "Usage: inspect_tree [OPTIONS] INDEX_DIRECTORY" << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  const fs::path& input_dir = vm["input-dir"].as<fs::path>();
  const bool needsReferences = NeedsReferences(
      static_cast<ParentReferenceType>(vm["parent-ref"].as<int>()));
  const bool flif_enabled = vm["flif"].as<bool>();
  if (flif_enabled) JXL_ABORT("not implemented");
  // TODO(research): FLIF対応

  IndexFields index = ReadIndex(input_dir);
  std::set<uint32_t> used_cluster(index.assignments.cbegin(),
                                  index.assignments.cend());
  std::atomic_size_t property_counts[kMaxPropertyCount] = {0};

  tbb::parallel_for(
      tbb::blocked_range<uint32_t>(0, index.n_clusters),
      [&](tbb::blocked_range<uint32_t> cluster_range) {
        boost::iostreams::mapped_file_source mf;
        std::vector<const uint8_t*> offsets;

        // 値は使わないけれど、デコードに必要
        std::vector<uint32_t> pointers;
        std::vector<uint32_t> references;

        for (uint32_t cluster_idx = cluster_range.begin();
             cluster_idx != cluster_range.end(); ++cluster_idx) {
          if (used_cluster.find(cluster_idx) == used_cluster.end()) {
            // 1枚も画像のないクラスタにはクラスタファイルがない可能性がある
            continue;
          }

          mf.open(input_dir / fmt::format("cluster{}.bin", cluster_idx));
          Span<const uint8_t> cluster_span(mf);
          BitReader cluster_header_reader(cluster_span);
          ClusterHeader header(index.width, index.height, index.n_channel,
                               flif_enabled);
          if (!Bundle::Read(&cluster_header_reader, &header))
            JXL_ABORT("Failed to read header of cluster %" PRIu32, cluster_idx);

          uint32_t n_images = 0;
          for (const auto& x : header.combined_images) n_images += x.n_images;

          pointers.resize(n_images);
          DecodeClusterPointers(cluster_header_reader, pointers);

          if (needsReferences) {
            for (const auto& ci_info : header.combined_images) {
              references.resize(ci_info.n_images - 1);
              DecodeReferences(cluster_header_reader, references);
            }
          }

          if (!cluster_header_reader.JumpToByteBoundary())
            JXL_ABORT("Cluster %" PRIu32 " is broken (JumpToByteBoundary)",
                      cluster_idx);
          if (!cluster_header_reader.Close())
            JXL_ABORT("Cluster %" PRIu32 " is broken (Close)", cluster_idx);

          if (n_images > 0) {
            offsets.resize(header.combined_images.size());
            offsets[0] = cluster_header_reader.GetSpan().data();
            for (size_t ci_idx = 1; ci_idx < header.combined_images.size();
                 ci_idx++) {
              const auto& ci_info = header.combined_images[ci_idx - 1];
              offsets[ci_idx] =
                  offsets[ci_idx - 1] +
                  (ci_info.n_bytes + (flif_enabled ? ci_info.n_flif_bytes : 0));
            }

            tbb::parallel_for(
                size_t(0), header.combined_images.size(), [&](size_t ci_idx) {
                  Span<const uint8_t> ci_span(
                      offsets[ci_idx], header.combined_images[ci_idx].n_bytes);
                  BitReader ci_reader(ci_span);
                  Tree tree;
                  const size_t tree_size_limit =
                      std::numeric_limits<int32_t>::max();
                  if (!DecodeTree(&ci_reader, &tree, tree_size_limit))
                    JXL_ABORT("Failed to decode tree (cluster %" PRIu32
                              ", ci %" PRIuS ")",
                              cluster_idx, ci_idx);
                  JXL_CHECK(ci_reader.Close());

                  for (const PropertyDecisionNode& node : tree) {
                    if (node.property < 0) continue;  // leaf
                    if (node.property >=
                        static_cast<int16_t>(kMaxPropertyCount)) {
                      JXL_ABORT("Too large property index %" PRId16
                                " (cluster %" PRIu32 ", ci %" PRIuS ")",
                                node.property, cluster_idx, ci_idx);
                    }
                    property_counts[node.property].fetch_add(
                        1, std::memory_order_release);
                  }
                });
          }

          mf.close();
        }
      });

  std::string header;
  std::string values;
  for (size_t i = 0; i < kMaxPropertyCount; i++) {
    if (i > 0) {
      header += ',';
      values += ',';
    }

    header += PropertyName(i);
    values +=
        std::to_string(property_counts[i].load(std::memory_order_acquire));
  }

  std::cout << header << "\n" << values << std::endl;

  return 0;
}
