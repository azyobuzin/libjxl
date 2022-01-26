// enc_without_header でエンコードしたファイルの決定木を集計する

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>
#include <set>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_debug_tree.h"
#include "lib/jxl/modular/encoding/encoding.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace jxl;

namespace research {

// modular/encoding/encoding.cc で定義
Status ModularDecodeMulti(BitReader *br, Image &image, size_t group_id,
                          ModularOptions *options, const Tree *global_tree,
                          const ANSCode *global_code,
                          const std::vector<uint8_t> *global_ctx_map,
                          const DecodingRect *rect,
                          const MultiOptions &multi_options,
                          std::vector<size_t> *context_freqs);

}  // namespace research

constexpr size_t kMaxPropertyCount =
    kNumNonrefProperties + 3 * kExtraPropsPerChannel;

int main(int argc, char *argv[]) {
  po::options_description pos_ops;
  pos_ops.add_options()(
      "input-files",
      po::value<std::vector<fs::path>>()->multitoken()->required());

  po::positional_options_description pos_desc;
  pos_desc.add("input-files", -1);

  po::options_description ops_desc;
  // clang-format off
  ops_desc.add_options()
    ("weight-freq", po::bool_switch(), "シンボル出現頻度にコンテキスト利用回数で重み付けする")
    ("width", po::value<size_t>())
    ("height", po::value<size_t>());
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
              << "Usage: inspect_tree_woh [OPTIONS] FILE..." << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  std::vector<fs::path> paths;

  for (const auto &arg : vm["input-files"].as<std::vector<fs::path>>()) {
    if (fs::is_directory(arg)) {
      for (const fs::directory_entry &e :
           fs::recursive_directory_iterator(arg)) {
        if (e.is_regular_file()) paths.emplace_back(e);
      }
    } else {
      paths.push_back(arg);
    }
  }

  const bool weight_freq = vm["weight-freq"].as<bool>();

  if (weight_freq && (vm["width"].empty() || vm["height"].empty())) {
    JXL_ABORT("width and height are required if weight-freq");
  }

  std::atomic_size_t tree_bits = 0;
  std::atomic_size_t histo_bits = 0;
  std::atomic_size_t property_counts[kMaxPropertyCount] = {0};
  tbb::concurrent_vector<uint16_t> freqs;

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, paths.size(), 2),
      [&](const tbb::blocked_range<size_t> range) {
        boost::iostreams::mapped_file_source mf;
        Tree tree;
        std::vector<size_t> context_freqs;

        for (size_t i = range.begin(); i != range.end(); ++i) {
          mf.open(paths[i]);
          Span<const uint8_t> span(mf);
          BitReader reader(span);

          const size_t tree_size_limit = std::numeric_limits<int32_t>::max();
          if (!DecodeTree(&reader, &tree, tree_size_limit)) {
            JXL_ABORT("Failed to decode tree: %s", paths[i].c_str());
          }
          size_t bit_pos = reader.TotalBitsConsumed();
          tree_bits.fetch_add(bit_pos, std::memory_order_release);

          ANSCode code;
          std::vector<uint8_t> context_map;
          if (!DecodeHistograms(&reader, (tree.size() + 1) / 2, &code,
                                &context_map)) {
            JXL_ABORT("Failed to decode histograms: %s", paths[i].c_str());
          }
          histo_bits.fetch_add(reader.TotalBitsConsumed() - bit_pos,
                               std::memory_order_release);

          if (code.use_prefix_code) {
            JXL_ABORT("Prefix code not supported: %s", paths[i].c_str());
          }

          if (weight_freq) {
            Image image(vm["width"].as<size_t>(), vm["height"].as<size_t>(), 8,
                        3);
            ModularOptions options;
            DecodingRect dr = {"inspect_tree_woh", 0, 0, 0};
            if (!research::ModularDecodeMulti(&reader, image, 0, &options,
                                              &tree, &code, &context_map, &dr,
                                              {}, &context_freqs)) {
              JXL_ABORT("Failed to decode: %s", paths[i].c_str());
            }
            JXL_CHECK(reader.JumpToByteBoundary());
          }

          JXL_CHECK(reader.Close());
          mf.close();

          // プロパティの出現回数
          std::set<uint8_t> used_context;
          for (const PropertyDecisionNode &node : tree) {
            if (node.property >= static_cast<int16_t>(kMaxPropertyCount)) {
              JXL_ABORT("Too large property index %" PRId16 " found in %s",
                        node.property, paths[i].c_str());
            } else if (node.property >= 0) {
              property_counts[node.property].fetch_add(
                  1, std::memory_order_release);
            } else {
              // leaf
              used_context.insert(context_map.at(node.lchild));
            }
          }

          // シンボルの出現回数
          for (uint8_t ctx : used_context) {
            const AliasTable::Entry *table =
                &reinterpret_cast<AliasTable::Entry *>(
                    code.alias_tables.get())[ctx << code.log_alpha_size];
            for (size_t entry_idx = 0; entry_idx < (1u << code.log_alpha_size);
                 entry_idx++) {
              const AliasTable::Entry *entry = &table[entry_idx];
              uint16_t freq = entry->freq0 ^ entry->freq1_xor_freq0;
              if (freq == 0) {
                // エントリーなし
                continue;
              }
              if (weight_freq) {
                if (ctx < context_freqs.size()) {
                  for (size_t i = 0; i < context_freqs[ctx]; i++)
                    freqs.push_back(freq);
                }
              } else {
                freqs.push_back(freq);
              }
              if (freq == ANS_TAB_SIZE) {
                // 唯一のエントリー
                break;
              }
            }
          }
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
    values += std::to_string(property_counts[i]);
  }

  std::cout << header << "\n"
            << values
            << "\ntree bits: " << tree_bits.load(std::memory_order_acquire)
            << "\nhisto bits: " << histo_bits.load(std::memory_order_acquire)
            << "\n\n";

  for (auto freq : freqs) {
    std::cout << freq << ",";
  }

  std::cout << std::endl;

  return 0;
}
