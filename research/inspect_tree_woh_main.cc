// enc_without_header でエンコードしたファイルの決定木を集計する

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include <boost/iostreams/device/mapped_file.hpp>
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

namespace fs = std::filesystem;

using namespace jxl;

constexpr size_t kMaxPropertyCount =
    kNumNonrefProperties + 3 * kExtraPropsPerChannel;

int main(int argc, char* argv[]) {
  std::vector<fs::path> paths;

  {
    fs::path path_buf;
    for (int i = 1; i < argc; i++) {
      std::string_view arg = argv[i];
      if (arg == "-h" || arg == "--help") {
        std::cerr << "Usage: inspect_tree_woh FILE..." << std::endl;
        return 1;
      }

      path_buf = arg;
      if (fs::is_directory(path_buf)) {
        for (const fs::directory_entry& e :
             fs::recursive_directory_iterator(path_buf)) {
          if (e.is_regular_file()) paths.emplace_back(e);
        }
      } else {
        paths.push_back(path_buf);
      }
    }
  }

  std::atomic_size_t property_counts[kMaxPropertyCount];
  tbb::concurrent_vector<uint16_t> freqs;

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, paths.size(), 2),
      [&](const tbb::blocked_range<size_t> range) {
        boost::iostreams::mapped_file_source mf;
        Tree tree;

        for (size_t i = range.begin(); i != range.end(); ++i) {
          mf.open(paths[i]);
          Span<const uint8_t> span(mf);
          BitReader reader(span);

          const size_t tree_size_limit = std::numeric_limits<int32_t>::max();
          if (!DecodeTree(&reader, &tree, tree_size_limit)) {
            JXL_ABORT("Failed to decode tree: %s", paths[i].c_str());
          }

          ANSCode code;
          std::vector<uint8_t> context_map;
          if (!DecodeHistograms(&reader, (tree.size() + 1) / 2, &code,
                                &context_map)) {
            JXL_ABORT("Failed to decode histograms: %s", paths[i].c_str());
          }

          if (code.use_prefix_code) {
            JXL_ABORT("Prefix code not supported: %s", paths[i].c_str());
          }

          JXL_CHECK(reader.Close());
          mf.close();

          // プロパティの出現回数
          std::set<uint8_t> used_context;
          for (const PropertyDecisionNode& node : tree) {
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
            const AliasTable::Entry* table =
                &reinterpret_cast<AliasTable::Entry*>(
                    code.alias_tables.get())[ctx << code.log_alpha_size];
            for (size_t entry_idx = 0; entry_idx < (1u << code.log_alpha_size);
                 entry_idx++) {
              const AliasTable::Entry* entry = &table[entry_idx];
              uint16_t freq = entry->freq0 ^ entry->freq1_xor_freq0;
              if (freq == 0) {
                // エントリーなし
                continue;
              }
              freqs.push_back(freq);
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

  std::cout << header << "\n" << values << "\n\n";

  for (auto freq : freqs) {
    std::cout << freq << ",";
  }

  std::cout << std::endl;

  return 0;
}
