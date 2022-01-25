// enc_without_header でエンコードしたファイルの決定木を集計する

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <boost/iostreams/device/mapped_file.hpp>
#include <filesystem>
#include <iostream>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_debug_tree.h"

namespace fs = std::filesystem;

using namespace jxl;

namespace {

constexpr size_t kMaxPropertyCount =
    kNumNonrefProperties + 3 * kExtraPropsPerChannel;

struct AggregateTreeBody {
  const std::vector<fs::path>& paths;
  size_t property_counts[kMaxPropertyCount];

  AggregateTreeBody(const std::vector<fs::path>& paths)
      : paths(paths), property_counts{0} {}

  AggregateTreeBody(const AggregateTreeBody& x, tbb::split)
      : AggregateTreeBody(x.paths) {}

  void operator()(const tbb::blocked_range<size_t> range) {
    boost::iostreams::mapped_file_source mf;
    for (size_t i = range.begin(); i != range.end(); ++i) {
      mf.open(paths[i]);
      Span<const uint8_t> span(mf);
      BitReader reader(span);

      Tree tree;
      const size_t tree_size_limit = std::numeric_limits<int32_t>::max();
      Status status =
          JXL_STATUS(DecodeTree(&reader, &tree, tree_size_limit), "DecodeTree");
      JXL_CHECK(reader.Close());
      mf.close();

      if (!status) JXL_ABORT("Failed to decode tree: %s", paths[i].c_str());

      for (const PropertyDecisionNode& node : tree) {
        if (node.property < 0) continue;  // leaf
        if (node.property >= static_cast<int16_t>(kMaxPropertyCount)) {
          JXL_ABORT("Too large property index %" PRId16 " found in %s",
                    node.property, paths[i].c_str());
        }
        property_counts[node.property]++;
      }
    }
  }

  void join(const AggregateTreeBody& rhs) {
    for (size_t i = 0; i < kMaxPropertyCount; i++) {
      property_counts[i] += rhs.property_counts[i];
    }
  }
};

}  // namespace

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

  AggregateTreeBody body(paths);
  tbb::parallel_reduce(tbb::blocked_range<size_t>(0, paths.size(), 2), body);

  std::string header;
  std::string values;
  for (size_t i = 0; i < kMaxPropertyCount; i++) {
    if (i > 0) {
      header += ',';
      values += ',';
    }

    header += PropertyName(i);
    values += std::to_string(body.property_counts[i]);
  }

  std::cout << header << "\n" << values << std::endl;

  return 0;
}
