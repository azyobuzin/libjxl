#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <ostream>

#include "images_provider.h"
#include "lib/jxl/modular/options.h"
#include "progress.h"

namespace research {

// ある画像を利用して別の画像を圧縮するときのコストを表すグラフ。
// 頂点0はゼロ画像を表し、頂点1～Nが各画像を表す。
// 辺の重みが圧縮コストを表す。頂点0からの重みは、1枚だけで圧縮したときのコストを表す。
typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::bidirectionalS,
    boost::property<boost::vertex_name_t, std::string>,
    boost::property<boost::edge_weight_t, float>>
    Graph;

struct ImageTree {
  size_t image_idx;
  std::shared_ptr<ImageTree> parent;
  std::vector<std::shared_ptr<ImageTree>> children;
  std::vector<float> costs;
};

// ImageTree を DOT (Graphviz) 形式で出力する
void PrintImageTreeDot(std::ostream &dst, std::shared_ptr<ImageTree> root,
                       ImagesProvider *images);

// ある画像から学習した決定木を使って別の画像を圧縮したときのサイズを利用して、コストグラフを作成する。
Graph CreateGraphWithDifferentTree(ImagesProvider &images,
                                   const jxl::ModularOptions &options,
                                   ProgressReporter *progress);

// 1枚だけで圧縮したときのコストがもっとも小さい画像を根としてMSTを求める。
std::shared_ptr<ImageTree> CreateMstWithDifferentTree(
    ImagesProvider &images, const jxl::ModularOptions &options,
    ProgressReporter *progress);

}  // namespace research
