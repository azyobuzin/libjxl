#include "images_provider.h"

#include <tbb/parallel_for.h>

#include <opencv2/imgcodecs.hpp>

#include "lib/jxl/modular/transform/enc_transform.h"

using namespace jxl;

namespace research {

Image FileImagesProvider::get(size_t idx) {
  const std::string& path = paths.at(idx);
  Image img = LoadImage(path, ycocg);

  if (img.error) JXL_ABORT("Failed to load %s", path.c_str());

  if (only_first_channel)
    img.channel.erase(img.channel.begin() + (img.nb_meta_channels + 1),
                      img.channel.end());

  return img;
}

Image LoadImage(const std::string& path, bool ycocg) {
  cv::Mat mat = cv::imread(path, cv::IMREAD_COLOR);

  // Imageの引数なしコンストラクタはエラーを表現する
  if (mat.empty()) return {};

  Image img(mat.cols, mat.rows, 8, 3);

  tbb::parallel_for(0, mat.rows, [&](int y) {
    const auto* src = mat.ptr<cv::Point3_<uint8_t>>(y);
    auto* dst_r = img.channel[0].Row(y);
    auto* dst_g = img.channel[1].Row(y);
    auto* dst_b = img.channel[2].Row(y);

    for (int x = 0; x < mat.cols; x++) {
      dst_b[x] = src[x].x;
      dst_g[x] = src[x].y;
      dst_r[x] = src[x].z;
    }
  });

  if (ycocg) {
    Transform t(TransformId::kRCT);
    t.rct_type = 6;
    t.begin_c = img.nb_meta_channels;
    if (TransformForward(t, img, weighted::Header{}, nullptr))
      img.transform.push_back(std::move(t));
  }

  return img;
}

}  // namespace research
