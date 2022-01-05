// ディレクトリの画像から平均画像と、その画像からの差分画像を作成する

#include <fmt/core.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>

#include "enc_cluster.h"
#include "images_provider.h"
#include "jxl_parallel.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/modular/transform/enc_transform.h"
#include "lib/jxl/modular/transform/transform.h"
#include "progress.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace research;

namespace jxl {

bool do_transform(Image& image, const Transform& tr,
                  const weighted::Header& wp_header,
                  jxl::ThreadPool* pool = nullptr);

}  // namespace jxl

namespace {

struct SumImageBody {
  const std::vector<jxl::Image>& images;
  jxl::Image sum_img;

  SumImageBody(const std::vector<jxl::Image>& images) noexcept
      : images(images) {}

  SumImageBody(const SumImageBody& other, tbb::split) noexcept
      : images(other.images) {}

  void AddImage(const jxl::Image& img) {
    JXL_ASSERT(!sum_img.error);
    JXL_CHECK(sum_img.channel.size() == img.channel.size());

    for (size_t j = 0; j < sum_img.channel.size(); j++) {
      const jxl::Channel& src_chan = img.channel[j];
      jxl::Channel& dst_chan = sum_img.channel[j];
      JXL_CHECK(src_chan.w == dst_chan.w && src_chan.h == dst_chan.h);

      for (size_t y = 0; y < src_chan.h; y++) {
        const jxl::pixel_type* src_row = src_chan.Row(y);
        jxl::pixel_type* dst_row = dst_chan.Row(y);
        for (size_t x = 0; x < src_chan.w; x++) dst_row[x] += src_row[x];
      }
    }
  }

  void operator()(const tbb::blocked_range<size_t>& range) {
    size_t i = range.begin();

    if (sum_img.error) {
      // sum_imgが未初期化なので、クローンで読み込む
      sum_img = images.at(i).clone();
      i++;
    }

    for (; i < range.end(); i++) AddImage(images.at(i));
  }

  void join(SumImageBody& rhs) {
    if (sum_img.error)
      sum_img = std::move(rhs.sum_img);
    else
      AddImage(rhs.sum_img);
  }
};

bool EncodeAndWrite(jxl::Image image, const jxl::ModularOptions& options,
                    const fs::path& dst_path) {
  // enc_without_header_main.cc とほぼ同じ
  CombinedImage ci = CombineImage(std::move(image));
  jxl::BitWriter writer;
  jxl::ThreadPool pool(TbbParallelRunner, nullptr);

  jxl::CompressParams cparams;
  cparams.SetLossless();

  // Global palette
  jxl::Transform global_palette(jxl::TransformId::kPalette);
  global_palette.begin_c = ci.image.nb_meta_channels;
  global_palette.num_c = ci.image.channel.size() - ci.image.nb_meta_channels;
  global_palette.nb_colors = std::min((int)(ci.image.w * ci.image.h / 8),
                                      std::abs(cparams.palette_colors));
  global_palette.ordered_palette = cparams.palette_colors >= 0;
  global_palette.lossy_palette = false;
  if (jxl::TransformForward(global_palette, ci.image, {}, &pool)) {
    ci.image.transform.push_back(std::move(global_palette));
  }

  // Local channel palette
  JXL_ASSERT(cparams.channel_colors_percent > 0);
  for (size_t i = ci.image.nb_meta_channels; i < ci.image.channel.size(); i++) {
    int min, max;
    jxl::compute_minmax(ci.image.channel[i], &min, &max);
    int colors = max - min + 1;
    jxl::Transform local_palette(jxl::TransformId::kPalette);
    local_palette.begin_c = i;
    local_palette.num_c = 1;
    local_palette.nb_colors =
        std::min((int)(ci.image.w * ci.image.h * 0.8),
                 (int)(cparams.channel_colors_percent / 100. * colors));
    if (jxl::do_transform(ci.image, local_palette, {}, &pool)) {
      ci.image.transform.push_back(std::move(local_palette));
    }
  }

  jxl::ModularOptions local_options = options;
  local_options.wp_mode = FindBestWPMode(ci.image);

  jxl::Tree tree = LearnTree(writer, ci, local_options, 0);
  EncodeImages(writer, ci, local_options, 0, tree);
  writer.ZeroPadToByte();
  jxl::Span<const uint8_t> span = writer.GetSpan();

  bool failed = false;
  FILE* fp = fopen(dst_path.c_str(), "wb");
  if (fp) {
    if (fwrite(span.data(), 1, span.size(), fp) != span.size()) {
      std::cerr << "Failed to write " << dst_path.c_str() << std::endl;
      failed = true;
    }
    fclose(fp);
  } else {
    std::cerr << "Failed to open " << dst_path.c_str() << std::endl;
    failed = true;
  }

  return !failed;
}

}  // namespace

int main(int argc, char* argv[]) {
  po::options_description pos_ops;
  pos_ops.add_options()(
      "image-file",
      po::value<std::vector<std::string>>()->multitoken()->required());

  po::positional_options_description pos_desc;
  pos_desc.add("image-file", -1);

  po::options_description ops_desc;
  // clang-format off
  ops_desc.add_options()
    ("fraction", po::value<float>()->default_value(.5f), "サンプリングする画素の割合 (0, 1]")
    ("refchan", po::value<uint16_t>()->default_value(0), "画像内のチャンネル参照数")
    ("out-dir", po::value<fs::path>()->required(), "圧縮結果の出力先ディレクトリ");
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
              << "Usage: enc_avg [OPTIONS] IMAGE_FILE..." << std::endl
              << ops_desc << std::endl;
    return 1;
  }

  const std::vector<std::string>& paths =
      vm["image-file"].as<std::vector<std::string>>();
  FileImagesProvider images(paths);
  images.ycocg = true;

  // Tortoise相当
  jxl::ModularOptions options{
      .nb_repeats = vm["fraction"].as<float>(),
      .max_properties = vm["refchan"].as<uint16_t>(),
      .splitting_heuristics_properties = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3,
                                          4, 5, 6, 7, 8},
      .max_property_values = 256,
      .predictor = jxl::Predictor::Variable};

  const size_t n_images = images.size();
  const size_t n_jobs = n_images + 1;
  ConsoleProgressReporter progress("Encoding");

  // すべての画像を使用するので、すべてロードする
  std::vector<jxl::Image> images_vec(n_images);
  tbb::parallel_for(size_t(0), n_images,
                    [&](size_t i) { images_vec[i] = images.get(i); });

  const fs::path& out_dir = vm["out-dir"].as<fs::path>();
  fs::create_directories(out_dir);

  // 平均画像を求める
  jxl::Image avg_img;
  {
    SumImageBody body(images_vec);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n_images, 2), body);
    avg_img = std::move(body.sum_img);

    for (jxl::Channel& chan : avg_img.channel) {
      for (size_t y = 0; y < chan.h; y++) {
        jxl::pixel_type* row = chan.Row(y);
        for (size_t x = 0; x < chan.w; x++) row[x] /= n_images;
      }
    }
  }

  EncodeAndWrite(avg_img.clone(), options, out_dir / "avg.bin");
  progress.report(1, n_jobs);

  std::atomic_size_t n_completed = 1;
  std::atomic_bool failed = false;

  // 差分画像を求める
  tbb::parallel_for(size_t(0), n_images, [&](size_t i) {
    jxl::Image& img = images_vec.at(i);

    for (size_t c = 0; c < img.channel.size(); c++) {
      const jxl::Channel& avg_chan = avg_img.channel[c];
      jxl::Channel& img_chan = img.channel[c];
      JXL_ASSERT(avg_chan.w == img_chan.w && avg_chan.h == img_chan.h);

      for (size_t y = 0; y < img_chan.h; y++) {
        const jxl::pixel_type* avg_row = avg_chan.Row(y);
        jxl::pixel_type* img_row = img_chan.Row(y);
        for (size_t x = 0; x < img_chan.w; x++) {
          img_row[x] -= avg_row[x];
        }
      }
    }

    if (!EncodeAndWrite(std::move(img), options,
                        out_dir / fmt::format("diff{}.bin", i))) {
      failed = true;
    }

    progress.report(++n_completed, n_jobs);
  });

  return failed ? 1 : 0;
}
