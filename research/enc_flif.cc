#include "enc_flif.h"

#include "flif/fileio.hpp"
#include "flif/flif-enc.cpp"
#include "flif/transform/ycocg.hpp"
#include "lib/jxl/base/status.h"

namespace research {

// Yチャネルの最大ビット数
constexpr int kBits = 10;

constexpr int kChannel = 3;

jxl::PaddedBytes EncodeColorSignalWithFlif(
    const std::vector<std::shared_ptr<const jxl::Image>>& images,
    int learn_repeats, int additional_props) {
  JXL_CHECK(images.size() > 0);

  flif_options options = FLIF_DEFAULT_OPTIONS;
  options.learn_repeats = learn_repeats;
  options.additional_props = additional_props;
  options.skip_p0 = 1;

  // FLIF の Image にコピー
  Images flif_images;
  flif_images.reserve(images.size());
  for (const auto& image : images) {
    JXL_CHECK(image->channel.size() - image->nb_meta_channels == kChannel);
    auto& flif_image =
        flif_images.emplace_back(image->w, image->h, 0, 255, kChannel);

    for (size_t chan = 0; chan < kChannel; chan++) {
      const auto& src = image->channel[image->nb_meta_channels + chan];
      auto& dst = flif_image.getPlane(chan);
      for (size_t y = 0; y < src.h; y++) {
        const auto& src_row = src.Row(y);
        for (size_t x = 0; x < src.w; x++) dst.set(y, x, src_row[x]);
      }
    }
  }

  auto& flif_image = flif_images[0];
  std::unique_ptr<const ColorRanges> base_ranges(getRanges(flif_image));
  ColorRangesYCoCg ranges(64, base_ranges.get());

  Progress progress;
  progress.pixels_todo = static_cast<int64_t>(flif_image.rows()) *
                         flif_image.cols() * 2 * (options.learn_repeats + 1);

  BlobIO io;
  RacOut<BlobIO> rac(io);
  std::vector<Tree> forest(ranges.numPlanes(), Tree());
  RacDummy dummy;

  // https://github.com/FLIF-hub/FLIF/blob/0074d6fd095d27ce81346aa3fbe9bab59105053e/src/flif.cpp#L323
  bool interlaced =
      flif_image.rows() * flif_image.cols() * flif_images.size() >= 10000;
  options.method.encoding =
      interlaced ? flifEncoding::interlaced : flifEncoding::nonInterlaced;

  int roughZL = 0;

  if (interlaced) {
    // 予測器を決定
    for (int p = 0; p < ranges.numPlanes(); p++) {
      if (options.predictor[p] == -2) {
        if (ranges.min(p) < ranges.max(p)) {
          options.predictor[p] = find_best_predictor(flif_images, &ranges, p, 1,
                                                     options.additional_props);
          // predictor 0 is usually the safest choice, so only pick a different
          // one if it's the best at zoomlevel 0 too
          if (options.predictor[p] > 0 &&
              find_best_predictor(flif_images, &ranges, p, 0,
                                  options.additional_props) !=
                  options.predictor[p])
            options.predictor[p] = 0;
        } else
          options.predictor[p] = 0;
      }
    }

    // 高ZLは決定木なしで符号化
    roughZL = flif_image.zooms() - NB_NOLEARN_ZOOMS - 1;
    if (roughZL < 0) roughZL = 0;
    UniformSymbolCoder<RacOut<BlobIO>> metaCoder(rac);
    metaCoder.write_int(0, flif_image.zooms(), roughZL);
    flif_encode_FLIF2_pass<
        BlobIO, RacOut<BlobIO>,
        FinalPropertySymbolCoder<FLIFBitChancePass2, RacOut<BlobIO>, kBits>>(
        io, rac, flif_images, &ranges, forest, flif_image.zooms(), roughZL + 1,
        1, options, progress);

    // 学習
    flif_encode_FLIF2_pass<
        BlobIO, RacDummy,
        PropertySymbolCoder<FLIFBitChancePass1, RacDummy, kBits>>(
        io, dummy, flif_images, &ranges, forest, roughZL, 0,
        options.learn_repeats, options, progress);
  } else {
    flif_encode_scanlines_pass<
        BlobIO, RacDummy,
        PropertySymbolCoder<FLIFBitChancePass1, RacDummy, kBits>>(
        io, dummy, flif_images, &ranges, forest, options.learn_repeats, options,
        progress);
  }

  // 決定木を出力
  flif_encode_tree<BlobIO, FLIFBitChanceTree, RacOut<BlobIO>>(
      io, rac, &ranges, forest, options.method.encoding, flif_images.size(),
      options.additional_props, options.skip_p0, options.print_tree);

  options.divisor = 0;
  options.min_size = 0;
  options.split_threshold = 0;

  // 符号化
  if (interlaced) {
    flif_encode_FLIF2_pass<
        BlobIO, RacOut<BlobIO>,
        FinalPropertySymbolCoder<FLIFBitChancePass2, RacOut<BlobIO>, kBits>>(
        io, rac, flif_images, &ranges, forest, roughZL, 0, 1, options,
        progress);
  } else {
    flif_encode_scanlines_pass<
        BlobIO, RacOut<BlobIO>,
        FinalPropertySymbolCoder<FLIFBitChancePass2, RacOut<BlobIO>, kBits>>(
        io, rac, flif_images, &ranges, forest, 1, options, progress);
  }

  rac.flush();

  // バイト列のコピー
  size_t array_size;
  uint8_t* buf_ptr = io.release(&array_size);
  jxl::PaddedBytes result;
  result.append(buf_ptr, buf_ptr + array_size);
  delete[] buf_ptr;
  return result;
}

}  // namespace research
