#include "dec_flif.h"

#include "flif/fileio.hpp"
#include "flif/flif-dec.cpp"
#include "flif/transform/ycocg.hpp"

namespace research {

// Yチャネルの最大ビット数
constexpr int kBits = 10;

constexpr int kChannel = 3;

jxl::Status DecodeColorSignalWithFlif(std::vector<jxl::Image>& images,
                                      jxl::Span<const uint8_t> data,
                                      int additional_props) {
  JXL_CHECK(images.size() > 0);

  flif_options options = FLIF_DEFAULT_OPTIONS;
  options.additional_props = additional_props;

  bool interlaced = images[0].w * images[0].h * images.size() >= 10000;
  options.method.encoding =
      interlaced ? flifEncoding::interlaced : flifEncoding::nonInterlaced;

  Images flif_images;
  flif_images.reserve(images.size());
  for (const auto& image : images) {
    JXL_CHECK(image.channel.size() - image.nb_meta_channels == 1);
    auto& flif_image =
        flif_images.emplace_back(image.w, image.h, 0, 255, kChannel);

    // Yチャネルをコピー
    const auto& src = image.channel[image.nb_meta_channels];
    auto& dst = flif_image.getPlane(0);
    for (size_t y = 0; y < src.h; y++) {
      const auto& src_row = src.Row(y);
      for (size_t x = 0; x < src.w; x++) dst.set(y, x, src_row[x]);
    }
  }

  auto& flif_image = flif_images[0];
  std::unique_ptr<const ColorRanges> base_ranges(getRanges(flif_image));
  ColorRangesYCoCg ranges(64, base_ranges.get());

  Progress progress;
  progress.pixels_todo =
      static_cast<int64_t>(flif_image.rows()) * flif_image.cols() * 2;

  BlobReader io(data.data(), data.size());
  RacIn<BlobReader> rac(io);
  std::vector<Transform<BlobReader>*> transforms;
  std::vector<Tree> forest(ranges.numPlanes(), Tree());

  int roughZL = 0;
  if (interlaced) {
    UniformSymbolCoder<RacIn<BlobReader>> metaCoder(rac);
    roughZL = metaCoder.read_int(0, flif_image.zooms());
    if (!flif_decode_FLIF2_pass<
            BlobReader, RacIn<BlobReader>,
            FinalPropertySymbolCoder<FLIFBitChancePass2, RacIn<BlobReader>,
                                     kBits>>(
            io, rac, flif_images, &ranges, forest, flif_image.zooms(),
            roughZL + 1, options, transforms, /*callback=*/nullptr,
            /*user_data=*/nullptr, flif_images, progress))
      return JXL_STATUS(false, "failed to decode rough image");
  }

  if (!flif_decode_tree<BlobReader, FLIFBitChanceTree, RacIn<BlobReader>>(
          io, rac, &ranges, forest, options.method.encoding, images.size(),
          options.additional_props, options.print_tree))
    return JXL_STATUS(false, "flif_decode_tree");

  if (interlaced) {
    if (!flif_decode_FLIF2_pass<
            BlobReader, RacIn<BlobReader>,
            FinalPropertySymbolCoder<FLIFBitChancePass2, RacIn<BlobReader>,
                                     kBits>>(
            io, rac, flif_images, &ranges, forest, roughZL, 0, options,
            transforms, /*callback=*/nullptr, /*user_data=*/nullptr,
            flif_images, progress))
      return JXL_STATUS(false, "flif_decode_FLIF2_pass");
  } else {
    if (!flif_decode_scanlines_pass<
            BlobReader, RacIn<BlobReader>,
            FinalPropertySymbolCoder<FLIFBitChancePass2, RacIn<BlobReader>,
                                     kBits>>(
            io, rac, flif_images, &ranges, forest, options, transforms,
            /*callback=*/nullptr, /*user_data=*/nullptr, flif_images, progress))
      return JXL_STATUS(false, "flif_decode_scanlines_pass");
  }

  // デコード結果を書き戻す
  for (size_t i = 0; i < images.size(); i++) {
    for (size_t chan = 1; chan < kChannel; chan++) {
      auto& src_plane = flif_images[i].getPlane(chan);
      auto& dst_chan = images[i].channel.emplace_back(images[i].w, images[i].h);
      for (size_t y = 0; y < images[i].h; y++) {
        jxl::pixel_type* dst_row = dst_chan.Row(y);
        for (size_t x = 0; x < images[i].w; x++) {
          dst_row[x] = src_plane.get(y, x);
        }
      }
    }
  }

  return true;
}

}  // namespace research
