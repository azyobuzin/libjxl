#include <optional>

#include "lib/jxl/modular/modular_image.h"

namespace research {

// TODO: next, reset ではなくランダムアクセスできるようにするほうが並列化に対応できる
//       例えば size_t size() と Image get(size_t idx)

class ImagesProvider {
 public:
  virtual std::optional<jxl::Image> next() = 0;
  virtual void reset() = 0;

 protected:
  virtual ~ImagesProvider() {}
};

class FileImagesProvider : public ImagesProvider {
  std::vector<std::string> paths;
  size_t current_idx;

  public:
  FileImagesProvider(std::vector<std::string> paths) : paths(paths), current_idx(0) {}
  std::optional<jxl::Image> next() override;
  void reset() override;

  // YCoCg 変換を行うか
  bool ycocg = false;

  // 最初のチャネル(Y)だけを使うか
  bool only_first_channel = false;
};

jxl::Image LoadImage(const std::string& path, bool ycocg);

}  // namespace research
