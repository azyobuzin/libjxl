#include <optional>

#include "lib/jxl/modular/modular_image.h"

namespace research {

class ImagesProvider {
 public:
  virtual size_t size() const noexcept = 0;
  virtual jxl::Image get(size_t idx) = 0;
  virtual std::string get_label(size_t idx);

 protected:
  virtual ~ImagesProvider() {}
};

class FileImagesProvider : public ImagesProvider {
  std::vector<std::string> paths;

 public:
  FileImagesProvider(std::vector<std::string> paths) : paths(paths) {}
  size_t size() const noexcept override { return paths.size(); }
  jxl::Image get(size_t idx) override;
  std::string get_label(size_t idx) override;

  // YCoCg 変換を行うか
  bool ycocg = false;

  // 最初のチャネル(Y)だけを使うか
  bool only_first_channel = false;
};

jxl::Image LoadImage(const std::string& path, bool ycocg);

}  // namespace research
