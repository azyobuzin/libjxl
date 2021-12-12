#include <optional>

#include "lib/jxl/modular/modular_image.h"

namespace research {

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
};

jxl::Image LoadImage(const std::string& path);

}  // namespace research
