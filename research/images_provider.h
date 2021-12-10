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

}  // namespace research
