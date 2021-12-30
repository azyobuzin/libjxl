#include "jxl/parallel_runner.h"

namespace research {

JxlParallelRetCode TbbParallelRunner(void* runner_opaque, void* jpegxl_opaque,
                                     JxlParallelRunInit init,
                                     JxlParallelRunFunction func,
                                     uint32_t start_range, uint32_t end_range);

}
