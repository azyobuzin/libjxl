#include "jxl_parallel.h"

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <thread>

#include "lib/jxl/base/status.h"

namespace research {

JxlParallelRetCode TbbParallelRunner([[maybe_unused]] void* runner_opaque,
                                     void* jpegxl_opaque,
                                     JxlParallelRunInit init,
                                     JxlParallelRunFunction func,
                                     uint32_t start_range, uint32_t end_range) {
  if (start_range > end_range) return -1;
  if (start_range == end_range) return 0;

  const size_t n_thread =
      std::min<size_t>(std::max<size_t>(std::thread::hardware_concurrency(), 1),
                       end_range - start_range);

  int ret = init(jpegxl_opaque, n_thread);
  if (ret != 0) return ret;

  auto worker = [&](const tbb::blocked_range<uint32_t>& r) {
    size_t thread_id = std::max(tbb::task_arena::current_thread_index(), 0);
    JXL_ASSERT(thread_id < n_thread);

    for (uint32_t task = r.begin(); task != r.end(); task++)
      func(jpegxl_opaque, task, thread_id);
  };

  tbb::task_arena arena(n_thread);
  arena.execute([&] {
    tbb::parallel_for(tbb::blocked_range<uint32_t>(start_range, end_range),
                      worker);
  });

  return 0;
}

}  // namespace research
