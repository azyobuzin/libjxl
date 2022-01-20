#include "progress.h"

#include <iostream>

namespace research {

ConsoleProgressReporter::ConsoleProgressReporter(std::string_view message)
    : message(message), percent(0), closed(false) {
  std::cerr << message << ": 0%\r";
  std::cerr.flush();
}

void ConsoleProgressReporter::report(size_t completed_jobs, size_t n_jobs) {
  int_least8_t new_percent = completed_jobs * 100.0 / n_jobs;
  while (!closed.load(std::memory_order_acquire)) {
    auto old_percent = percent.load();
    if (new_percent <= old_percent) return;
    if (percent.compare_exchange_weak(old_percent, new_percent)) {
      // 表示すべきパーセンテージが増加したので、ロックを取って描画する
      {
        tbb::spin_mutex::scoped_lock lock(mutex);
        if (new_percent < percent.load()) return;  // ロック内で再チェック
        std::cerr << message << ": " << static_cast<int>(new_percent) << "%\r";
      }
      std::cerr.flush();
      return;
    }
  }
}

void ConsoleProgressReporter::close() {
  if (!closed.exchange(true, std::memory_order_release)) std::cerr << std::endl;
}

ConsoleProgressReporter::~ConsoleProgressReporter() { close(); }

}  // namespace research
