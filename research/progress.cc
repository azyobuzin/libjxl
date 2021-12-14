#include "progress.h"

#include <iostream>

namespace research {

ConsoleProgressReporter::ConsoleProgressReporter(std::string_view message)
    : message(message), percent(0) {
  std::cerr << message << ": 0%\r";
  std::cerr.flush();
}

void ConsoleProgressReporter::report(size_t completed_jobs, size_t n_jobs) {
  int_least8_t new_percent = completed_jobs * 100.0 / n_jobs;
  auto old_percent = percent.load();
  if (new_percent > old_percent &&
      percent.compare_exchange_weak(old_percent, new_percent)) {
    {
      tbb::spin_mutex::scoped_lock lock(mutex);
      std::cerr << message << ": " << static_cast<int>(new_percent) << "%\r";
    }
    std::cerr.flush();
  }
}

ConsoleProgressReporter::~ConsoleProgressReporter() { std::cerr << std::endl; }

}  // namespace research
