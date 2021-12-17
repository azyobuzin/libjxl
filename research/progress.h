#pragma once

#include <tbb/spin_mutex.h>

#include <atomic>
#include <string_view>

namespace research {

class ProgressReporter {
 public:
  virtual void report(size_t completed_jobs, size_t n_jobs) = 0;
  virtual ~ProgressReporter() {}
};

class ConsoleProgressReporter : public ProgressReporter {
  std::string_view message;
  std::atomic_int_least8_t percent;
  tbb::spin_mutex mutex;

 public:
  ConsoleProgressReporter(std::string_view message);
  void report(size_t completed_jobs, size_t n_jobs) override;
  ~ConsoleProgressReporter();
};

}  // namespace research
