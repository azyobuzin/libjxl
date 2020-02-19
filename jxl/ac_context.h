// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef JXL_AC_CONTEXT_H_
#define JXL_AC_CONTEXT_H_

#include "jxl/base/bits.h"
#include "jxl/base/status.h"
#include "jxl/coeff_order_fwd.h"

namespace jxl {

// Block context used for scanning order, number of non-zeros, AC coefficients.
// Equal to the channel.
constexpr uint32_t kDCTOrderContextStart = 0;
constexpr uint32_t kOrderContexts = 10;

// The number of predicted nonzeros goes from 0 to 1008. We use
// ceil(log2(predicted+1)) as a context for the number of nonzeros, so from 0 to
// 10, inclusive.
constexpr uint32_t kNonZeroBuckets = 37;

static const uint16_t kCoeffFreqContext[64] = {
    0xBAD, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15,    15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
    23,    23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
    27,    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
};

static const uint16_t kCoeffNumNonzeroContext[64] = {
    0xBAD, 0,   31,  62,  62,  93,  93,  93,  93,  123, 123, 123, 123,
    152,   152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
    180,   180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
};

// Supremum of ZeroDensityContext(x, y) + 1.
constexpr int kZeroDensityContextCount = 458;

/* This function is used for entropy-sources pre-clustering.
 *
 * Ideally, each combination of |nonzeros_left| and |k| should go to its own
 * bucket; but it implies (64 * 63 / 2) == 2016 buckets. If there is other
 * dimension (e.g. block context), then number of primary clusters becomes too
 * big.
 *
 * To solve this problem, |nonzeros_left| and |k| values are clustered. It is
 * known that their sum is at most 64, consequently, the total number buckets
 * is at most A(64) * B(64).
 */
// TODO(user): investigate, why disabling pre-clustering makes entropy code
// less dense. Perhaps we would need to add HQ clustering algorithm that would
// be able to squeeze better by spending more CPU cycles.
static JXL_INLINE size_t ZeroDensityContext(size_t nonzeros_left, size_t k,
                                            size_t covered_blocks,
                                            size_t log2_covered_blocks,
                                            size_t prev) {
  JXL_DASSERT(1 << log2_covered_blocks == covered_blocks);
  nonzeros_left = (nonzeros_left + covered_blocks - 1) >> log2_covered_blocks;
  k >>= log2_covered_blocks;
  JXL_DASSERT(k > 0);
  JXL_DASSERT(k < 64);
  JXL_DASSERT(nonzeros_left > 0);
  JXL_DASSERT(nonzeros_left + k < 65);
  return (kCoeffNumNonzeroContext[nonzeros_left] + kCoeffFreqContext[k]) * 2 +
         prev;
}

// Context map for AC coefficients consists of 2 blocks:
//  |kOrderContexts x          : context for number of non-zeros in the block
//   kNonZeroBuckets|            computed from block context and predicted value
//                               (based top and left values)
//  |kOrderContexts x          : context for AC coefficient symbols,
//   kZeroDensityContextCount|   computed from block context,
//                               number of non-zeros left and
//                               index in scan order
constexpr uint32_t kNumContexts = (kOrderContexts * kNonZeroBuckets) +
                                  (kOrderContexts * kZeroDensityContextCount);

// Non-zero context is based on number of non-zeros and block context.
// For better clustering, contexts with same number of non-zeros are grouped.
static inline uint32_t NonZeroContext(uint32_t non_zeros, uint32_t block_ctx) {
  uint32_t ctx;
  if (non_zeros >= 64) non_zeros = 64;
  if (non_zeros < 8)
    ctx = non_zeros;
  else
    ctx = 4 + non_zeros / 2;
  return ctx * kOrderContexts + block_ctx;
}

// Non-zero context is based on number of non-zeros and block context.
// For better clustering, contexts with same number of non-zeros are grouped.
constexpr uint32_t ZeroDensityContextsOffset(uint32_t block_ctx) {
  return kOrderContexts * kNonZeroBuckets +
         kZeroDensityContextCount * block_ctx;
}

}  // namespace jxl

#endif  // JXL_AC_CONTEXT_H_