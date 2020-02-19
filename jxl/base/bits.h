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

#ifndef JXL_BASE_BITS_H_
#define JXL_BASE_BITS_H_

// Specialized instructions for processing register-sized bit arrays.

#ifdef _MSC_VER
#include <intrin.h>
#include <nmmintrin.h>
#endif

#include <stddef.h>
#include <stdint.h>

#include <hwy/type_traits.h>

#include "jxl/base/compiler_specific.h"

namespace jxl {

static JXL_INLINE size_t PopCount(hwy::SizeTag<4> /* tag */, const uint32_t x) {
#ifdef _MSC_VER
  return _mm_popcnt_u32(x);
#else
  return static_cast<size_t>(__builtin_popcount(x));
#endif
}
static JXL_INLINE size_t PopCount(hwy::SizeTag<8> /* tag */, const uint64_t x) {
#ifdef _MSC_VER
  return _mm_popcnt_u64(x);
#else
  return static_cast<size_t>(__builtin_popcountll(x));
#endif
}
template <typename T>
static JXL_INLINE size_t PopCount(T x) {
  static_assert(!hwy::IsSigned<T>(), "PopCount: use unsigned");
  return PopCount(hwy::SizeTag<sizeof(T)>(), x);
}

// Undefined results for x == 0.
static JXL_INLINE size_t NumZeroBitsAboveMSBNonzero(hwy::SizeTag<4> /* tag */,
                                                    const uint32_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanReverse(&index, x);
  return 31 - index;
#else
  return static_cast<size_t>(__builtin_clz(x));
#endif
}
static JXL_INLINE size_t NumZeroBitsAboveMSBNonzero(hwy::SizeTag<8> /* tag */,
                                                    const uint64_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanReverse64(&index, x);
  return 63 - index;
#else
  return static_cast<size_t>(__builtin_clzll(x));
#endif
}
template <typename T>
static JXL_INLINE size_t NumZeroBitsAboveMSBNonzero(const T x) {
  static_assert(!hwy::IsSigned<T>(),
                "NumZeroBitsAboveMSBNonzero: use unsigned");
  return NumZeroBitsAboveMSBNonzero(hwy::SizeTag<sizeof(T)>(), x);
}

// Undefined results for x == 0.
static JXL_INLINE size_t NumZeroBitsBelowLSBNonzero(hwy::SizeTag<4> /* tag */,
                                                    const uint32_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward(&index, x);
  return index;
#else
  return static_cast<size_t>(__builtin_ctz(x));
#endif
}
static JXL_INLINE size_t NumZeroBitsBelowLSBNonzero(hwy::SizeTag<8> /* tag */,
                                                    const uint64_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward64(&index, x);
  return index;
#else
  return static_cast<size_t>(__builtin_ctzll(x));
#endif
}
template <typename T>
static JXL_INLINE size_t NumZeroBitsBelowLSBNonzero(T x) {
  static_assert(!hwy::IsSigned<T>(),
                "NumZeroBitsBelowLSBNonzero: use unsigned");
  return NumZeroBitsBelowLSBNonzero(hwy::SizeTag<sizeof(T)>(), x);
}

// Returns bit width for x == 0.
template <typename T>
static JXL_INLINE size_t NumZeroBitsAboveMSB(const T x) {
  return (x == 0) ? sizeof(T) * 8 : NumZeroBitsAboveMSBNonzero(x);
}

// Returns bit width for x == 0.
template <typename T>
static JXL_INLINE size_t NumZeroBitsBelowLSB(const T x) {
  return (x == 0) ? sizeof(T) * 8 : NumZeroBitsBelowLSBNonzero(x);
}

// Returns base-2 logarithm, rounded down.
template <typename T>
static JXL_INLINE size_t FloorLog2Nonzero(const T x) {
  return (sizeof(T) * 8 - 1) ^ NumZeroBitsAboveMSBNonzero(x);
}

// Returns base-2 logarithm, rounded up.
template <typename T>
static JXL_INLINE size_t CeilLog2Nonzero(const T x) {
  const size_t floor_log2 = FloorLog2Nonzero(x);
  if ((x & (x - 1)) == 0) return floor_log2;  // power of two
  return floor_log2 + 1;
}

// Reverses bit order.
static JXL_INLINE uint8_t FlipByte(const uint8_t x) {
  // TODO(veluca): consider trying out alternative strategies, such as a single
  // LUT.
  static constexpr uint8_t kNibbleLut[16] = {
      0b0000, 0b1000, 0b0100, 0b1100, 0b0010, 0b1010, 0b0110, 0b1110,
      0b0001, 0b1001, 0b0101, 0b1101, 0b0011, 0b1011, 0b0111, 0b1111,
  };
  return (kNibbleLut[x & 0xF] << 4) | kNibbleLut[x >> 4];
}

}  // namespace jxl

#endif  // JXL_BASE_BITS_H_