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

#ifndef JXL_MODULAR_MA_RAC_ENC_H_
#define JXL_MODULAR_MA_RAC_ENC_H_

#include <stddef.h>
#include <stdint.h>

#include "jxl/modular/ma/rac.h"

namespace jxl {

template <class Config, typename IO>
class RacOutput {
 public:
  typedef typename Config::data_t rac_t;

 protected:
  IO& io;

 private:
  rac_t range;
  rac_t low;
  int delayed_byte;
  int delayed_count;

  void inline output() {
    while (range <= Config::MIN_RANGE) {
      int byte = low >> Config::MIN_RANGE_BITS;
      if (delayed_byte < 0) {  // first generated byte
        delayed_byte = byte;
      } else if (((low + range) >> 8) <
                 Config::MIN_RANGE) {  // definitely no overflow
        io.fputc(delayed_byte);
        while (delayed_count) {
          io.fputc(0xFF);
          delayed_count--;
        }
        delayed_byte = byte;
      } else if ((low >> 8) >= Config::MIN_RANGE) {  // definitely overflow
        io.fputc(delayed_byte + 1);
        while (delayed_count) {
          io.fputc(0);
          delayed_count--;
        }
        delayed_byte = byte & 0xFF;
      } else {
        delayed_count++;
      }
      low = (low & (Config::MIN_RANGE - 1)) << 8;
      range <<= 8;
    }
  }
  void inline put(rac_t chance, bool bit) {
    JXL_DASSERT(chance >= 0);
    JXL_DASSERT(chance < range);
    if (bit) {
      low += range - chance;
      range = chance;
    } else {
      range -= chance;
    }
    output();
  }

 public:
  explicit RacOutput(IO& ioin)
      : io(ioin),
        range(Config::BASE_RANGE),
        low(0),
        delayed_byte(-1),
        delayed_count(0) {}

  void inline write_12bit_chance(uint16_t b12, bool bit) {
    put(Config::chance_12bit_chance(b12, range), bit);
  }

  void inline write_bit(bool bit) { put(range >> 1, bit); }

  void inline flush() {
    low += (Config::MIN_RANGE - 1);
    // is this the correct way to reliably flush?
    range = Config::MIN_RANGE - 1;
    output();
    range = Config::MIN_RANGE - 1;
    output();
    range = Config::MIN_RANGE - 1;
    output();
    range = Config::MIN_RANGE - 1;
    output();
  }
};

template <typename IO>
class RacDummy {
 public:
  static void inline write_12bit_chance(uint16_t /*b12*/, bool /*unused*/) {}
  static void inline write_bit(bool /*unused*/) {}
  static void inline flush() {}

  explicit RacDummy(IO& io) {}
};

template <typename IO>
class RacOutput24 : public RacOutput<RacConfig24, IO> {
 public:
  explicit RacOutput24(IO& io) : RacOutput<RacConfig24, IO>(io) {}
};

template <typename IO>
using RacOut = RacOutput24<IO>;

}  // namespace jxl

#endif  // JXL_MODULAR_MA_RAC_ENC_H_