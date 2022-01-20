// EncodeClusterPointers / DecodeClusterPointers の動作確認

#include <algorithm>
#include <iostream>
#include <random>

#include "dec_cluster.h"
#include "enc_cluster.h"

using namespace research;
using namespace jxl;

int main(void) {
  std::vector<uint32_t> pointers(10);
  for (size_t i = 0; i < pointers.size(); i++) pointers[i] = i;

  {
    std::random_device seed_gen;
    std::mt19937 rng(seed_gen());
    std::shuffle(pointers.begin(), pointers.end(), rng);
  }

  std::cout << "Input: ";
  for (auto x : pointers) {
    std::cout << x << ", ";
  }
  std::cout << std::endl;

  BitWriter writer;
  EncodeClusterPointers(writer, pointers);

  std::cout << writer.BitsWritten() << " bits written" << std::endl;

  writer.ZeroPadToByte();
  BitReader reader(writer.GetSpan());
  std::vector<uint32_t> decoded_pointers(pointers.size());
  DecodeClusterPointers(reader, decoded_pointers);

  std::cout << "Decoded: ";
  for (auto x : decoded_pointers) {
    std::cout << x << ", ";
  }
  std::cout << std::endl;

  JXL_CHECK(reader.Close());

  bool ok = std::equal(pointers.begin(), pointers.end(),
                       decoded_pointers.begin(), decoded_pointers.end());
  std::cout << (ok ? "OK\n" : "NG\n");
  return ok ? 0 : 1;
}
