// SPDX-License-Identifier: Apache-2.0
#include "sampler.h"
#include "tokenizer.h"
#include <iostream>
#include <stdexcept>
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("vocab argument missing");
    trie_tokenizer t;
    if (t.load(argv[1]) != 0)
      throw std::runtime_error("vocab load failed");
    for (const auto &text :
         std::vector<std::string>{"Hello RWKV!", "你好，世界！\n",
                                  "a\tb\\c\"d'", std::string("\0\1\xff", 3)})
      if (t.decode(t.encode(text)) != text)
        throw std::runtime_error("Tokenizer roundtrip failed");
    rwkvmobile::NucleusSampler sampler;
    std::vector<float> data = {-3, 2, 8, 1};
    rwkvmobile::Tensor1D view{data.data(), rwkvmobile::TensorDType::F32,
                              data.size()};
    if (sampler.sample(view, 4, 1, 1, 1) != 2)
      throw std::runtime_error("Greedy sampling failed");
    std::map<int, float> counts{{2, 2}};
    sampler.apply_penalties(view, 4, counts, {}, 1, 0.5, 0.9);
    if (data[2] != 6 || std::abs(counts[2] - 1.8f) > 1e-6)
      throw std::runtime_error("Penalty failed");
    sampler.set_seed(42);
    std::vector<int> sequence;
    for (int i = 0; i < 50; ++i)
      sequence.push_back(sampler.sample(view, 4, 1, 4, 1));
    sampler.set_seed(42);
    for (int i : sequence)
      if (i != sampler.sample(view, 4, 1, 4, 1))
        throw std::runtime_error("Seed not reproducible");
    std::cout << "Tokenizer, greedy, penalties, seeded sampling passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
