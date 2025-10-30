#include "hash_func.h"

std::vector<int> hashFunc::simpleHash (double x) {
    std::uint32_t seed = static_cast<std::uint32_t>(std::round(x * 100000));
    std::mt19937 rng(seed); 

    std::vector<int> hash(bit);
    for (int j = 0; j < bit; ++j) {
        hash[j] = rng() % 2; 
    return hash;
}