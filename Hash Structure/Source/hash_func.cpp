#include "hash_func.h"

std::vector<int> hashFunc::simpleHash (double x) {    
        // 1. Phải sử dụng 64-bit (long long) vì 10^10 lớn hơn 32-bit int
        // ULL = Unsigned Long Long
        const uint64_t multiplier = 10000000000ULL; 
        uint64_t input_64 = static_cast<uint64_t>(x);
        uint64_t result_64 = input_64 * multiplier;
        uint32_t result_32 = static_cast<uint32_t>(result_64);
    
        std::bitset<32> b(result_32);
        
        std::vector<int> bits(32);
        for (size_t i = 0; i < 32; ++i) {
            bits[i] = b[31 - i];
        }
        
        return bits;
}