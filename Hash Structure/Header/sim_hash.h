#ifndef SIM_HASH_H
#define SIM_HASH_H

#include "MurmurHash3.h"
#include <iostream>
#include <vector>
#include <list>
#include <cstring>
#include <random>
#include <algorithm>
#include <cmath>

constexpr size_t HASH_BITS = 128;

class SimHash {
private:
    size_t bits;                        
    std::vector<double> idfWeights;     // Trọng số IDF cho từng đặc trưng

public:
    explicit SimHash(size_t bit = HASH_BITS);
    ~SimHash();
    std::vector<int> hashify_double_by_murmur128(double x, int idx, int bits_count = 64);
    std::string encode_double(double x, int idx);
    void IDF(const std::vector<std::vector<double>>& allFeatures);
    size_t hashFunction(const std::vector<double>& featureVector);
};

#endif // SIM_HASH_H
