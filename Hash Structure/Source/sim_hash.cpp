#include "sim_hash.h"
#include <cmath>
#include <bitset>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <iostream>

// ================== Constructor / Destructor ==================
SimHash::SimHash(size_t bit) : bits(bit) {}

SimHash::~SimHash() {}

// ================== Helper functions ==================
std::string SimHash::encode_double(double x, int idx) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%d:%.9f", idx, round(x));
    return std::string(buf);
}


std::vector<int> SimHash::hashify_double_by_murmur128(double x, int idx, int bits_count) {
    std::string token = encode_double(x, idx);

    uint64_t hash_output[2];
    MurmurHash3_x64_128(token.c_str(), token.size(), 0, hash_output);

    std::vector<int> bits(bits_count);

    if (bits_count <= 64) {
        for (int i = 0; i < bits_count; ++i) {
            uint64_t bit = (hash_output[0] >> i) & 1;
            bits[i] = bit ? 1 : -1;
        }
    } else if (bits_count <= 128) {
        for (int i = 0; i < 64; ++i) {
            bits[i] = (hash_output[0] >> i) & 1 ? 1 : -1;
            if (64 + i < bits_count)
                bits[64 + i] = (hash_output[1] >> i) & 1 ? 1 : -1;
        }
    } else {
        throw std::invalid_argument("Bits <= 128");
    }

    return bits;
}

void SimHash::IDF(const std::vector<std::vector<double>>& allFeatures) {
    const int N = allFeatures.size();
    const int dim = allFeatures[0].size();
    idfWeights.assign(dim, 0.0);
    std::vector<int> docFreq(dim, 0);

    for (const auto& img : allFeatures) {
        for (int j = 0; j < dim; ++j) {
            if (img[j] > 1e-9)  
                docFreq[j]++;
        }
    }

    for (int j = 0; j < dim; ++j) {
        idfWeights[j] = std::log(static_cast<double>(N) / (1.0 + docFreq[j]));
    }
}

size_t SimHash::hashFunction(const std::vector<double>& featureVector) {
    std::vector<double> V(bits, 0.0);
    const int dim = featureVector.size();

    for (int i = 0; i < dim; ++i) {
        double tf = featureVector[i];
        double weight = tf * idfWeights[i];
        if (std::abs(weight) < 1e-9) continue;

        std::vector<int> phi = hashify_double_by_murmur128(featureVector[i], i, bits);
        for (size_t j = 0; j < bits; ++j) {
            V[j] += phi[j] * weight;
        }
    }

    uint64_t hashValue = 0;
    for (size_t j = 0; j < bits; ++j) {
        int bit = (V[j] >= 0.0) ? 1 : 0;
        hashValue = (hashValue << 1) | bit;
    }

    return static_cast<size_t>(hashValue);
}
