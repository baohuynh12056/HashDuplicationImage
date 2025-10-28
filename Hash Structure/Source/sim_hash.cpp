#include "sim_hash.h"

simHash::simHash (size_t bit) {
    bits = bit;
}

simHash::~simHash () {}

int simHash::distance (std::vector<double>& other) {
    if (hashValue.size() != other.size()) {
        throw std::invalid_argument("Vectors must be the same length");
    }

    int distance = 0;
    const int len = other.size();
    for (int i = 0; i < len; ++i) {
        if (other[i] != hashValue[i]) {
            ++distance;
        }
    }
    return distance;
}
std::vector<int> simHash::simpleHash(double x) {    
    uint64_t raw;
    std::memcpy(&raw, &x, sizeof(double));
    uint32_t folded = static_cast<uint32_t>((raw >> 32) ^ raw);
    std::bitset<32> b(folded);
    std::vector<int> bits(simHash::bits, 0);
    for (size_t i = 0; i < simHash::bits; ++i)
        bits[i] = b[31 - (i % 32)];

    return bits;
}
size_t simHash::hashFunction(const std::vector<double>& featureVector) {
    std::vector<double> W(bits, 0);
    for (size_t i = 0; i < 2048; ++i) {
        double weight = std::abs(featureVector[i]); 
        std::vector<int> phi = simpleHash(featureVector[i]);
        for (size_t j = 0; j < bits; ++j) {
            W[j] += (phi[j] == 1 ? weight : -weight);
        }
    }

    std::vector<int> B(bits);
    for (size_t j = 0; j < bits; ++j) {
        B[j] = W[j] >= 0 ? 1 : 0;
    }
    uint64_t hash_value_as_int = 0;
    for (int bit : B) {
        hash_value_as_int = (hash_value_as_int << 1) | bit;
    }    
    size_t hash_as_size_t = hash_value_as_int;
    return hash_as_size_t;
}


std::vector<double> simHash::computeWeights(std::vector<double>& u) {
    std::vector<double> weights(u.size());
    for (size_t i = 0; i < u.size(); ++i) {
        weights[i] = std::abs(u[i]); 
    }
    return weights;
}
