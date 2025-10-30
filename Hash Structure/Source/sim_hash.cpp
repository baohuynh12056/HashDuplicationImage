#include "sim_hash.h"

simHash::simHash (std::vector<double>& data, int bit, int weight) : hashFunc (data, bit) {}

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

bool simHash::hash () {
    if (data.empty()) return false;
    std::vector<double> W(bit, 0);
    for (size_t i = 0; i < (int) data.size(); ++i) {
        double weight = std::abs(data[i]); 
        std::vector<int> phi = simpleHash(data[i]);

        for (int j = 0; j < bit; ++j) {
            W[j] += (phi[j] == 1 ? weight : -weight);
        }
    }

    std::vector<int> B(bit);
    for (int j = 0; j < bit; ++j) {
        B[j] = W[j] >= 0 ? 1 : 0;
    }

    this->hashValue = B;
    return true;
}


std::vector<double> simHash::computeWeights(std::vector<double>& u) {
    std::vector<double> weights(u.size());
    for (size_t i = 0; i < u.size(); ++i) {
        weights[i] = std::abs(u[i]); 
    }
    return weights;
}
