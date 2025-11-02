#include "../Header/bloom_filter.h"

BloomFilter::BloomFilter(size_t numPlanes, size_t dimension, size_t k)
    : numPlanes(numPlanes), dimension(dimension), k(k) {
    generateRandomHyperplanes();
}

void BloomFilter::generateRandomHyperplanes() {
    randomHyperplanes.resize(numPlanes, std::vector<double>(dimension));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < numPlanes; ++i) {
        double norm = 0.0;
        for (size_t j = 0; j < dimension; ++j) {
            randomHyperplanes[i][j] = dist(gen);
            norm += randomHyperplanes[i][j] * randomHyperplanes[i][j];
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < dimension; ++j) {
            randomHyperplanes[i][j] /= norm;
        }
    }
}

std::vector<size_t> BloomFilter::hashFunction(const std::vector<double>& featureVector) {
    std::vector<size_t> hashValues(k, 0);
    size_t bitsPerHash = numPlanes / k;  // chia đều số mặt phẳng cho mỗi hash

    for (size_t group = 0; group < k; ++group) {
        size_t hashValue = 0;
        size_t start = group * bitsPerHash;
        size_t end = (group + 1) * bitsPerHash;

        for (size_t i = start; i < end && i < numPlanes; ++i) {
            double dotProduct = 0.0;
            for (size_t j = 0; j < dimension; ++j) {
                dotProduct += featureVector[j] * randomHyperplanes[i][j];
            }
            if (dotProduct > 0) {
                hashValue |= (1ULL << (i - start));  // set bit trong phạm vi nhóm
            }
        }
        hashValues[group] = hashValue;
    }

    return hashValues;
}
