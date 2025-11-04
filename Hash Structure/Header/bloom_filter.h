#ifndef BLOOM_FILTER_H
#define BLOOM_FILTER_H

#include <vector>
#include <random>
#include <cmath>
#include <cstddef>

class BloomFilter
{
private:
    size_t numPlanes; // Tổng số hyperplanes
    size_t dimension; // Số chiều của feature vector
    size_t k;         // Số hash values (Bloom filter size)
    std::vector<std::vector<double>> randomHyperplanes;

    void generateRandomHyperplanes(); // Sinh ngẫu nhiên hyperplanes

public:
    BloomFilter(size_t numPlanes, size_t dimension, size_t k);

    // Sinh ra k hashValue, mỗi cái là 1 nhóm bit
    std::vector<size_t> hashFunction(const std::vector<double> &featureVector);
};

#endif // BLOOM_FILTER_H