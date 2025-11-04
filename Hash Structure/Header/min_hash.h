#ifndef MINHASH_H
#define MINHASH_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cstdint>

class MinHash
{
public:
    explicit MinHash(size_t numPlanes = 128, size_t dimension = 2048);

    // Sinh hyperplanes ngẫu nhiên
    void generateRandomHyperplanes();

    // Tính signature cho toàn bộ dataset (và lưu mean, median, stddev)
    std::vector<std::vector<uint8_t>> computeSignatures(
        const std::vector<std::vector<double>> &matrix,
        bool useMedianThreshold = false);

    std::vector<uint8_t> hashFunction(
        const std::vector<double> &vec,
        bool useMedianThreshold = false) const;

    const std::vector<std::vector<double>> &getHyperplanes() const;
    const std::vector<double> &getMean() const;
    const std::vector<double> &getMedian() const;
    const std::vector<double> &getStddev() const;

private:
    size_t numPlanes;
    size_t dimension;
    std::vector<std::vector<double>> randomHyperplanes; // [m][2048]
    std::vector<double> mean;
    std::vector<double> median;
    std::vector<double> stddev;
};

#endif // MINHASH_H