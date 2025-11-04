#include "min_hash.h"

MinHash::MinHash(size_t numPlanes, size_t dimension)
    : numPlanes(numPlanes),
      dimension(dimension),
      mean(dimension, 0.0),
      median(dimension, 0.0),
      stddev(dimension, 1.0)
{
    randomHyperplanes.resize(numPlanes, std::vector<double>(dimension));
    generateRandomHyperplanes();
}

void MinHash::generateRandomHyperplanes()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < numPlanes; ++i)
    {
        double norm = 0.0;
        for (size_t j = 0; j < dimension; ++j)
        {
            randomHyperplanes[i][j] = dist(gen);
            norm += randomHyperplanes[i][j] * randomHyperplanes[i][j];
        }
        norm = std::sqrt(norm);
        if (norm == 0.0)
            norm = 1e-12;
        for (size_t j = 0; j < dimension; ++j)
            randomHyperplanes[i][j] /= norm;
    }
}

std::vector<std::vector<uint8_t>>
MinHash::computeSignatures(const std::vector<std::vector<double>> &matrix,
                           bool useMedianThreshold)
{
    size_t n = matrix.size();
    if (n == 0)
        return {};

    std::vector<std::vector<double>> v(n, std::vector<double>(dimension));

    for (size_t i = 0; i < n; ++i)
    {
        const auto &row = matrix[i];
        for (size_t j = 0; j < dimension; ++j)
        {
            double minval = std::numeric_limits<double>::infinity();
            for (size_t p = 0; p < numPlanes; ++p)
            {
                double prod = row[j] * randomHyperplanes[p][j];
                if (prod < minval)
                    minval = prod;
            }
            v[i][j] = minval;
        }
    }

    std::vector<double> tmp;
    tmp.reserve(n);

    for (size_t j = 0; j < dimension; ++j)
    {
        tmp.clear();
        for (size_t i = 0; i < n; ++i)
            tmp.push_back(v[i][j]);

        // mean
        double s = 0.0;
        for (double x : tmp)
            s += x;
        mean[j] = s / static_cast<double>(n);

        // median
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        double med = tmp[tmp.size() / 2];
        if (tmp.size() % 2 == 0)
        {
            double other = *std::max_element(tmp.begin(), tmp.begin() + tmp.size() / 2);
            med = 0.5 * (med + other);
        }
        median[j] = med;

        // stddev
        double var = 0.0;
        for (double x : tmp)
        {
            double d = x - mean[j];
            var += d * d;
        }
        var /= static_cast<double>(n);
        stddev[j] = std::sqrt(var);
        if (stddev[j] < 1e-12)
            stddev[j] = 1e-12;
    }

    std::vector<std::vector<uint8_t>> signatures(n, std::vector<uint8_t>(dimension, 0));

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < dimension; ++j)
        {
            bool bit;
            if (useMedianThreshold)
                bit = (v[i][j] >= median[j]);
            else
            {
                double z = (v[i][j] - mean[j]) / stddev[j];
                bit = (z > 0.0);
            }
            signatures[i][j] = static_cast<uint8_t>(bit);
        }
    }

    return signatures;
}

std::vector<uint8_t>
MinHash::hashFunction(const std::vector<double> &vec,
                      bool useMedianThreshold) const
{
    if (median.empty() || mean.empty() || stddev.empty())
        throw std::runtime_error("Must call computeSignatures() before hashFunction()");

    std::vector<double> minVals(dimension);
    for (size_t j = 0; j < dimension; ++j)
    {
        double minval = std::numeric_limits<double>::infinity();
        for (size_t p = 0; p < numPlanes; ++p)
        {
            double prod = vec[j] * randomHyperplanes[p][j];
            if (prod < minval)
                minval = prod;
        }
        minVals[j] = minval;
    }

    std::vector<uint8_t> bits(dimension, 0);
    for (size_t j = 0; j < dimension; ++j)
    {
        bool bit;
        if (useMedianThreshold)
            bit = (minVals[j] >= median[j]);
        else
        {
            double z = (minVals[j] - mean[j]) / stddev[j];
            bit = (z > 0.0);
        }
        bits[j] = static_cast<uint8_t>(bit);
    }

    return bits;
}

// Getters
const std::vector<std::vector<double>> &MinHash::getHyperplanes() const
{
    return randomHyperplanes;
}
const std::vector<double> &MinHash::getMean() const { return mean; }
const std::vector<double> &MinHash::getMedian() const { return median; }
const std::vector<double> &MinHash::getStddev() const { return stddev; }