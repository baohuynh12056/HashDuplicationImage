#include "../Header/hash_table.h"

HashTable::HashTable(size_t numPlanes, size_t dimension) : numPlanes(numPlanes), dimension(dimension)
{
    generateRandomHyperplanes();
}

void HashTable::generateRandomHyperplanes()
{
    randomHyperplanes.resize(numPlanes, vector<double>(dimension));

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
        for (size_t j = 0; j < dimension; ++j)
            randomHyperplanes[i][j] /= norm;
    }
}

size_t HashTable::hashFunction(const vector<double> &featureVector)
{
    size_t hashValue = 0;
    for (size_t i = 0; i < numPlanes; ++i)
    {
        double dotProduct = 0;
        for (size_t j = 0; j < dimension; ++j)
        {
            dotProduct += featureVector[j] * randomHyperplanes[i][j];
        }
        if (dotProduct > 0)
        {
            hashValue |= (1 << i);
        }
    }
    return hashValue;
}