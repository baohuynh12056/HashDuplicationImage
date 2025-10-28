#ifndef SIM_HASH_H
#define SIM_HASH_H
#include "hash_func.h"
#include <iostream>
#include <vector>
#include <list>
#include <cstring>
#include <random>
#include <algorithm>
class simHash {
    private:
        std::vector<int> hashValue;
        size_t bits;
    public:
        simHash (size_t bit);
        ~simHash ();
        std::vector<int> simpleHash (double x);
        int distance (std::vector<double>& other) ;
        size_t hashFunction(const std::vector<double> &featureVector);
        std::string extractBinary(long long value);
        std::vector<double> computeWeights (std::vector<double>&);
};

#endif