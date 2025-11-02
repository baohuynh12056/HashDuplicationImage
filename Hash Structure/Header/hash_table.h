#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <iostream>
#include <vector>
#include <list>
#include <random>
#include <algorithm>
using namespace std;

class HashTable
{
private:
    size_t numPlanes = 32;
    size_t dimension = 2048;
    vector<vector<double>> randomHyperplanes;
    void generateRandomHyperplanes();
    

public:
    HashTable(size_t numPlanes, size_t dimension);
    size_t hashFunction(const vector<double> &featureVector);
};

#endif // HASHTABLE_H