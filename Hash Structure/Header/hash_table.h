#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <iostream>
#include <vector>
#include <list>
#include <random>
#include <unordered_map>

using namespace std;

class HashTable
{
private:
    size_t numPlanes;
    size_t dimension;
    unordered_map<size_t, std::list<std::vector<double>>> table;
    vector<vector<double>> randomHyperplanes;
    void generateRandomHyperplanes();

public:
    HashTable(size_t numPlanes, size_t dimension);
    size_t hashFunction(const vector<double> &featureVector);
    void addItem(const vector<double> &featureVector);
    vector<pair<int, double>> HashTable::searchSimilarImages(const vector<double> &queryFeature, double threshold);
    list<vector<double>> getBucket(int index);
    void print() const;
};

#endif // HASHTABLE_H