#ifndef MIN_HASH_H
#define MIN_HASH_H

#include <vector>
#include <set>
#include <random>
#include <iostream>
#include <limits>
#include <algorithm>

using namespace std;

class MinHash {
private:
    size_t numHashes;    
    size_t dimension;  
    
    static constexpr long long prime = 2147483647; // 2^31 - -1
    
    vector<long long> a;
    vector<long long> b;
    
    struct Item {
        vector<double> featureVector;
        vector<unsigned int> signature;
    };
    vector<Item> items;
    
    void generateCoefficients();
    
    unsigned int hash(int hashIndex, int element) const;

public:
    MinHash(size_t numHashes, size_t dimension);
    

    set<int> convertToSet(const vector<double>& featureVector);
    

    vector<unsigned int> computeSignature(const set<int>& elementSet) const;
    

    void addItem(const vector<double>& featureVector);
    

    double estimateJaccardFromSignatures(const vector<unsigned int>& sig1, 
                                         const vector<unsigned int>& sig2) const;
    

    static double JaccardSimilarity(const set<int>& A, const set<int>& B);
    

    vector<pair<int, double>> searchSimilarImages(const vector<double>& queryFeature, 
                                            double threshold);
    

    size_t estimateCardinality(const vector<unsigned int>& signature, 
                               size_t k, size_t range) const;

    void print() const;

    
    const Item& getItem(size_t index) const { 
        return items[index]; 
    }
};

#endif // MIN_HASH_H