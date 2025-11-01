#include "../Header/hash_table.h"

HashTable::HashTable(size_t bucket, size_t numPlanes,size_t dimension) : bucket(bucket), numPlanes(numPlanes), dimension(dimension) {
    table.resize(bucket);
    generateRandomHyperplanes();
}

void HashTable::generateRandomHyperplanes() {
        randomHyperplanes.resize(numPlanes, vector<double>(dimension));
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-1000, 1000);

        for (size_t i = 0; i < numPlanes; ++i) {
            for (size_t j = 0; j < dimension; ++j) {
                randomHyperplanes[i][j] = dist(gen); 
            }
        }
    }

size_t HashTable::hashFunction(const vector<double>& featureVector) {
        size_t hashValue = 0;
        for (size_t i = 0; i < numPlanes; ++i) {
            double dotProduct = 0;
            for (size_t j = 0; j < dimension; ++j) {
                dotProduct += featureVector[j] * randomHyperplanes[i][j]; 
            }
            if (dotProduct > 0) {
                hashValue |= (1 << i);
            }
        }
        return hashValue % bucket;
    }

void HashTable::addItem(const vector<double>& featureVector) {
        size_t hashValue = hashFunction(featureVector);
        table[hashValue].push_back(featureVector);
    }

list<vector<double>> HashTable::search(const vector<double>& featureVector) {
        size_t hashValue = hashFunction(featureVector);
        return table[hashValue];  
    }

void HashTable::print() {
    for (size_t i = 0; i < bucket; ++i) {
        cout << "Bucket " << i << " contains " << table[i].size() << " images." << endl;
    }
}
