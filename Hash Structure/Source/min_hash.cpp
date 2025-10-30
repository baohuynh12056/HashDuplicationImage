#include "../Header/min_hash.h"

void MinHash::generateCoefficients(){
    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<long long> distA(1, prime - 1);
    uniform_int_distribution<long long> distB(0, prime - 1);

    a.reserve(numHashes);
    b.reserve(numHashes);

    for (size_t i = 0; i < numHashes; ++i) {
        a.push_back(distA(gen));
        b.push_back(distB(gen));
    }
}

MinHash::MinHash(size_t numHashes, size_t dimension) 
    : numHashes(numHashes), dimension(dimension) {
    generateCoefficients();
}

set<int> MinHash::convertToSet(const vector<double>& featureVector){
    set<int> result;
    double mean = 0;
    for (double val : featureVector) {
        mean += val;
    }
    mean /= featureVector.size();
    
    for (size_t i = 0; i < featureVector.size(); ++i) {
        if (featureVector[i] > mean) {
            result.insert(static_cast<int>(i));
        }
    }
    
    if (result.empty()) {
        result.insert(0);
    }
    
    return result;
}

unsigned int MinHash::hash(int hashIndex, int element) const {
    long long h = (a[hashIndex] * element + b[hashIndex]) % prime;
    return static_cast<unsigned int>(h);
}

vector<unsigned int> MinHash::computeSignature(const set<int>& elementSet) const {
    vector<unsigned int> signature(numHashes);

    for (size_t i = 0; i < numHashes; ++i) {
        unsigned int minHash = numeric_limits<unsigned int>::max();
        

        for (int element : elementSet) {
            unsigned int hashVal = hash(i, element);
            minHash = min(minHash, hashVal);
        }
        
        signature[i] = minHash;
    }
    
    return signature;
}

void MinHash::addItem(const vector<double>& featureVector) {
    set<int> elementSet = convertToSet(featureVector);
    vector<unsigned int> signature = computeSignature(elementSet);
    
    items.push_back({featureVector, signature});
}

double MinHash::estimateJaccardFromSignatures(const vector<unsigned int>& sig1, 
                                               const vector<unsigned int>& sig2) const {
    if (sig1.size() != sig2.size() || sig1.empty()) {
        return 0.0;
    }
    
    int matches = 0;
    for (size_t i = 0; i < sig1.size(); ++i) {
        if (sig1[i] == sig2[i]) {
            ++matches;
        }
    }
    
    return static_cast<double>(matches) / sig1.size();
}

double MinHash::JaccardSimilarity(const set<int>& A, const set<int>& B) {
    if (A.empty() && B.empty()) return 1.0;
    if (A.empty() || B.empty()) return 0.0;
    
    vector<int> intersection;
    set_intersection(A.begin(), A.end(), B.begin(), B.end(), 
                     back_inserter(intersection));
    
    vector<int> unionSet;
    set_union(A.begin(), A.end(), B.begin(), B.end(), 
              back_inserter(unionSet));
    
    return static_cast<double>(intersection.size()) / unionSet.size();
}

vector<pair<int, double>> MinHash::searchSimilarImages(const vector<double>& queryFeature, 
                                                  double threshold) {
    set<int> querySet = convertToSet(queryFeature);
    vector<unsigned int> querySignature = computeSignature(querySet);
    
    vector<pair<int, double>> results;
    
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        double estimatedSim = estimateJaccardFromSignatures(querySignature, item.signature);
        
        if (estimatedSim >= threshold * 0.8) {
            set<int> itemSet = convertToSet(item.featureVector);
            double exactSim = JaccardSimilarity(querySet, itemSet);
            
            if (exactSim >= threshold) {
                results.push_back({i, exactSim});
            }
        }
    }

    sort(results.begin(), results.end(),
         [](const pair<int, double>& a, const pair<int, double>& b) {
             return a.second > b.second;
         });
    
    return results;
}


size_t MinHash::estimateCardinality(const vector<unsigned int>& signature, 
                                    size_t k, size_t range) const {
    if (k >= signature.size() || signature[k] == 0) {
        return 0;
    }
    

    vector<unsigned int> sortedSig = signature;
    sort(sortedSig.begin(), sortedSig.end());
    
    unsigned int kthMin = sortedSig[k];
    

    double normalized = static_cast<double>(kthMin) / range;
    

    if (normalized > 0) {
        return static_cast<size_t>(k / normalized - 1);
    }
    
    return 0;
}

void MinHash::print() const {
    if (items.size() >= 2) {
        std::cout << "Similarity estimates:" << std::endl;
        for (size_t i = 0; i < items.size() - 1; ++i) {
            for (size_t j = i + 1; j < items.size(); ++j) {
                double sim = estimateJaccardFromSignatures(
                    items[i].signature, items[j].signature);
                std::cout << "Item " << i << " vs Item " << j << ": " << sim << std::endl;
            }
        }
    }
}

