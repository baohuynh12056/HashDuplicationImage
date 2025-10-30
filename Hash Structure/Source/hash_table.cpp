#include "../Header/hash_table.h"

HashTable::HashTable(size_t numPlanes, size_t dimension) : numPlanes(numPlanes), dimension(dimension)
{
    generateRandomHyperplanes();
}

void HashTable::generateRandomHyperplanes()
{
    randomHyperplanes.resize(numPlanes, vector<double>(dimension));
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < numPlanes; ++i)
    {
        for (size_t j = 0; j < dimension; ++j)
        {
            randomHyperplanes[i][j] = dist(gen);
        }
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
            hashValue |= (1ULL << i);
        }
    }
    return hashValue;
}

void HashTable::addItem(const vector<double> &featureVector)
{
    size_t hashValue = hashFunction(featureVector);
    table[hashValue].push_back(featureVector);
}

vector<pair<int, double>> HashTable::searchSimilarImages(
    const vector<double> &queryFeature, double threshold)
{
    size_t hashValue = hashFunction(queryFeature);

    if (table.find(hashValue) == table.end())
    {
        return {};
    }

    const auto &bucket = table.at(hashValue);

    vector<pair<int, double>> results;
    int idx = 0;

    for (const auto &featureVector : bucket)
    {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (size_t i = 0; i < dimension; ++i)
        {
            dot += queryFeature[i] * featureVector[i];
            normA += queryFeature[i] * queryFeature[i];
            normB += featureVector[i] * featureVector[i];
        }
        double similarity = dot / (sqrt(normA) * sqrt(normB) + 1e-9);

        if (similarity >= threshold)
        {
            results.push_back({idx, similarity});
        }
        ++idx;
    }

    sort(results.begin(), results.end(),
         [](const pair<int, double> &a, const pair<int, double> &b)
         {
             return a.second > b.second;
         });

    return results;
}

list<vector<double>> HashTable::getBucket(int index)
{
    return table[index];
}

void HashTable::print() const
{
    cout << "---HashTable---" << endl;
    cout << "Total buckets (keys) created: " << table.size() << endl;

    // Lặp qua một map (key, value)
    for (const auto &pair : table)
    {
        size_t bucket_key = pair.first;
        const auto &bucket_content = pair.second;
        cout << "Bucket " << bucket_key << " contains "
             << bucket_content.size() << " images." << endl;
    }
}
