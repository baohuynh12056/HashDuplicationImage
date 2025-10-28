#ifndef BLOOMFILTER_H
#define BLOOMFILTER_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <opencv2/opencv.hpp>
using namespace std;

class BloomFilter {
private:
    size_t n;        // số ảnh dự kiến (n)
    double p;    // tỉ lệ false positive mong muốn (p)
    int k;               // số hàm băm (k)
    size_t m;              // số bit của mảng (m)
    vector<bool> bitArray;  // mảng bit lưu trữ

public:
    BloomFilter(size_t n, double fPr);

    void insert(const string &item);
    bool contains(const string &item) const;

};

// -------------------- pHash (Perceptual Hash) --------------------
string pHash(const string &imagePath);

// -------------------- Hamming Distance --------------------
int hammingDistance(const string &hash1, const string &hash2);

#endif // BLOOMFILTER_H