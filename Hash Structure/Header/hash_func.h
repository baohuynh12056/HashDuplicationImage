#ifndef HASH_FUNC_H
#define HASH_FUNC_H
#include <iostream>
#include <vector>
#include <random>
#include <functional>

class hashFunc {
    protected:
        std::vector<double> data;
        int bit;
    public:
        hashFunc (std::vector<double>& data, int bit) : data (data), bit (bit) {}
        ~hashFunc () {} 
        virtual int distance (std::vector<double>& other)=0;
        virtual bool hash () = 0;
        std::vector<int> simpleHash (double);

};

#endif