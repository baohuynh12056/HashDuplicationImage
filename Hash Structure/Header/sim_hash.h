#ifndef SIM_HASH_H
#define SIM_HASH_H
#include "hash_func.h"
class simHash : public hashFunc {
    private:
        std::vector<int> hashValue;
    public:
        simHash (std::vector<double>& data, int bit, int seed);
        ~simHash ();
        int distance (std::vector<double>& other) override;
        bool hash () override;
        std::string extractBinary(long long value);
        std::vector<double> computeWeights (std::vector<double>&);
};

#endif