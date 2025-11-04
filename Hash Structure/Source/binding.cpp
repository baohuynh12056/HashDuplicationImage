#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../Header/hash_table.h"
#include "../Header/bloom_filter.h"
#include "../Header/min_hash.h"
#include "../Header/sim_hash.h" 

namespace py = pybind11;

PYBIND11_MODULE(MyHash, m)
{
    py::class_<HashTable>(m, "HashTable")
        .def(py::init<size_t, size_t>(), py::arg("numPlanes") = 32, py::arg("dimension") = 2048)
        .def("hashFunction", &HashTable::hashFunction, py::arg("featureVector"));

    py::class_<BloomFilter>(m, "BloomFilter")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("numPlanes") = 64,
             py::arg("dimension") = 2048,
             py::arg("k") = 4)
        .def("hashFunction", &BloomFilter::hashFunction,
             py::arg("featureVector"));

    py::class_<MinHash>(m, "MinHash")
        .def(py::init<size_t,size_t>(), py::arg("num_planes") = 128,py::arg("dimension") = 2048)

        .def("computeSignatures", &MinHash::computeSignatures,
             py::arg("matrix"), py::arg("useMedianThreshold") = false)

        .def("hashFunction", &MinHash::hashFunction,
             py::arg("vec"), py::arg("useMedianThreshold") = false);

    py::class_<SimHash>(m, "SimHash")
        .def(py::init<size_t>(), py::arg("bit") = HASH_BITS)
        .def("IDF", &SimHash::IDF,
             py::arg("allFeatures"))
        .def("hashFunction", &SimHash::hashFunction,
             py::arg("featureVector"));
}