#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../Header/bloom_filter.h"

namespace py = pybind11;

PYBIND11_MODULE(bloom_filter_py, m)
{
    py::class_<BloomFilter>(m, "BloomFilter")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("numPlanes") = 64,
             py::arg("dimension") = 2048,
             py::arg("k") = 4)
        .def("hashFunction", &BloomFilter::hashFunction,
             py::arg("featureVector"));
}
