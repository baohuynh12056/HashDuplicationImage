#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "min_hash.h"

namespace py = pybind11;

PYBIND11_MODULE(minhash_py, m) {
    py::class_<MinHash>(m, "MinHash")
        .def(py::init<size_t,size_t>(), py::arg("num_planes") = 128,py::arg("dimension") = 2048)

        .def("computeSignatures", &MinHash::computeSignatures,
             py::arg("matrix"), py::arg("useMedianThreshold") = false)

        .def("hashFunction", &MinHash::hashFunction,
             py::arg("vec"), py::arg("useMedianThreshold") = false);

}
