#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../Header/sim_hash.h"   // include header của class simHash

namespace py = pybind11;

PYBIND11_MODULE(simhash_py, m)
{
    py::class_<SimHash>(m, "SimHash")
        .def(py::init<size_t>(), py::arg("bit") = HASH_BITS)
        .def("IDF", &SimHash::IDF,
             py::arg("allFeatures"))
        .def("hashFunction", &SimHash::hashFunction,
             py::arg("featureVector"));
}
