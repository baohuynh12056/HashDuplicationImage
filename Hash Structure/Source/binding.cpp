#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../Header/hash_table.h"
#include "../Header/min_hash.h"
#include "../Header/sim_hash.h"

namespace py = pybind11;

PYBIND11_MODULE(MyHash, m)
{
    py::class_<HashTable>(m, "HashTable")
        .def(py::init<size_t, size_t>(),
             py::arg("numPlanes"), py::arg("dimension"))
        .def("hashFunction", &HashTable::hashFunction, py::arg("featureVector"))
        .def("addItem", &HashTable::addItem, py::arg("featureVector"))
        .def("searchSimilarImages", &HashTable::searchSimilarImages, py::arg("queryFeature"), py::arg("threshold"))
        .def("getBucket", &HashTable::getBucket, py::arg("index"))
        .def("print", &HashTable::print);

    py::class_<MinHash>(m, "MinHash")
        .def(py::init<size_t, size_t>(),
             py::arg("numHashes"), py::arg("dimension"))
        .def("addItem", &MinHash::addItem, py::arg("featureVector"))
        .def("searchSimilarImages", &MinHash::searchSimilarImages, py::arg("queryFeature"), py::arg("threshold"))
        .def("getItem", &MinHash::getItem, py::arg("index"))
        .def("print", &MinHash::print);
}
