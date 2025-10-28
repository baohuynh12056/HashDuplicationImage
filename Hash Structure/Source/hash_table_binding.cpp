#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../Header/hash_table.h"

namespace py = pybind11;

PYBIND11_MODULE(hash_table_py, m)
{
    py::class_<HashTable>(m, "HashTable")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("bucket"), py::arg("numPlanes"), py::arg("dimension"))
        .def("addItem", &HashTable::addItem, py::arg("featureVector"))
        .def("search", &HashTable::search, py::arg("featureVector"))
        .def("hashFunction", &HashTable::hashFunction, py::arg("featureVector"))
        .def("print", &HashTable::print);
}