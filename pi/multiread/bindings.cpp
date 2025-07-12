#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ADXL.hpp"

namespace py = pybind11;

PYBIND11_MODULE(adxl, m) {
    py::class_<ADXL345>(m, "ADXL345")
        .def(py::init<int,bool,int>())
        .def("startup", &ADXL345::startup)
        .def("get_data", &ADXL345::get_data)
        .def("register_callback",&ADXL345::register_callback)
        .def("close", &ADXL345::close)
        .def("__del__", &ADXL345::close)
        .def_readonly("x", &ADXL345::x)
        .def_readonly("y", &ADXL345::y)
        .def_readonly("z", &ADXL345::z);
}