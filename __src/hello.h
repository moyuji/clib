//#ifndef MYMODULE_HELLO_H
//#define MYMODULE_HELLO_H
//#include <numpy/arrayobject.h>
//
//// Always include Python.h in the very first line in all header files.
//#include <Python.h>
//
//PyObject* hello(PyObject *self, PyObject *args);
//PyObject* load(PyObject *self, PyObject *args);
//PyObject* eval(PyObject *self, PyObject *args);
//
//#endif
#ifndef MYPYBINDMODULE_HELLO_H
#define MYPYBINDMODULE_HELLO_H

// Always include <pybind11/pybind11.h> in the very first line of all header files and module.cpp
// In general, make sure that you include pybind11.h BEFORE all other header files.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

#include <string>


// Our hello world function definition
void hello();
vector<vector<int>> eval(py::array_t<double>, int);
int load(py::array_t<double>);

#endif
