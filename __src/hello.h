#ifndef MYMODULE_HELLO_H
#define MYMODULE_HELLO_H
#include <numpy/arrayobject.h>

// Always include Python.h in the very first line in all header files.
#include <Python.h>

PyObject* hello(PyObject *self, PyObject *args);
PyObject* load(PyObject *self, PyObject *args);
PyObject* eval(PyObject *self, PyObject *args);

#endif
