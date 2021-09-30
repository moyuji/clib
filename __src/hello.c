// In all C files, include its corresponding header file in the very first line.
// No need to include <Python.h> as we did that already in the header file.
// Just make sure that <Python.h> is included BEFORE any other header file.
#include "hello.h"
#define EPS 0.000001
#define LOOKUP(A, R, C)  *(double *)(A->data + R*A->strides[0] + C*A->strides[1])

int num_dims;
int num_rows;
struct item {
    int row_id;
    double weight;
};

struct item* matrix;
struct item** heads;
double* score;
int* ranked_list;

int compare_item(const void *a,const void *b) {
    double l = score[*(int *)a];
    double r = score[*(int *)b];
    if (r - l > 0) {
        return 1;
    }
    if (l - r > 0) {
        return -1;
    }
    return 0;
}

PyObject* hello(PyObject *self, PyObject *args) {
    Py_RETURN_NONE;
}

PyObject* load(PyObject *self, PyObject *args) {
    PyObject *input;
    PyArrayObject *array;
    if (!PyArg_ParseTuple(args, "O", &input)) {
        return NULL;
    }
    array = (PyArrayObject *) input;

    if (array == NULL) {
        return NULL;
    }
    num_rows = array->dimensions[0];
    num_dims = array->dimensions[1];
    score = malloc(sizeof(double) * num_rows);
    ranked_list = malloc(sizeof(int) * num_rows);
    int ret = 0;
    for (int r=0; r<num_rows; r++) {
        ranked_list[r] = r;
        for(int c=0; c<num_dims; c++) {
            if (fabs(LOOKUP(array, r, c)) > EPS) {
                ret += 1;
            }
        }
    }
    matrix = malloc(sizeof(struct item) * ret);
    heads = malloc(sizeof(struct item*) * (num_dims + 1));
    struct item* loc = matrix;
    for(int c=0; c<num_dims; c++) {
        heads[c] = loc;
        for (int r=0; r<num_rows; r++) {
            double weight = LOOKUP(array, r, c);
            if (fabs(weight) > EPS) {
                loc->weight = weight;
                loc->row_id = r;
                loc++;
            }
        }
    }
    heads[num_dims] = loc;
    return PyLong_FromLong(ret);
}


PyObject* eval(PyObject *self, PyObject *args) {
    PyObject *input;
    PyArrayObject *array;
    int k;
    if (!PyArg_ParseTuple(args, "Oi", &input, &k)) {
        return NULL;
    }
    array = (PyArrayObject *) input;

    if (array == NULL) {
        return NULL;
    }
    if (num_dims != array->dimensions[1]) {
        PyErr_SetString(PyExc_ValueError, "mismatched dimension");
        return NULL;
    }
    memset(score, 0, sizeof(double) * num_rows);
    struct item* loc;
    for(int c=0; c<num_dims; c++) {
        double weight = LOOKUP(array, 0, c);
        if (fabs(weight) < EPS) {
            continue;
        }
        for(loc=heads[c]; loc < heads[c + 1]; loc ++) {
            score[loc->row_id] += weight * loc->weight;
        }
    }
    qsort(ranked_list, num_rows, sizeof(int), compare_item);
    PyObject* python_val = PyList_New(k);
    PyObject* python_score = PyList_New(k);
    for (int i = 0; i < k; ++i) {
        PyList_SetItem(python_val, i, Py_BuildValue("i", ranked_list[i]));
        PyList_SetItem(python_score, i, Py_BuildValue("d", score[ranked_list[i]]));
    }
    return PyTuple_Pack(2, python_val, python_score);
}
