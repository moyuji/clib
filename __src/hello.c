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
int* ranked_list_b;

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
    printf("Hello!\n");
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
    ranked_list_b = malloc(sizeof(int) * num_rows);
    int ret = 0;
    for (int r=0; r<num_rows; r++) {
        ranked_list[r] = r;
        score[r] = 0;
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
    printf("Loaded!\n");
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
    int perf = 0;
    int n_res = array->dimensions[0];
    PyObject* python_ret = PyList_New(n_res);
    for(int r=0; r<n_res; r++) {
        int counter_left_now = 0;
        int counter_right_now = 0;
        int counter_right_end = 0;
        struct item* loc;
        for(int c=0; c<num_dims; c++) {
            double weight = LOOKUP(array, r, c);
            if (fabs(weight) < EPS) {
                continue;
            }
            counter_left_now = 0;
            counter_right_now = 0;
    //        printf("c=%d\n", c);
            for(loc=heads[c]; loc < heads[c + 1]; loc ++) {
    //            printf("counter_right_now=%d counter_left_now=%d weight=%f\n", counter_right_now, counter_left_now, weight);
                perf+=1;
                while(counter_right_now < counter_right_end &&
                    ranked_list_b[counter_right_now] < loc->row_id) {
                    ranked_list[counter_left_now++] = ranked_list_b[counter_right_now++];
                    perf+=1;
                }
                if (counter_right_now >= counter_right_end ||
                    ranked_list_b[counter_right_now] > loc->row_id) {
                    score[loc->row_id] = 0.0;
                    ranked_list[counter_left_now++] = loc->row_id;
                } else {
                    ranked_list[counter_left_now++] = ranked_list_b[counter_right_now++];
                }
                score[loc->row_id] += weight * loc->weight;
            }
            int * swp = ranked_list_b;
            ranked_list_b = ranked_list;
            ranked_list = swp;
            counter_right_end = counter_left_now;
//            for (int i = 0; i < counter_right_end; i++) {
//                printf("%d:%f ", ranked_list_b[i], score[ranked_list_b[i]]);
//            }
//            printf("\n");
        }
        qsort(ranked_list_b, counter_left_now, sizeof(int), compare_item);
        PyObject* python_val = PyList_New(k);
        for (int i = 0; i < k; ++i) {
            PyList_SetItem(python_val, i, Py_BuildValue("i", ranked_list_b[i]));
        }
        PyList_SetItem(python_ret, r, Py_BuildValue("O", python_val));
    }
    printf("perf=%d\n", perf);
    return python_ret;
}
