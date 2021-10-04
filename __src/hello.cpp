// In all C files, include its corresponding header file in the very first line.
// No need to include <Python.h> as we did that already in the header file.
// Just make sure that <Python.h> is included BEFORE any other header file.
#include "hello.h"
#define EPS 0.000001

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
double* umap;

bool compare_item(const int a,const int b) {
    return umap[a] > umap[b];
}

void hello() {
    printf("Hello!\n");
}

int load(py::array_t<double> array){
    py::buffer_info buf = array.request();
    num_rows = buf.shape[0];
    num_dims = buf.shape[1];
    double *ptr = (double *) buf.ptr;

    int ret = 0;
    for (int r=0; r<num_rows; r++) {
         for(int c=0; c<num_dims; c++) {
            if (fabs(ptr[r*num_dims + c] ) > EPS) {
                ret += 1;
            }
        }
    }
    matrix = (struct item*) malloc(sizeof(struct item) * ret);
    heads = (struct item**) malloc(sizeof(struct item*) * (num_dims + 1));
    struct item* loc = matrix;
    for(int c=0; c<num_dims; c++) {
        heads[c] = loc;
        for (int r=0; r<num_rows; r++) {
            double weight = ptr[r*num_dims + c];
            if (fabs(weight) > EPS) {
                loc->weight = weight;
                loc->row_id = r;
                loc++;
            }
        }
    }
    heads[num_dims] = loc;
    printf("Loaded!\n");
    return ret;
}

vector<vector<int>> eval(py::array_t<double> array, int k) {
    py::buffer_info buf = array.request();
    double *ptr = (double *) buf.ptr;
    int n_res =buf.shape[0];
    vector<vector<int>> python_ret;
    umap = (double *)malloc(sizeof(double) * num_rows);
    if (num_rows < k) {
        k = num_rows;
    }
    int * hp = (int *)malloc(sizeof(int) * k);
    for(int r=0; r<n_res; r++) {
        struct item* loc;
        memset(umap, 0, sizeof(double) * num_rows);
        for(int c=0; c<num_dims; c++) {
            double weight = ptr[r*num_dims + c];
            if (fabs(weight) < EPS) {
                continue;
            }
           for(loc=heads[c]; loc < heads[c + 1]; loc ++) {
                umap[loc->row_id] += weight * loc->weight;
            }
        }

        for (int i =0; i<k; i++) {
            hp[i] = i;
        }
        make_heap(hp, hp + k, compare_item);
        double tp = umap[hp[0]];
        for (int i = k; i<num_rows; i++) {
            if(tp < umap[i]) {
                // higher score than heap top
                pop_heap(hp, hp + k, compare_item);
                hp[k - 1] = i;
                push_heap(hp, hp + k, compare_item);
                tp = umap[hp[0]];
            }
        }
        sort(hp, hp + k, compare_item);
        vector<int> v1(k);
        for(int i=0; i <k; i++) {
            v1[i]=hp[i];
        }
        python_ret.push_back(v1);
    }
    return python_ret;
}
