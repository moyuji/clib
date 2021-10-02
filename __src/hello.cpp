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
unordered_map<int, double> umap;

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
    for(int r=0; r<n_res; r++) {
        struct item* loc;
        umap.clear();
        for(int c=0; c<num_dims; c++) {
            double weight = ptr[r*num_dims + c];
            if (fabs(weight) < EPS) {
                continue;
            }
           for(loc=heads[c]; loc < heads[c + 1]; loc ++) {
                auto fd = umap.find(loc->row_id);
                if (fd == umap.end()) {
                    umap[loc->row_id] = weight * loc->weight;
                } else {
                    fd->second += weight * loc->weight;
                }
            }
        }
        vector<int> v1;
        for (auto it=umap.begin();it!=umap.end(); ++it) {
            if(k - v1.size() <= 0) {
                if(umap[v1[0]] < it->second) {
                    // higher score than heap top
                    pop_heap(v1.begin(), v1.end(), compare_item);
                    v1.back() = it->first;
                    push_heap(v1.begin(), v1.end(), compare_item);
                }
                // lower score than heap top, skip
                continue;
            }
            // not full yet, just push
            v1.push_back(it->first);
            push_heap(v1.begin(), v1.end(), compare_item);
        }
        sort(v1.begin(), v1.end(), compare_item);
//        for(auto it=v1.begin();it != v1.end();it++) {
//            printf("%f ", umap[*it]);
//        }
//        printf("\n");
//        printf("ret length: %lu\n", v1.size());
        python_ret.push_back(v1);
    }
    return python_ret;
}
