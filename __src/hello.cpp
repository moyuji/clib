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

bool compare_item(const int a,const int b) {
    return score[a] > score[b];
}

void hello() {
    printf("Hello!\n");
}

int load(py::array_t<double> array){
    py::buffer_info buf = array.request();
    num_rows = buf.shape[0];
    num_dims = buf.shape[1];
    double *ptr = (double *) buf.ptr;
    score = (double *) malloc(sizeof(double) * num_rows);
    ranked_list = (int *) malloc(sizeof(int) * num_rows);
    ranked_list_b = (int *) malloc(sizeof(int) * num_rows);

    int ret = 0;
    for (int r=0; r<num_rows; r++) {
        ranked_list[r] = r;
        ranked_list_b[r] = r;
        score[r] = 0;
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
        int counter_left_now = 0;
        int counter_right_now = 0;
        int counter_right_end = 0;
        struct item* loc;
        for(int c=0; c<num_dims; c++) {
            double weight = ptr[r*num_dims + c];
            if (fabs(weight) < EPS) {
                continue;
            }
            counter_left_now = 0;
            counter_right_now = 0;
            for(loc=heads[c]; loc < heads[c + 1]; loc ++) {
                while(counter_right_now < counter_right_end &&
                    ranked_list_b[counter_right_now] < loc->row_id) {
                    ranked_list[counter_left_now++] = ranked_list_b[counter_right_now++];
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
        }
        sort(ranked_list_b, ranked_list_b + min(k,counter_right_end), compare_item);
        vector<int> python_val;
        for (int i = 0; i < k; ++i) {
            python_val.push_back(ranked_list_b[i]);
        }
        python_ret.push_back(python_val);
    }
    return python_ret;
}
