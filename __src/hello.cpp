// In all C files, include its corresponding header file in the very first line.
// No need to include <Python.h> as we did that already in the header file.
// Just make sure that <Python.h> is included BEFORE any other header file.
#include "hello.h"
#define EPS 0.000001

int num_dims;
int num_rows;
int n_res;
struct item {
    int row_id;
    double weight;
};

struct item* matrix;
struct item** heads;
struct item* scoreboard;
struct item* res;
int topk;

inline bool compare_item(const struct item a,const struct item b) {
    return a.weight > b.weight;
}

inline void heap_replace_top(struct item * begin, struct item val) {
    begin--;
    int i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;
        if (i1 > topk)
            break;
        if (i2 == topk + 1 || compare_item(begin[i2], begin[i1])) {
            if (compare_item(begin[i1], val))
                break;
            begin[i] = begin[i1];
            i = i1;
        } else {
            if (compare_item(begin[i2], val))
                break;
            begin[i] = begin[i2];
            i = i2;
        }
    }
    begin[i] = val;
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

void eval(py::array_t<double> array, int k, int squeeze) {
    py::buffer_info buf = array.request();
    topk = k;
    double *ptr = (double *) buf.ptr;
    n_res =buf.shape[0];
    if (num_rows < k) {
        k = num_rows;
    }
    int n_score = (num_rows + (1 << squeeze) - 1) >> squeeze;
    scoreboard = (struct item *) malloc(sizeof(struct item ) * n_score);
    res = (struct item *)malloc(sizeof(struct item ) * n_res * k);
    struct item * hp = res;
    for(int r=0; r<n_res; r++) {
        struct item* loc;
        memset(scoreboard, 0, sizeof(struct item ) * n_score);
        for(int c=0; c<num_dims; c++) {
            double weight = ptr[r*num_dims + c];
            if (fabs(weight) < EPS) {
                continue;
            }
           for(loc=heads[c]; loc < heads[c + 1]; loc ++) {
                int row_id = loc->row_id;
                int key = row_id >> squeeze;
                double rw = weight * loc->weight;
                if (scoreboard[key].row_id > 0 && scoreboard[key].row_id != row_id) {
                    continue;
                } else {
                    scoreboard[key].row_id = row_id;
                    scoreboard[key].weight += rw;
                }
            }
        }

        for (int i =0; i<k; i++) {
            hp[i] = scoreboard[i];
        }
        make_heap(hp, hp + k, compare_item);
        double tp = hp[0].weight;
        for (int i = k; i<n_score; i++) {
            if(tp < scoreboard[i].weight) {
                heap_replace_top(hp, scoreboard[i]);
                tp = hp[0].weight;
            }
        }
        sort_heap(hp, hp + k, compare_item);
        hp += k;
    }
}

vector<vector<int>> result() {
    vector<vector<int>> python_ret;
    vector<int> v1(topk);
    struct item * hp = res;
    for(int j =0 ;j<n_res; j++) {
        for(int i=0; i<topk; i++) {
            v1[i]=hp->row_id;
            hp ++;
        }
        python_ret.push_back(v1);
    }
    return python_ret;
}
