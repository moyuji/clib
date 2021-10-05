// In all C files, include its corresponding header file in the very first line.
// No need to include <Python.h> as we did that already in the header file.
// Just make sure that <Python.h> is included BEFORE any other header file.
#include "hello.h"
#define EPS 0.000001

unsigned long num_dims;
unsigned long num_rows;
unsigned long n_res;
struct item {
    unsigned long row_id;
    double weight;
};

struct item* matrix;
struct item** heads;
double* score;
unsigned long* ranked_list;
unsigned long* ranked_list_b;
unsigned long * res;
double* umap;
unsigned long topk;
inline bool compare_item(const unsigned long a,const unsigned long b) {
    return umap[a] > umap[b];
}

inline bool compare_weight(const struct item a, const struct item b) {
    return fabs(a.weight) < fabs(b.weight);
}

inline void heap_replace_top(unsigned long * begin, unsigned long val) {
    begin--;
    unsigned long i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;
        if (i1 > topk)
            break;
        if (i2 == topk + 1 || umap[begin[i1]] < umap[begin[i2]]) {
            if (umap[val] < umap[begin[i1]])
                break;
            begin[i] = begin[i1];
            i = i1;
        } else {
            if (umap[val] < umap[begin[i2]])
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

unsigned long load(py::array_t<double> array){
    printf("Loading...\n");
    py::buffer_info buf = array.request();
    num_rows = buf.shape[0];
    num_dims = buf.shape[1];
    double *ptr = (double *) buf.ptr;
    score = (double *)malloc(sizeof(double) * num_rows);
    umap = (double *)malloc(sizeof(double) * num_rows);
    unsigned long ret = 0;
    for (unsigned long r=0; r<num_rows; r++) {
         for(unsigned long c=0; c<num_dims; c++) {
            if (fabs(ptr[r*num_dims + c] ) > EPS) {
                ret += 1;
            }
        }
    }
    matrix = (struct item*) malloc(sizeof(struct item) * ret);
    heads = (struct item**) malloc(sizeof(struct item*) * (num_dims + 1));
    struct item* loc = matrix;
    for(unsigned long c=0; c<num_dims; c++) {
        heads[c] = loc;
        for (unsigned long r=0; r<num_rows; r++) {
            double weight = ptr[r*num_dims + c];
            if (fabs(weight) > EPS) {
                loc->weight = weight;
                loc->row_id = r;
                loc++;
            }
        }
        sort(heads[c], loc, compare_weight);
    }
    heads[num_dims] = loc;
    printf("Loaded!\n");
    return ret;
}

void eval(py::array_t<double> array, unsigned long k) {
    py::buffer_info buf = array.request();
    topk = k;
    double *ptr = (double *) buf.ptr;
    n_res =buf.shape[0];
    if (num_rows < k) {
        k = num_rows;
    }
    unsigned long counter_prod = 0;
    unsigned long counter_heap = 0;
    res = (unsigned long *)malloc(sizeof(unsigned long) * n_res * k);
//    memset(res, 0, sizeof(unsigned long) * n_res * k);
    unsigned long * hp = res;
    for(unsigned long r=0; r<n_res; r++) {
        struct item* loc;
        for(unsigned long c=0; c<num_dims; c++) {
            double weight = ptr[r*num_dims + c];
            if (fabs(weight) < EPS) {
                continue;
            }
           for(loc=heads[c]; loc < heads[c + 1]; loc ++) {
                score[loc->row_id] += weight * loc->weight;
                ++counter_prod;
            }
        }
        unsigned long hp_fill = 0;
        double tp = 0.0;
        for(unsigned long c=0; c<num_dims; c++) {
            double weight = ptr[r*num_dims + c];
            if (fabs(weight) < EPS) {
                continue;
            }
            for(loc=heads[c]; loc < heads[c + 1]; loc ++) {
                unsigned long row_id = loc->row_id;
                double score_now = score[loc->row_id];
                if (fabs(score_now) > EPS) {
                    umap[row_id] = score_now;
                    score[row_id] = 0.0;
                    if (hp_fill < k) {
                        hp[hp_fill++] = row_id;
                        if (hp_fill == k) {
                            make_heap(hp, hp + k, compare_item);
                            tp = umap[hp[0]];
                        }
                    } else {
                        if(tp < score_now) {
                            heap_replace_top(hp, row_id);
                            tp = umap[hp[0]];
                            ++counter_heap;
                        }
                    }
                }
            }
        }
        sort_heap(hp, hp + hp_fill, compare_item);
        hp += k;
    }
    printf("%lu %lu\n", counter_prod, counter_heap);
}

vector<vector<unsigned long>> result() {
    vector<vector<unsigned long>> python_ret;
    vector<unsigned long> v1(topk);
    unsigned long * hp = res;
    for(unsigned long j =0 ;j<n_res; j++) {
        for(unsigned long i=0; i<topk; i++) {
            v1[i]=*hp;
            hp ++;
        }
        python_ret.push_back(v1);
    }
    return python_ret;
}
