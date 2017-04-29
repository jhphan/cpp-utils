#ifndef __NORMALIZE_H__
#define __NORMALIZE_H__

#include <miblab/matrix.h>
#include <miblab/quicksort.h>

using namespace std;

void quantile_reference(Matrix<float>& x, vector<int>& vec_index, vector<float>& vec_ref);
void quantile_normalize_matrix(Matrix<float>& src_x, Matrix<float>& dest_x, vector<int>& vec_index, vector<float>& vec_ref);

#endif
