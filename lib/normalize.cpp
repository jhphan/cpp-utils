#include <miblab/normalize.h>

// create a reference distribution for quantile normalization
void quantile_reference(Matrix<float>& x, vector<int>& vec_index, vector<float>& vec_ref) {
	int i_num_samples = x.getWidth();
	int i_num_features = x.getHeight();

	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples,0);

	// sort each column
	vector<vector<int> > vec_sort_index(i_num_samples);
	int i_used_samples = 0;
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0) {
			i_used_samples++;
			vector<float> vec_col = x.getColumn(i);
			quicksort_i(vec_col, vec_sort_index[i]);
		}
	}

	// average the columns
	vec_ref = vector<float>(i_num_features,0);
	for (int i=0; i<i_num_features; i++) {
		for (int j=0; j<i_num_samples; j++)
			if (vec_index[j] == 0) vec_ref[i] += x[ vec_sort_index[j][i] ][ j ];
		vec_ref[i] /= (float)i_used_samples;
	}
}

// using sorted reference, vec_ref, as distribution, normalize indexed columns of matrix
void quantile_normalize_matrix(Matrix<float>& src_x, Matrix<float>& dest_x, vector<int>& vec_index, vector<float>& vec_ref) {
	int i_num_samples = src_x.getWidth();
	int i_num_features = src_x.getHeight();
	
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples,0);

	dest_x = src_x;

	// sort each column
	vector<vector<int> > vec_sort_index(i_num_samples);
	int i_used_samples = 0;
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0) {
			quicksort_i(dest_x, vec_sort_index[i], i);
			i_used_samples++;
		}
	}

	// unsort using reference
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0) {
			for (int j=0; j<i_num_features; j++) {
				dest_x[vec_sort_index[i][j]][i] = vec_ref[j];
			}
		}
	}
}
