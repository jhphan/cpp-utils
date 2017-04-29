#include <miblab/classifiers/lr.h>

LR::LR() : Classifier() {
	_lrt = NULL;
	_lrp = NULL;
}

LR::~LR() {
	free_model();
}

void LR::free_model() {
	if (_lrt != NULL) {
		free_lr_train(_lrt);
		_lrt = NULL;
	}
	if (_lrp != NULL) {
		free_lr_predict(_lrp);
		_lrp = NULL;
	}
}

void LR::train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
	int i_total_samples = y->size();
	int i_used_samples = 0;
	int i_dimensions = x->getHeight();
	if (vec_index.size() == 0) {
		vec_index = vector<int>(i_total_samples,0);
		i_used_samples = i_total_samples;
	} else {
		for (int i=0; i<i_total_samples; i++)
			if (vec_index[i] == 0) i_used_samples++;
	}
	
	// copy data matrix to lr dym (matrix) and dyv (vector) structures
	dym* factors = mk_dym(i_used_samples, i_dimensions);
	dyv* outputs = mk_dyv(i_used_samples);

	int i_cur_sample = 0;
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			for (int j=0; j<i_dimensions; j++) {
				dym_set(factors, i_cur_sample, j, (*x)[j][i]);
			}
			dyv_set(outputs, i_cur_sample, ((*y)[i] > 0)?1:0);
			i_cur_sample++;
		}
	}

	lr_options* opts;
	opts = mk_lr_options();

	free_model();
	_lrt = mk_lr_train(NULL, factors, outputs, NULL, opts);
	_lrp = mk_lr_predict(lrt_b0_ref(_lrt), lrt_b_ref(_lrt));
	free_lr_options(opts);
	free_dym(factors);
	free_dyv(outputs);
}

float LR::predict_point_in_matrix(Matrix<float>* x, int i_index) {
	float f_dist;

	int i_dimensions = x->getHeight();
	dyv* testfactors = mk_dyv(i_dimensions);
	for (int i=0; i<i_dimensions; i++)
		dyv_set(testfactors, i, (*x)[i][i_index]);

	f_dist = (float)lr_predict_predict(NULL, testfactors, _lrp);
	f_dist = 2*(f_dist-0.5);
	free_dyv(testfactors);

	return f_dist;
}

void LR::predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_dimensions = x->getHeight();
	int i_num_samples = x->getWidth();
	int i_used_samples = 0;
	vec_dists = vector<float>(i_num_samples, 0);
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);
	
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0) {
			dyv* testfactors = mk_dyv(i_dimensions);
			for (int j=0; j<i_dimensions; j++)
				dyv_set(testfactors, j, (*x)[j][i]);
			vec_dists[i] = (float)lr_predict_predict(NULL, testfactors, _lrp);
			vec_dists[i] = 2*(vec_dists[i]-0.5);
			free_dyv(testfactors);
		}
	}
}

void LR::predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	predict_points_in_matrix(x, vec_index, vec_dists);
	float f_largest = 0;
	int i_num_samples = x->getWidth();
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0)
			if ( fabs(vec_dists[i]) > f_largest) f_largest = fabs(vec_dists[i]);
	if (f_largest < FLT_EPS) f_largest = FLT_EPS;
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0)
			vec_dists[i] = vec_dists[i]/f_largest;
}

float LR::predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold) {
	vector<float> vec_dists;
	predict_points_in_matrix(test_x, vec_test_index, vec_dists);
	if (i_metric == METRIC_ACC) {
		return compute_accuracy_metric(test_y, vec_dists, vec_test_index, f_threshold);
	} else if (i_metric == METRIC_AUC) {
		return compute_AUC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class);
	} else {
		return compute_MCC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class, f_threshold);
	}
}

