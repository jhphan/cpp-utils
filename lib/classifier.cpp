#include <miblab/classifiers/classifier.h>

Classifier::Classifier() {
}

float Classifier::resubstitution(Matrix<float>* x, vector<float>* y, int i_metric, float f_threshold) {
	vector<int> vec_index;
	train(x, y, vec_index);

	vector<vector<int> > vec2_index_class;
	index_samples(y, vec2_index_class);

	return predict_metric(x, y, vec_index, vec2_index_class, i_metric, f_threshold);
}

float Classifier::leave_n_out(Matrix<float>* x, vector<float>* y, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_metric, float f_threshold) {
	vector<int> vec_test_index(vec_index.size(), 0);
	for (int i=0; i<vec_index.size(); i++)
		if (vec_index[i] == 0) vec_test_index[i] == 1;

	train(x, y, vec_index);

	return predict_metric(x, y, vec_test_index, vec2_index_class, i_metric, f_threshold);
}

float Classifier::cross_validation(Matrix<float>* x, vector<float>* y, int i_folds, int i_iterations, int i_metric, float f_threshold) {
	vector<vector<int> > vec2_index_class;
	index_samples(y, vec2_index_class);
	
	float f_metric_cv = 0;
	for (int i=0; i<i_iterations; i++) {
		vector<vector<int> > vec2_folds(i_folds, vector<int>(y->size(), 0));
		stratified_cv(i_folds, vec2_index_class, vec2_folds);
		for (int j=0; j<i_folds; j++)
			f_metric_cv += leave_n_out(x, y, vec2_folds[j], vec2_index_class, i_metric, f_threshold);
	}

	return f_metric_cv/(float)(i_iterations*i_folds);
}

float Classifier::bootstrap(Matrix<float>* x, vector<float>* y, int i_bs_type, int i_iterations, int i_metric, float f_threshold) {
	vector<vector<int> > vec2_index_class;
	index_samples(y, vec2_index_class);

	float f_metric_resub = 0;
	if (i_bs_type == BS_0632 || i_bs_type == BS_0632P)
		f_metric_resub = resubstitution(x, y, i_metric, f_threshold);

	float f_metric_bs = 0;
	vector<int> vec_bs_index(y->size(), 0);
	for (int i=0; i<i_iterations; i++) {
		stratified_bs(vec2_index_class, vec_bs_index);
		f_metric_bs += leave_n_out(x, y, vec_bs_index, vec2_index_class, i_metric, f_threshold);
	}
	f_metric_bs = f_metric_bs/(float)i_iterations;

	return compute_bs_metric(f_metric_resub, f_metric_bs, i_bs_type);
}

