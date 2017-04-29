#include <miblab/classifiers/dtree.h>

DTree::DTree(int i_max_depth, int i_min_sample_count, int i_cv_folds) : Classifier() {
	_i_max_depth = i_max_depth;
	_i_min_sample_count = i_min_sample_count;
	_i_cv_folds = i_cv_folds;
	_model = NULL;
}

DTree::~DTree() {
	free_model();
}

void DTree::free_model() {
	if (_model != NULL) {
		delete _model;
		_model = NULL;
	}
}

void DTree::set_max_depth(int i_max_depth) {
	_i_max_depth = i_max_depth;
}

void DTree::set_min_sample_count(int i_min_sample_count) {
	_i_min_sample_count = i_min_sample_count;
}

void DTree::set_cv_folds(int i_cv_folds) {
	_i_cv_folds = i_cv_folds;
}

float DTree::predict_point_in_matrix(Matrix<float>* x, int i_index) {
	// index = 0 for single point
	int i_dimensions = x->getHeight();

	// convert point to open cv format
	CvMat* cvmat_sample = cvCreateMat(1, i_dimensions, CV_32FC1);
	for (int i=0; i<i_dimensions; i++)
		cvmat_sample->data.fl[i] = (*x)[i][i_index];

	CvDTreeNode* cvdtree_node = _model->predict(cvmat_sample);
	float f_dist = cvdtree_node->value;
	
	cvReleaseMat(&cvmat_sample);
	
	return f_dist;
}

// predict multple points and return list of decision values
void DTree::predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_total_samples = x->getWidth();

	if (vec_index.size() == 0) vec_index = vector<int>(i_total_samples,0);

	// compute distance weighted decision values	
	vec_dists = vector<float>(i_total_samples,0);
	for (int i=0; i<i_total_samples; i++)
		if (vec_index[i] == 0)
			vec_dists[i] = predict_point_in_matrix(x, i);
}

// no scaling required for this classifier
void DTree::predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	predict_points_in_matrix(x, vec_index, vec_dists);
}

void DTree::train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
	int i_total_samples = y->size();
	int i_num_samples = 0;
	int i_dimensions = x->getHeight();

	if (vec_index.size() == 0) {
		vec_index = vector<int>(i_total_samples,0);
		i_num_samples = vec_index.size();
	} else {
		// count the number of samples
		for (int i=0; i<i_total_samples; i++)
			if (vec_index[i] == 0) i_num_samples++;
	}
	
	// convert training data and labels to open cv data type
	CvMat* cvmat_train_data = cvCreateMat(i_num_samples, i_dimensions, CV_32FC1);
	CvMat* cvmat_labels = cvCreateMat(i_num_samples, 1, CV_32SC1);

	_b_multiclass = false;
	int i_max_class = -1;
	int i_cur_sample = 0;
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			int i_index = i_dimensions*i_cur_sample;
			for (int j=0; j<i_dimensions; j++)
				cvmat_train_data->data.fl[i_index+j] = (*x)[j][i];
			cvmat_labels->data.i[i_cur_sample] = (int)(*y)[i];
			if ( (*y)[i] > i_max_class ) i_max_class = (int)(*y)[i];
			i_cur_sample++;
		}
	}
	if (i_max_class > 1) _b_multiclass = true;

	struct CvDTreeParams params;
	params.max_categories = i_max_class+1;
	params.max_depth = _i_max_depth;
	params.min_sample_count = _i_min_sample_count;
	params.cv_folds = _i_cv_folds;
	params.use_surrogates = false;
	params.use_1se_rule = false;
	params.truncate_pruned_tree = true;
	params.priors = 0;

	CvMat* var_type = cvCreateMat( i_dimensions+1, 1, CV_8U );
	cvSet( var_type, cvScalarAll(CV_VAR_ORDERED) );
	var_type->data.ptr[i_dimensions] = CV_VAR_CATEGORICAL;

	// train the classifier
	free_model();
	_model = new CvDTree();
	_model->train(cvmat_train_data, CV_ROW_SAMPLE, cvmat_labels, 0, 0, var_type, 0, params);
	
	// release matrix and labels
	cvReleaseMat(&cvmat_train_data);
	cvReleaseMat(&cvmat_labels);
}

float DTree::predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold) {
	int i_num_test_samples = test_y->size();
	vector<float> vec_dists(i_num_test_samples, 0);

	predict_points_in_matrix(test_x, vec_test_index, vec_dists);

	if (i_metric == METRIC_ACC || _b_multiclass) {
		return compute_accuracy_metric(test_y, vec_dists, vec_test_index, f_threshold);
	} else if (i_metric == METRIC_AUC) {
		return compute_AUC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class);
	} else { // MCC
		return compute_MCC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class, f_threshold);
	}
}
