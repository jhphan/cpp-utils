#include <miblab/classifiers/knn.h>

KNN::KNN(int i_k) : Classifier() {
	_i_k = i_k;
	_model = NULL;
}

KNN::~KNN() {
	free_model();
}

void KNN::free_model() {
	if (_model != NULL) {
		delete _model;
		_model = NULL;
	}
}

void KNN::set_k(int i_k) {
	_i_k = i_k;
}

float KNN::compute_distance_weighted_decision(CvMat* cvmat_nearest, CvMat* cvmat_dists, int i_index) {
	int i_offset = i_index*_i_k;
	if (_i_k == 1) {
		return cvmat_nearest->data.fl[i_offset];
	} else {
		float f_pos_weight = 0;
		float f_neg_weight = 0;
		float f_total_dist = 0;
		for (int k=0; k<_i_k; k++)
			f_total_dist += cvmat_dists->data.fl[i_offset+k];
		// use a linear distance weight
		for (int k=0; k<_i_k; k++) {
			if (cvmat_nearest->data.fl[i_offset+k] > 0) {
				f_pos_weight += f_total_dist-cvmat_dists->data.fl[i_offset+k];
			} else {
				f_neg_weight += f_total_dist-cvmat_dists->data.fl[i_offset+k];
			}
		}
		return 2*f_pos_weight/(f_pos_weight+f_neg_weight)-1;
	}
}

// unweighted vote
float KNN::compute_decision(CvMat* cvmat_nearest, int i_index) {
	int i_offset = i_index*_i_k;
	if (_i_k == 1) {
		return cvmat_nearest->data.fl[i_offset];
	} else {
		float f_pos_weight = 0;
		float f_neg_weight = 0;
		
		for (int k=0; k<_i_k; k++) {
			if (cvmat_nearest->data.fl[i_offset+k] > 0) {
				f_pos_weight += 1;
			} else {
				f_neg_weight += 1;
			}
		}
		return 2*f_pos_weight/(f_pos_weight+f_neg_weight)-1;
	}
}

float KNN::predict_point_in_matrix(Matrix<float>* x, int i_index) {
	// index = 0 for single point
	int i_dimensions = x->getHeight();

	// convert point to open cv format
	CvMat* cvmat_sample = cvCreateMat(1, i_dimensions, CV_32FC1);
	for (int i=0; i<i_dimensions; i++)
		cvmat_sample->data.fl[i] = (*x)[i][i_index];

	CvMat* cvmat_nearest = cvCreateMat(1, _i_k, CV_32FC1);
	CvMat* cvmat_dists = cvCreateMat(1, _i_k, CV_32FC1);

	// find nearest points
	float f_response = _model->find_nearest(cvmat_sample, _i_k, 0, 0, cvmat_nearest, cvmat_dists);
	// compute distance-weighted decision value
	if (!_b_multiclass)
		f_response = compute_decision(cvmat_nearest, 0);
		//f_response = compute_distance_weighted_decision(cvmat_nearest, cvmat_dists, 0);

	cvReleaseMat(&cvmat_sample);
	cvReleaseMat(&cvmat_nearest);
	cvReleaseMat(&cvmat_dists);
	
	return f_response;
}

// predict multple points and return list of decision values
void KNN::predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_dimensions = x->getHeight();
	int i_total_samples = x->getWidth();

	// count the number of used samples
	int i_used_samples = 0;
	if (vec_index.size() == 0) {
		vec_index = vector<int>(i_total_samples,0);
		i_used_samples = i_total_samples;
	} else {
		for (int i=0; i<i_total_samples; i++)
			if (vec_index[i] == 0)
				i_used_samples++;
	}

	// convert test points to open cv format
	CvMat* cvmat_samples = cvCreateMat(i_used_samples, i_dimensions, CV_32FC1);
	int i_cur_sample = 0;
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			int i_offset = i_cur_sample*i_dimensions;
			for (int j=0; j<i_dimensions; j++)
				cvmat_samples->data.fl[i_offset+j] = (*x)[j][i];
			i_cur_sample++;
		}
	}

	CvMat* cvmat_nearest = cvCreateMat(i_used_samples, _i_k, CV_32FC1);
	CvMat* cvmat_dists = cvCreateMat(i_used_samples, _i_k, CV_32FC1);
	CvMat* cvmat_results = cvCreateMat(i_used_samples, 1, CV_32FC1);

	_model->find_nearest(cvmat_samples, _i_k, cvmat_results, 0, cvmat_nearest, cvmat_dists);

	// compute distance weighted decision values	
	vec_dists = vector<float>(i_total_samples,0);
	i_cur_sample = 0;
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			if (_b_multiclass) 
				vec_dists[i] = cvmat_results->data.fl[i_cur_sample];
			else
				vec_dists[i] = compute_decision(cvmat_nearest, i_cur_sample);
				//vec_dists[i] = compute_distance_weighted_decision(cvmat_nearest, cvmat_dists, i_cur_sample);
			i_cur_sample++;
		}
	}

	// free memory
	cvReleaseMat(&cvmat_samples);
	cvReleaseMat(&cvmat_nearest);
	cvReleaseMat(&cvmat_dists);
	cvReleaseMat(&cvmat_results);
}

void KNN::predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	predict_points_in_matrix(x, vec_index, vec_dists);
}

void KNN::train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
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
	CvMat* cvmat_labels = cvCreateMat(i_num_samples, 1, CV_32FC1);

	_b_multiclass = false;
	int i_cur_sample = 0;
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			int i_index = i_dimensions*i_cur_sample;
			for (int j=0; j<i_dimensions; j++)
				cvmat_train_data->data.fl[i_index+j] = (*x)[j][i];
			cvmat_labels->data.fl[i_cur_sample] = (*y)[i];
			if ( (*y)[i] > 1 ) _b_multiclass = true;
			i_cur_sample++;
		}
	}

	// train the classifier
	free_model();
	_model = new CvKNearest(cvmat_train_data, cvmat_labels, 0, false, _i_k);
	
	// release matrix and labels
	cvReleaseMat(&cvmat_train_data);
	cvReleaseMat(&cvmat_labels);
}

float KNN::predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold) {
	int i_num_test_samples = test_y->size();
	vector<float> vec_dists(i_num_test_samples, 0);

	predict_points_in_matrix(test_x, vec_test_index, vec_dists);

	if (_b_multiclass)
		return compute_accuracy_multi(test_y, vec_dists, vec_test_index);

	if (i_metric == METRIC_ACC) {
		return compute_accuracy_metric(test_y, vec_dists, vec_test_index, f_threshold);
	} else if (i_metric == METRIC_AUC) {
		return compute_AUC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class);
	} else { // MCC
		return compute_MCC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class, f_threshold);
	}
}
