#include <miblab/classifiers/parzen.h>

Parzen::Parzen(float f_h, float f_min_val) : Classifier() {
	_f_h = f_h;
	_f_min_val = f_min_val;
	_i_num_samples = 0;
	_model = NULL;
}

Parzen::~Parzen() {
	free_model();
}

void Parzen::free_model() {
	if (_model != NULL) {
		delete _model;
		_model = NULL;
	}
}

void Parzen::set_h(float f_h) {
	_f_h = f_h;
}

void Parzen::set_min_val(float f_min_val) {
	_f_min_val = f_min_val;
}

float Parzen::compute_parzen_response(CvMat* cvmat_nearest, CvMat* cvmat_dists, int i_index) {
	int i_offset = i_index*_i_num_samples;

	float f_neg = 0;
	int i_total_neg = 0;
	float f_pos = 0;
	int i_total_pos = 0;
	for (int i=0; i<_i_num_samples; i++) {
		float f_dist = cvmat_dists->data.fl[i_offset+i];
		float f_val = exp( -1*f_dist*f_dist/(_f_h*_f_h) );
		if (f_val < _f_min_val)
			break;
		if (cvmat_nearest->data.fl[i_offset+i] > 0) {
			f_pos += f_val;
			i_total_pos++;
		} else {
			f_neg += f_val;
			i_total_neg++;
		}
	}
	if (i_total_pos == 0 && i_total_neg == 0) {
		return 0;
	}
	if (i_total_pos == 0) {
		return -1;
	}
	if (i_total_neg == 0) {
		return 1;
	}
	f_pos = f_pos/(i_total_pos*_f_h*SQRT2PI);
	f_neg = f_neg/(i_total_neg*_f_h*SQRT2PI);

	if (f_pos > f_neg) {
		if (f_pos+f_neg < FLT_EPS) return 1;
		return 2*(f_pos/(f_pos+f_neg)-0.5);
	} else {
		if (f_pos+f_neg < FLT_EPS) return -1;
		return -2*(f_neg/(f_pos+f_neg)-0.5);
	}
}

float Parzen::predict_point_in_matrix(Matrix<float>* x, int i_index) {
	// index = 0 for single point
	int i_dimensions = x->getHeight();

	// convert point to open cv format
	CvMat* cvmat_sample = cvCreateMat(1, i_dimensions, CV_32FC1);
	for (int i=0; i<i_dimensions; i++)
		cvmat_sample->data.fl[i] = (*x)[i][i_index];

	CvMat* cvmat_nearest = cvCreateMat(1, _i_num_samples, CV_32FC1);
	CvMat* cvmat_dists = cvCreateMat(1, _i_num_samples, CV_32FC1);

	// find nearest points
	_model->find_nearest(cvmat_sample, _i_num_samples, 0, 0, cvmat_nearest, cvmat_dists);

	float f_response = compute_parzen_response(cvmat_nearest, cvmat_dists, 0);

	cvReleaseMat(&cvmat_sample);
	cvReleaseMat(&cvmat_nearest);
	cvReleaseMat(&cvmat_dists);
	
	return f_response;
}

// predict multple points and return list of decision values
void Parzen::predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
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

	CvMat* cvmat_nearest = cvCreateMat(i_used_samples, _i_num_samples, CV_32FC1);
	CvMat* cvmat_dists = cvCreateMat(i_used_samples, _i_num_samples, CV_32FC1);
	CvMat* cvmat_results = cvCreateMat(i_used_samples, 1, CV_32FC1);

	_model->find_nearest(cvmat_samples, _i_num_samples, cvmat_results, 0, cvmat_nearest, cvmat_dists);

	// compute distance weighted decision values	
	vec_dists = vector<float>(i_total_samples,0);
	i_cur_sample = 0;
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			vec_dists[i] = compute_parzen_response(cvmat_nearest, cvmat_dists, i_cur_sample);
			i_cur_sample++;
		}
	}

	// free memory
	cvReleaseMat(&cvmat_samples);
	cvReleaseMat(&cvmat_nearest);
	cvReleaseMat(&cvmat_dists);
	cvReleaseMat(&cvmat_results);
}

void Parzen::predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	predict_points_in_matrix(x, vec_index, vec_dists);
}

void Parzen::train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
	int i_total_samples = y->size();
	int i_dimensions = x->getHeight();

	if (vec_index.size() == 0) {
		vec_index = vector<int>(i_total_samples,0);
		_i_num_samples = vec_index.size();
	} else {
		// count the number of samples
		for (int i=0; i<i_total_samples; i++)
			if (vec_index[i] == 0) _i_num_samples++;
	}
	
	
	// convert training data and labels to open cv data type
	CvMat* cvmat_train_data = cvCreateMat(_i_num_samples, i_dimensions, CV_32FC1);
	CvMat* cvmat_labels = cvCreateMat(_i_num_samples, 1, CV_32FC1);

	int i_cur_sample = 0;
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			int i_index = i_dimensions*i_cur_sample;
			for (int j=0; j<i_dimensions; j++)
				cvmat_train_data->data.fl[i_index+j] = (*x)[j][i];
			cvmat_labels->data.fl[i_cur_sample] = (*y)[i];
			i_cur_sample++;
		}
	}

	// train the classifier
	free_model();
	_model = new CvKNearest(cvmat_train_data, cvmat_labels, 0, false, _i_num_samples);
	
	// release matrix and labels
	cvReleaseMat(&cvmat_train_data);
	cvReleaseMat(&cvmat_labels);
}

float Parzen::predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold) {
	int i_num_test_samples = test_y->size();
	vector<float> vec_dists(i_num_test_samples, 0);

	predict_points_in_matrix(test_x, vec_test_index, vec_dists);

	if (i_metric == METRIC_ACC) {
		return compute_accuracy_metric(test_y, vec_dists, vec_test_index, f_threshold);
	} else if (i_metric == METRIC_AUC) {
		return compute_AUC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class);
	} else { // MCC
		return compute_MCC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class, f_threshold);
	}
}
