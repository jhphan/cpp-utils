#include <miblab/classifiers/em.h>

EM::EM(int i_clusters, int i_cov_mat_type, int i_max_iter, float f_eps) : Classifier() {
	_i_clusters = i_clusters;
	_i_cov_mat_type = i_cov_mat_type;
	_i_max_iter = i_max_iter;
	_f_eps = f_eps;
	_i_num_classes = 0;
	_model = NULL;
}

EM::~EM() {
	free_model();
}

void EM::free_model() {
	if (_i_num_classes > 0) {
		for (int i=0; i<_i_num_classes; i++)
			delete _model[i];
		delete[] _model;
		_model = NULL;
	}
}

void EM::set_clusters(int i_clusters) {
	_i_clusters = i_clusters;
}

void EM::set_cov_mat_type(int i_cov_mat_type) {
	_i_cov_mat_type = i_cov_mat_type;
}

void EM::set_max_iter(int i_max_iter) {
	_i_max_iter = i_max_iter;
}

void EM::set_eps(float f_eps) {
	_f_eps = f_eps;
}

float EM::predict_point_in_matrix(Matrix<float>* x, int i_index) {
	// index = 0 for single point
	int i_dimensions = x->getHeight();

	// convert point to open cv format
	CvMat* cvmat_sample = cvCreateMat(1, i_dimensions, CV_32FC1);
	for (int i=0; i<i_dimensions; i++)
		cvmat_sample->data.fl[i] = (*x)[i][i_index];

	// allocate memory for probabilities
	CvMat** cvmat_probs = new CvMat*[_i_num_classes];
	for (int i=0; i<_i_num_classes; i++)
		cvmat_probs[i] = cvCreateMat(1, _i_clusters, CV_64FC1);
	
	// compute probability of sample in each class
	vector<float> vec_responses(_i_num_classes, 0);
	float f_sum_responses = 0;
	for (int i=0; i<_i_num_classes; i++) {
		_model[i]->predict(cvmat_sample, cvmat_probs[i]);
		for (int j=0; j<_i_clusters; j++)
			vec_responses[i] += cvmat_probs[i]->data.db[j];
		f_sum_responses += vec_responses[i];
	}

	// release memory	
	cvReleaseMat(&cvmat_sample);
	for (int i=0; i<_i_num_classes; i++)
		cvReleaseMat(&(cvmat_probs[i]));
	delete[] cvmat_probs;

	// find max
	int i_max_class = 0;
	float f_max_response = 0;
	for (int i=0; i<_i_num_classes; i++) {
		//vec_responses[i] = vec_responses[i];
		if (vec_responses[i] > f_max_response) {
			i_max_class = i;
			f_max_response = vec_responses[i];
		}
	}

	if (_i_num_classes == 2) {
		// return signed probability for 2-class problems
		if (vec_responses[0] > vec_responses[1]) {
			if (f_sum_responses < FLT_EPS) return -1;
			float f_response = -1*vec_responses[0]/f_sum_responses;
			if (isnan(f_response)) return -1;
			return f_response;
		} else {
			if (f_sum_responses < FLT_EPS) return 1;
			float f_response = vec_responses[1]/f_sum_responses;
			if (isnan(f_response)) return 1;
			return f_response;
		}
	} else {
		// return the predicted class for multi-class problems
		return (float)i_max_class;
	}
}

void EM::train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {

	int i_total_samples = y->size();
	int i_num_samples = 0;
	int i_dimensions = x->getHeight();

	// check the index and count number of usable samples
	if (vec_index.size() == 0) {
		vec_index = vector<int>(i_total_samples,0);
		i_num_samples = vec_index.size();
	} else {
		// count the number of samples
		for (int i=0; i<i_total_samples; i++)
			if (vec_index[i] == 0) i_num_samples++;
	}

	// count the number of classes
	int i_max_class = 0;
	for (int i=0; i<i_total_samples; i++)
		if ( (*y)[i] > i_max_class ) i_max_class = (int)(*y)[i];

	int i_num_classes;
	if (i_max_class > 1) {
		i_num_classes = i_max_class+1;
	} else {
		i_num_classes = 2;
	}

	// count the number of samples in each class
	vector<int> vec_num_class(i_num_classes, 0);
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			if ((*y)[i] < 0) 
				vec_num_class[0]++; 
			else 
				vec_num_class[(int)(*y)[i]]++;
		}
	}

	// convert training data and labels to open cv data type
	CvMat** cvmat_train_data = new CvMat*[i_num_classes];
	for (int i=0; i<i_num_classes; i++)
		cvmat_train_data[i] = cvCreateMat(vec_num_class[i], i_dimensions, CV_32FC1);

	vector<int> vec_cur_sample(i_num_classes, 0);
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			if ((*y)[i] < 0) {
				int i_offset = i_dimensions*vec_cur_sample[0];
				for (int j=0; j<i_dimensions; j++)
					cvmat_train_data[0]->data.fl[i_offset+j] = (*x)[j][i];
				vec_cur_sample[0]++;
			} else {
				int i_offset = i_dimensions*vec_cur_sample[(int)(*y)[i]];
				for (int j=0; j<i_dimensions; j++)
					cvmat_train_data[(int)(*y)[i]]->data.fl[i_offset+j] = (*x)[j][i];
				vec_cur_sample[(int)(*y)[i]]++;
			}
		}
	}

	// set EM parameters
	struct CvEMParams params;
	params.nclusters = _i_clusters;
	params.covs = NULL;
	params.means = NULL;
	params.weights = NULL;
	params.probs = NULL;
	params.cov_mat_type = _i_cov_mat_type;
	params.start_step = CvEM::START_AUTO_STEP;
	params.term_crit.max_iter = _i_max_iter;
	params.term_crit.epsilon = _f_eps;
	params.term_crit.type = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;

	// make sure previous models have been freed from memory
	free_model();
	_i_num_classes = i_num_classes;

	// train new models
	_model = new CvEM*[i_num_classes];
	for (int i=0; i<i_num_classes; i++) {
		_model[i] = new CvEM(cvmat_train_data[i], 0, params);
	}
	
	// release data
	for (int i=0; i<i_num_classes; i++)
		cvReleaseMat(&(cvmat_train_data[i]));
	delete[] cvmat_train_data;
}

void EM::predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_num_samples = x->getWidth();
	vec_dists = vector<float>(i_num_samples, 0);
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0)
			vec_dists[i] = predict_point_in_matrix(x, i);
}

void EM::predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_num_samples = x->getWidth();
	vec_dists = vector<float>(i_num_samples, 0);
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0) {
			vec_dists[i] = predict_point_in_matrix(x, i);	
			if (vec_dists[i] < 0) {
				vec_dists[i] = 2*(vec_dists[i]+0.5);
			} else {
				vec_dists[i] = 2*(vec_dists[i]-0.5);
			}
		}
}

float EM::predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold) {
	vector<float> vec_dists;
	predict_points_in_matrix_scaled(test_x, vec_test_index, vec_dists);

	if (_i_num_classes > 2)
		return compute_accuracy_multi(test_y, vec_dists, vec_test_index);

	if (i_metric == METRIC_ACC) {
		return compute_accuracy_metric(test_y, vec_dists, vec_test_index, f_threshold);
	} else if (i_metric == METRIC_AUC) {
		return compute_AUC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class);
	} else { // MCC
		return compute_MCC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class, f_threshold);
	}
}
