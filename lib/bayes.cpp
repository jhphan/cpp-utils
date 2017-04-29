#include <miblab/classifiers/bayes.h>

Bayes::Bayes(bool b_pooled, int i_cov_type) : Classifier() {
	_i_num_classes = 0;
	_b_multiclass = false;
	_model = NULL;
	_b_pooled = b_pooled;
	_i_cov_type = i_cov_type;
}

Bayes::~Bayes() {
	free_model();
}

void Bayes::free_model() {
	if (_model != NULL) {
		delete _model;
		_model = NULL;
	}
}

float Bayes::predict_point_in_matrix(Matrix<float>* x, int i_index) {
	// index = 0 for single point
	int i_dimensions = x->getHeight();

	// convert point to open cv format
	CvMat* cvmat_sample = cvCreateMat(1, i_dimensions, CV_32FC1);
	for (int i=0; i<i_dimensions; i++)
		cvmat_sample->data.fl[i] = (*x)[i][i_index];

	CvMat* cvmat_probs = cvCreateMat(1, _i_num_classes, CV_64FC1);
	CvMat* cvmat_labels = cvCreateMat(1, _i_num_classes, CV_32SC1);

	// train classifier
	float f_response = _model->predict(cvmat_sample, 0, cvmat_probs, cvmat_labels);
	float f_dist;
	if (!_b_multiclass) {
		if ((int)f_response == cvmat_labels->data.i[0]) {
			f_dist = cvmat_probs->data.db[0];
		} else {
			f_dist = cvmat_probs->data.db[1];
		}
		if (isnan(f_dist)) {
			f_dist = f_response;
		} else {
			if (f_response < 0)
				f_dist = -1*f_dist;
		}
	} else {
		f_dist = f_response;
	}

	cvReleaseMat(&cvmat_sample);
	cvReleaseMat(&cvmat_probs);
	cvReleaseMat(&cvmat_labels);
	
	return f_dist;
}
float Bayes::log_likelihood_point_in_matrix(Matrix<float>* x, vector<float>* y, int i_index) {
	// index = 0 for single point
	int i_dimensions = x->getHeight(), i;
//cout << "Bayes::log_likelihood_point_in_matrix" << endl;

	// convert point to open cv format
	CvMat* cvmat_sample = cvCreateMat(1, i_dimensions, CV_32FC1);
	for (int i=0; i<i_dimensions; i++)
		cvmat_sample->data.fl[i] = (*x)[i][i_index];

	CvMat* cvmat_probs = cvCreateMat(1, _i_num_classes, CV_64FC1);
	CvMat* cvmat_labels = cvCreateMat(1, _i_num_classes, CV_32SC1);

//if (i_index<10) cout << "x = (" << cvmat_sample->data.fl[0] << ", " << cvmat_sample->data.fl[1] << ")\t";
	// train classifier
	bool scale_probs=false; // compute log P(point|model)
	float f_response = _model->predict(cvmat_sample, 0, cvmat_probs, cvmat_labels, scale_probs);
	float f_dist=-1;

//if (i_index<10) cout << "prob = (" << cvmat_probs->data.db[0]/2/3.141592653589793 << ", " << cvmat_probs->data.db[1]/2/3.141592653589793 << ")" << endl;

	for (i=0; i<_i_num_classes; i++){
		if ((int) (*y)[i_index] == cvmat_labels->data.i[i]){
			f_dist = cvmat_probs->data.db[i];
			f_dist /= 2*3.141592653589793; // opencv returns prob*2*pi.
			break;
		}
	}
	if (i==_i_num_classes) {
		cout << "Error Bayes::log_likelihood_point_in_matrix: class label " << (*y)[i_index] << " not found." << endl;
	}
	cvReleaseMat(&cvmat_sample);
	cvReleaseMat(&cvmat_probs);
	cvReleaseMat(&cvmat_labels);
	
	return log(f_dist);
}

void Bayes::train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
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

	int i_max_class = 0;
	int i_cur_sample = 0;
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			int i_index = i_dimensions*i_cur_sample;
			for (int j=0; j<i_dimensions; j++)
				cvmat_train_data->data.fl[i_index+j] = (*x)[j][i];
			cvmat_labels->data.fl[i_cur_sample] = (*y)[i];
			if ( (*y)[i] > i_max_class ) i_max_class = (int)(*y)[i];
			i_cur_sample++;
		}
	}
	if (i_max_class > 1) {
		_b_multiclass = true;
		_i_num_classes = i_max_class+1;
	} else {
		_b_multiclass = false;
		_i_num_classes = 2;
	}

	// train the classifier
	free_model();
	const CvMat* varidx=0;
	const CvMat* sampleidx=0;

// memory leak was here:
	_model = new CvNormalBayesClassifier(cvmat_train_data, cvmat_labels, varidx, sampleidx, _b_pooled, _i_cov_type);
	

	// release matrix and labels
	cvReleaseMat(&cvmat_train_data);
	cvReleaseMat(&cvmat_labels);

}

void Bayes::train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index, Matrix<float> **mat_covs) {
	int i_dimensions = x->getHeight();
	train(x,y,vec_index);

	CvMat **covs = (CvMat**) cvAlloc(_i_num_classes*sizeof(CvMat*));
	for (int i=0;i<_i_num_classes;i++) covs[i] = cvCreateMat(i_dimensions, i_dimensions, CV_64FC1);

	_model->getCovs(covs);

	for (int i=0; i<_i_num_classes; i++){
//printf("class %d:\n",i);
		for (int j=0;j<i_dimensions;j++){
			for (int k=0;k<i_dimensions;k++){
//printf("%f\t",prodsums[i]->data.db[j*i_dimensions + k]);
				(*mat_covs[i])[j][k] = covs[i]->data.db[j*i_dimensions + k];
			}
//printf("\n");
		}
	}
	for (int i=0;i<_i_num_classes;i++) cvReleaseMat( &covs[i] );
	cvFree( &covs );
}

void Bayes::predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_num_samples = x->getWidth();
	vec_dists = vector<float>(i_num_samples, 0);
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0)
			vec_dists[i] = predict_point_in_matrix(x, i);
}

void Bayes::predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
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

float Bayes::log_likelihood_points_in_matrix(Matrix<float>* x, vector<float>* y, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_num_samples = x->getWidth();
	float joint=0.0;
//cout << "Bayes::log_likelihood_points_in_matrix" << endl;
	vec_dists = vector<float>(i_num_samples, 0);
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0){
			vec_dists[i] = log_likelihood_point_in_matrix(x, y, i);
			joint+=vec_dists[i];
		}
	}
	return joint;
}

float Bayes::predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold) {
	vector<float> vec_dists;
	predict_points_in_matrix_scaled(test_x, vec_test_index, vec_dists);

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
