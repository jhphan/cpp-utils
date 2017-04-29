#include <miblab/classifiers/svm.h>

SVM::SVM(int i_kernel_type, float f_eps, float f_cost, float f_gamma, float f_coef, float f_degree) : Classifier() {
	set_svm_params(f_eps, f_cost);
	set_kernel_params(i_kernel_type, f_gamma, f_coef, f_degree);
	_model = NULL;
	_x_space = NULL;
}

SVM::~SVM() {
	free_model();
}

void SVM::free_model() {
	if (_x_space != NULL) {
		delete[] _x_space;
		_x_space = NULL;
	}
	if (_model != NULL) {
		svm_free_and_destroy_model(&_model);
		_model = NULL;
	}
}

void SVM::set_svm_params(float f_eps, float f_cost) {
	_f_eps = f_eps;
	_f_cost = f_cost;
}

void SVM::set_kernel_params(int i_kernel_type, float f_gamma, float f_coef, float f_degree) {
	_i_kernel_type = i_kernel_type;
	_f_gamma = f_gamma;
	_f_coef = f_coef;
	_f_degree = f_degree;
}

float SVM::predict_point_in_matrix(Matrix<float>* x, int i_index) {
	// index = 0 for single point
	int i_dimensions = x->getHeight();

	// convert to svm_node structure
	struct svm_node* point = new struct svm_node[i_dimensions+1];
	for (int i=1; i<=i_dimensions; i++) {
		point[i-1].index = i;
		point[i-1].value = (double)(*x)[i-1][i_index];
	}
	point[i_dimensions].index = -1;

	// compute decision value
	double d_dist = 0;

	// compute probability
	double* d_probs = new double[2];
	svm_predict_probability(_model, point, d_probs);


//cerr << d_probs[0] << " + " << d_probs[1] << " = " << d_probs[0]+d_probs[1];

//	d_dist = (d_probs[0]-0.5)*2.0*_model->label[0]; // assuming label is +/- 1.
//	delete[] d_probs;

	float f_label = svm_predict(_model, point);
//cerr << ", predicted label = " << f_label << endl;	

	if (_model->label[0] == 1) d_dist = d_probs[0];
	else d_dist = d_probs[1];
	d_dist = 2*d_dist - 1;
	delete[] d_probs;

/*	if (d_probs[0] > d_probs[1]) d_dist = d_probs[0]; else d_dist = d_probs[1];
	d_dist = (d_dist-0.5)*2;
	if (f_label < 0) d_dist = -1*d_dist;
	delete[] d_probs;
*/
	// compute regular distance
	/*
	svm_predict_values(_model, point, &d_dist);

	float f_label = svm_predict(_model, point);
	if (d_dist > 0) {
		if (f_label < 0) d_dist = -1*d_dist;
	} else {
		if (f_label > 0) d_dist = -1*d_dist;
	}
	*/


	delete[] point;

	return (float)d_dist;
}

void SVM::train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {

	struct svm_parameter param;
	struct svm_problem prob;

	// set libsvm C-SVC params
	param.svm_type = C_SVC;
	param.kernel_type = _i_kernel_type;
	param.degree = (int)_f_degree;
	param.gamma = _f_gamma;
	param.coef0 = _f_coef;
	param.nu = 0;
	param.C = 2*_f_cost;
	param.cache_size = 40;
	param.eps = _f_eps;
	param.p = 0;
	param.shrinking = 1;
	param.probability = 1;
	param.nr_weight = 2;
	param.weight_label = new int[2];
	param.weight_label[0] = 1;
	param.weight_label[1] = -1;
	param.weight = new double[2];

	// convert data to libsvm format
	int i_total_samples = y->size();
	if (vec_index.size() == 0) {
		vec_index = vector<int>(i_total_samples,0);
		prob.l = i_total_samples;
	} else {
		// count the number of samples
		prob.l = 0;
		for (int i=0; i<i_total_samples; i++)
			if (vec_index[i] == 0) prob.l++;
	}

	int i_dimensions = x->getHeight();
	int i_elements = prob.l*(i_dimensions+1);

	prob.y = new double[prob.l];
	prob.x = new struct svm_node*[prob.l];

	free_model();
	_x_space = new struct svm_node[i_elements];

	int n = 0;
	int i_sample_num = 0;
	int i_numclass1 = 0;
	for (int i=0; i<i_total_samples; i++) {
		if (vec_index[i] == 0) {
			if ((*y)[i] > 0)
				i_numclass1++;
			prob.y[i_sample_num] = (double)(*y)[i];
			prob.x[i_sample_num] = &(_x_space[n]);
			for (int j=1; j<=i_dimensions; j++) {
				_x_space[n].index = j;
				_x_space[n].value = (double)(*x)[j-1][i];
				//_x_space[n].sampindex = i_sample_num;
				n++;
			}
			_x_space[n++].index = -1;
			i_sample_num++;
		}
	}
	param.weight[0] = (prob.l-i_numclass1)/(double)prob.l;
	param.weight[1] = i_numclass1/(double)prob.l;
//cerr << i_numclass1 << " pos, " << i_sample_num-i_numclass1 << " neg" << endl;
//cerr << "+ weight: " << param.weight[(param.weight_label[1]+1)/2] << ", - weight: " << param.weight[(param.weight_label[0]+1)/2] << endl;
//cerr << "svm_train" << endl;
	// train svm
	_model = svm_train(&prob, &param);

	// free memory
	delete[] prob.y;
	delete[] prob.x;
	delete[] param.weight_label;
	delete[] param.weight;
}

void SVM::predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_num_samples = x->getWidth();
	vec_dists = vector<float>(i_num_samples, 0);
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0)
			vec_dists[i] = predict_point_in_matrix(x, i);
}

void SVM::predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_num_samples = x->getWidth();
	vec_dists = vector<float>(i_num_samples, 0);
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);
	//float f_largest = 0;
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0) {
			vec_dists[i] = predict_point_in_matrix(x, i);
			//if ( fabs(vec_dists[i]) > f_largest) f_largest = fabs(vec_dists[i]);
		}
	//if (f_largest < FLT_EPS) f_largest = FLT_EPS;
	//for (int i=0; i<i_num_samples; i++)
	//	if (vec_index[i] == 0)
	//		vec_dists[i] = vec_dists[i]/f_largest;
}

float SVM::predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold) {
	vector<float> vec_dists;
	predict_points_in_matrix(test_x, vec_test_index, vec_dists);

	if (i_metric == METRIC_ACC) {
		return compute_accuracy_metric(test_y, vec_dists, vec_test_index, f_threshold);
	} else if (i_metric == METRIC_AUC) {
		return compute_AUC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class);
	} else { // MCC
		return compute_MCC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class, f_threshold);
	}
}
