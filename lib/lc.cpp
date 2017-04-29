#include <miblab/classifiers/lc.h>

LC::LC(int i_classifier_type, float f_reg_gamma, int i_kernel_type, float f_gamma, float f_coef, float f_degree) : Classifier() {
	_i_classifier_type = i_classifier_type;
	_f_reg_gamma = f_reg_gamma;
	_i_kernel_type = i_kernel_type;
	_f_gamma = f_gamma;
	_f_coef = f_coef;
	_f_degree = f_degree;
	init_kernel_pointers();
}

LC::~LC() {
}

void LC::free_model() {
}

void LC::set_lc_params(int i_classifier_type, float f_reg_gamma) {
	_i_classifier_type = i_classifier_type;
	_f_reg_gamma = f_reg_gamma;
}

void LC::set_kernel_params(int i_kernel_type, float f_gamma, float f_coef, float f_degree) {
	_i_kernel_type = i_kernel_type;
	_f_gamma = f_gamma;
	_f_coef = f_coef;
	_f_degree = f_degree;
}

void LC::init_kernel_pointers() {
	_kernel[KERNEL_LINEAR] = &LC::kernel_linear;
	_kernel[KERNEL_POLY] = &LC::kernel_poly;
	_kernel[KERNEL_RBF] = &LC::kernel_rbf;
	_kernel[KERNEL_SIGMOID] = &LC::kernel_sigmoid;
}

double LC::kernel_linear(Matrix<float>* x1, Matrix<float>* x2, int i_index1, int i_index2) {
	int i_dimensions = x1->getHeight();
	double d_sum = 0;
	for (int i=0; i<i_dimensions; i++)
		d_sum += (*x1)[i][i_index1]*(*x2)[i][i_index2];
	return d_sum;
}

double LC::kernel_rbf(Matrix<float>* x1, Matrix<float>* x2, int i_index1, int i_index2) {
	int i_dimensions = x1->getHeight();
	double d_sum = 0;
	for (int i=0; i<i_dimensions; i++) {
		double d_diff = (*x1)[i][i_index1]-(*x2)[i][i_index2];
		d_sum += d_diff*d_diff;
	}
	return exp(-1*_f_gamma*d_sum);
}

double LC::kernel_poly(Matrix<float>* x1, Matrix<float>* x2, int i_index1, int i_index2) {
	int i_dimensions = x1->getHeight();
	double d_sum = 0;
	for (int i=0; i<i_dimensions; i++)
		d_sum += (*x1)[i][i_index1]*(*x2)[i][i_index2];
	d_sum = _f_gamma*d_sum+_f_coef;
	double d_result = d_sum;
	for (int i=0; i<(int)(_f_degree-1); i++)
		d_result *= d_sum;
	return d_result;
}

double LC::kernel_sigmoid(Matrix<float>* x1, Matrix<float>* x2, int i_index1, int i_index2) {
	int i_dimensions = x1->getHeight();
	double d_sum = 0;
	for (int i=0; i<i_dimensions; i++)
		d_sum += (*x1)[i][i_index1]*(*x2)[i][i_index2];
	d_sum = _f_gamma*d_sum+_f_coef;
	return tanh(d_sum);
}

// convert data to fortran style matrix for lapack
double* LC::convert_to_fortran(Matrix<double>& mat_data) {
	int i_rows = mat_data.getHeight();
	int i_cols = mat_data.getWidth();
	double* d_fortran_data = new double[i_rows*i_cols];
	for (int i=0; i<i_cols; i++)
		for (int j=0; j<i_rows; j++)
			d_fortran_data[i*i_rows+j] = mat_data[j][i];
	return d_fortran_data;
}

// convert fortran style matrix back to regular matrix
// assume mat_data has correct dimensions
void LC::convert_to_matrix(double* d_fortran_data, Matrix<double>& mat_data) {
	int i_rows = mat_data.getHeight();
	int i_cols = mat_data.getWidth();
	for (int i=0; i<i_rows; i++)
		for (int j=0; j<i_cols; j++)
			mat_data[i][j] = d_fortran_data[j*i_rows+i];
}

// compute dot product of row or col vectors in two matricies
double LC::dot_product(Matrix<double>* matrix1, int i_index1, bool b_row1, Matrix<double>* matrix2, int i_index2, bool b_row2) {
        double d_val = 0;
        if (b_row1) {
                int i_vec_size = matrix1->getWidth();
                if (b_row2) {
                        for (int i=0; i<i_vec_size; i++)
                                d_val += (*matrix1)[i_index1][i]*(*matrix2)[i_index2][i];
                } else {
                        for (int i=0; i<i_vec_size; i++)
                                d_val += (*matrix1)[i_index1][i]*(*matrix2)[i][i_index2];
                }
        } else {
                int i_vec_size = matrix1->getHeight();
                if (b_row2) {
                        for (int i=0; i<i_vec_size; i++)
                                d_val += (*matrix1)[i][i_index1]*(*matrix2)[i_index2][i];
                } else {
                        for (int i=0; i<i_vec_size; i++)
                                d_val += (*matrix1)[i][i_index1]*(*matrix2)[i][i_index2];
                }
        }
        return d_val;
}

// compute the mean vector of all column vectors in x (that are indexed)
void LC::compute_mean_vector(Matrix<float>* x, vector<int>& vec_index, Matrix<double>& mat_mean) {
	int i_num_samples = x->getWidth();
	int i_dimensions = x->getHeight();
	int i_used_samples = 0;

	mat_mean = Matrix<double>(i_dimensions, 1, 0);

	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0) {
			i_used_samples++;
			for (int j=0; j<i_dimensions; j++)
				mat_mean[j][0] += (*x)[j][i];
		}
	for (int j=0; j<i_dimensions; j++)
		mat_mean[j][0] = mat_mean[j][0]/(double)(i_used_samples);
}

// compute the mean vector of all column vectors for a particular class
void LC::compute_class_mean_vector(Matrix<float>* x, vector<float>* y, vector<int>& vec_index, int i_class, Matrix<double>& mat_mean) {
	int i_num_samples = x->getWidth();
	int i_dimensions = x->getHeight();
	int i_class_num_samples = 0;

	mat_mean = Matrix<double>(i_dimensions, 1, 0);
	for (int i=0; i<i_num_samples; i++)
		if ((int)(*y)[i] == i_class && vec_index[i] == 0) {
			i_class_num_samples++;
			for (int j=0; j<i_dimensions; j++)
				mat_mean[j][0] += (*x)[j][i];
		}
	for (int j=0; j<i_dimensions; j++)
		mat_mean[j][0] = mat_mean[j][0]/(double)(i_class_num_samples);
}

void LC::train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
	switch (_i_classifier_type) {
	case LINEAR_FD:
		train_linear_FD(x, y, vec_index);
		break;
	case KERNEL_FD:
		train_kernel_FD(x, y, vec_index);
		// store training data
		_mat_train_data = *x;
		_vec_train_index = vec_index;
		break;
	case LINEAR_SDF:
		train_linear_SDF(x, y, vec_index);
		break;
	case KERNEL_SDF:
		train_kernel_SDF(x, y, vec_index);
		// store training data
		_mat_train_data = *x;
		_vec_train_index = vec_index;
	}
}

float LC::linear_predict_point_in_matrix(Matrix<float>* x, int i_index) {
	float f_dist = 0;
	for (int i=0; i<x->getHeight(); i++)
		f_dist += _mat_alpha[i][0]*(*x)[i][i_index];
	return f_dist;
}

float LC::kernel_predict_point_in_matrix(Matrix<float>* mat_train_data, Matrix<float>* x, vector<int>& vec_train_index, int i_index) {
	float f_dist = 0;
	for (int i=0; i<vec_train_index.size(); i++)
		if (vec_train_index[i] == 0)
			f_dist += _mat_alpha[_vec_alpha_index[i]][0]
				*(this->*_kernel[_i_kernel_type])(mat_train_data, x, i, i_index);
	return f_dist;
}

float LC::predict_point_in_matrix(Matrix<float>* x, int i_index) {
	float f_dist;

	if (_i_classifier_type == LINEAR_FD || _i_classifier_type == LINEAR_SDF) {
		f_dist = linear_predict_point_in_matrix(x, i_index);
	} else {
		f_dist = kernel_predict_point_in_matrix(&_mat_train_data, x, _vec_train_index, i_index);
	}

	return _i_sign*(f_dist+_d_bias);
}

void LC::predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_num_samples = x->getWidth();
	vec_dists = vector<float>(i_num_samples, 0);
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);

	if (_i_classifier_type == LINEAR_FD || _i_classifier_type == LINEAR_SDF) {
		for (int i=0; i<i_num_samples; i++)
			if (vec_index[i] == 0)
				vec_dists[i] = linear_predict_point_in_matrix(x, i);
	} else {
		for (int i=0; i<i_num_samples; i++)
			if (vec_index[i] == 0)
				vec_dists[i] = kernel_predict_point_in_matrix(&_mat_train_data, x, _vec_train_index, i);
	}
}

void LC::predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) {
	int i_num_samples = x->getWidth();
	vec_dists = vector<float>(i_num_samples, 0);
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);

	float f_largest = 0;

	if (_i_classifier_type == LINEAR_FD || _i_classifier_type == LINEAR_SDF) {
		for (int i=0; i<i_num_samples; i++)
			if (vec_index[i] == 0) {
				vec_dists[i] = linear_predict_point_in_matrix(x, i);
				if ( fabs(vec_dists[i]) > f_largest ) f_largest = fabs(vec_dists[i]);
			}
	} else {
		for (int i=0; i<i_num_samples; i++)
			if (vec_index[i] == 0) {
				vec_dists[i] = kernel_predict_point_in_matrix(&_mat_train_data, x, _vec_train_index, i);
				if ( fabs(vec_dists[i]) > f_largest ) f_largest = fabs(vec_dists[i]);
			}
	}
	if (f_largest < FLT_EPS) f_largest = FLT_EPS;
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0) {
			vec_dists[i] = vec_dists[i]/f_largest;
		}
	}
}

float LC::predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold) {
	vector<float> vec_dists;
	predict_points_in_matrix_scaled(test_x, vec_test_index, vec_dists);
	if (i_metric == METRIC_ACC) {
		return compute_accuracy_metric(test_y, vec_dists, vec_test_index, f_threshold);
	} else if (i_metric == METRIC_AUC) {
		return compute_AUC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class);
	} else {
		return compute_MCC_metric(test_y, vec_dists, vec_test_index, vec2_test_index_class, f_threshold);
	}
}


void LC::train_linear_FD(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
	int i_num_samples = y->size();
	int i_dimensions = x->getHeight();

	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);

	// count class sizes
	int i_num_class1_samples = 0;
	int i_num_class2_samples = 0;
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0)
			if ((int)(*y)[i] == 1) i_num_class1_samples++; else i_num_class2_samples++;

	// mean vector of positive samples
	Matrix<double> mat_MuC_p;
	compute_class_mean_vector(x, y, vec_index, 1, mat_MuC_p);
	// mean vector of negative samples
	Matrix<double> mat_MuC_n;
	compute_class_mean_vector(x, y, vec_index, -1, mat_MuC_n);
	// mean of all samples
	Matrix<double> mat_Mu;
	compute_mean_vector(x, vec_index, mat_Mu);

	// compute between class scatter, Sb
	Matrix<double> mat_Sb(i_dimensions, i_dimensions, 0);
	Matrix<double> mat_temp_diff = mat_MuC_p-mat_Mu;
	for (int i=0; i<i_dimensions; i++) {
		double d_val1 = mat_temp_diff[i][0];
		for (int j=0; j<=i; j++) {
			double d_val2 = mat_temp_diff[j][0];
			mat_Sb[i][j] = i_num_class1_samples*d_val1*d_val2;
		}
	}
	mat_temp_diff = mat_MuC_n-mat_Mu;
	for (int i=0; i<i_dimensions; i++) {
		double d_val1 = mat_temp_diff[i][0];
		for (int j=0; j<=i; j++) {
			double d_val2 = mat_temp_diff[j][0];
			mat_Sb[i][j] += i_num_class2_samples*d_val1*d_val2;
		}
	}

	// compute within class scatter
	Matrix<double> mat_Sw(i_dimensions, i_dimensions, 0);
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0) {
			if ( (*y)[i] > 0 )
				for (int j=0; j<i_dimensions; j++)
					mat_temp_diff[j][0] = (*x)[j][i]-mat_MuC_p[j][0];
			else
				for (int j=0; j<i_dimensions; j++)
					mat_temp_diff[j][0] = (*x)[j][i]-mat_MuC_n[j][0];
			for (int j=0; j<i_dimensions; j++) {
				double d_val1 = mat_temp_diff[j][0];
				for (int k=0; k<=j; k++) {
					double d_val2 = mat_temp_diff[k][0];
					mat_Sw[j][k] += d_val1*d_val2;
					if (j == k) mat_Sw[j][k] += _f_reg_gamma;
				}
			}
		}
	}

	// convert Sb and Sw matrices to fortran style
	double* d_Sb = convert_to_fortran(mat_Sb);
	double* d_Sw = convert_to_fortran(mat_Sw);

	// solve the generalized eigen problem: Sb*alpha = lambda*Sw*alpha
	// alpha = eigenvector
	// lambda = eigenvalue
	// the eigenvector corresponding to the largest eigenvalue is the hyperplane
	int i_rows = i_dimensions;
	int i_cols = i_dimensions;
        int i_info = 0;
        int i_type = 1;
        char ch_job = 'V';
        char ch_range = 'I';
        char ch_uplo = 'L';
        double d_vl = 0;
        double d_vu = 0;
        int i_l = i_dimensions;
        int i_u = i_dimensions;
        char ch = 'S';
        double d_abstol = 2*dlamch_(&ch);
        int i_num_eigenvals = 0;
        double* d_eigenvalues = new double[i_dimensions];
        double* d_eigenvectors = new double[i_dimensions];
        int i_ldz = i_dimensions;
        double* d_work = new double[i_dimensions*8];
        int i_lwork = i_dimensions*8;
        int* i_work = new int[5*i_dimensions];
        int* i_fail = new int[i_dimensions];

        // generalized symmetric eigenvalue decomposition
        dsygvx_(&i_type, &ch_job, &ch_range, &ch_uplo, &i_rows, d_Sb, &i_rows, d_Sw, &i_rows, &d_vl, &d_vu, &i_l, &i_u, &d_abstol, &i_num_eigenvals, d_eigenvalues, d_eigenvectors, &i_ldz, d_work, &i_lwork, i_work, i_fail, &i_info);

        //if (info > 0) cerr << "eigen decomp error: " << info << endl;

	_mat_alpha = Matrix<double>(i_dimensions, 1, 0);
        convert_to_matrix(d_eigenvectors, _mat_alpha);
        _d_bias = 0;

        // bias is the negative dot product of data midpoint and w
        for (int i=0; i<i_dimensions; i++)
                _d_bias -= _mat_alpha[i][0]*(mat_MuC_p[i][0]+mat_MuC_n[i][0])*0.5;

        // compute the sign of a positive sample, use the positive class mean
        double d_dist = 0;
        for (int i=0; i<i_dimensions; i++)
                d_dist += _mat_alpha[i][0]*mat_MuC_p[i][0];
        d_dist += _d_bias;
        if (d_dist > 0) _i_sign = 1; else _i_sign = -1;

        // free memory -------------------------------------------------------------
        delete[] d_Sb;
        delete[] d_Sw;
        delete[] d_work;
        delete[] d_eigenvalues;
        delete[] d_eigenvectors;
        delete[] i_work;
        delete[] i_fail;
}

void LC::train_kernel_FD(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
	int i_num_samples = y->size();
	int i_dimensions = x->getHeight();

	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples,0);

	vector< vector<int> > vec2_index_class(2, vector<int>());
	vector<int> vec_all_samples;
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0) {
			if ((*y)[i] > 0) {
				vec2_index_class[0].push_back(i);
				vec_all_samples.push_back(i);
			} else {
				vec2_index_class[1].push_back(i);
				vec_all_samples.push_back(i);
			}
		}
	}
	int i_num_class1_samples = vec2_index_class[0].size();
	int i_num_class2_samples = vec2_index_class[1].size();

	i_num_samples = i_num_class1_samples+i_num_class2_samples;

	// --- construct M matrix --------------------------------------------------
		// define M1 and M2 arrays
		Matrix<double> M1(i_num_samples, 1, 0);
		Matrix<double> M2(i_num_samples, 1, 0);

		for (int i=0; i<i_num_samples; i++) {
			for (int j=0; j<i_num_class1_samples; j++) {
				M1[i][0] += (this->*_kernel[_i_kernel_type])(x, x, vec_all_samples[i], vec2_index_class[0][j]);
			}
			M1[i][0] = M1[i][0]/(double)i_num_class1_samples;
			for (int j=0; j<i_num_class2_samples; j++) {
				M2[i][0] += (this->*_kernel[_i_kernel_type])(x, x, vec_all_samples[i], vec2_index_class[1][j]);
			}
			M2[i][0] = M2[i][0]/(double)i_num_class2_samples;
		}

		Matrix<double> Md = M1-M2;	// difference matrix

		// construct M matrix
		Matrix<double> M(i_num_samples, i_num_samples, 0);
		for (int i=0; i<i_num_samples; i++)
			for (int j=0; j<=i; j++)
				M[i][j] = Md[i][0]*Md[j][0];
	
		// clear memory
		M1 = Matrix<double>();
		M2 = Matrix<double>();
		Md = Matrix<double>();

	// --- construct N matrix ----------------------------------------------------

		// construct K1 and K2 matrices
		Matrix<double> K1(i_num_samples, i_num_class1_samples, 0);
		Matrix<double> K2(i_num_samples, i_num_class2_samples, 0);

		for (int i=0; i<i_num_samples; i++) {
			for (int j=0; j<i_num_class1_samples; j++)
				K1[i][j] = (this->*_kernel[_i_kernel_type])(x, x, vec_all_samples[i], vec2_index_class[0][j]);
			for (int j=0; j<i_num_class2_samples; j++)
				K2[i][j] = (this->*_kernel[_i_kernel_type])(x, x, vec_all_samples[i], vec2_index_class[1][j]);
		}

		// construct I-lj matrices
		Matrix<double> I1(i_num_class1_samples, i_num_class1_samples, -1/(double)i_num_class1_samples);
		Matrix<double> I2(i_num_class2_samples, i_num_class2_samples, -1/(double)i_num_class2_samples);
		for (int i=0; i<i_num_class1_samples; i++) I1[i][i] = I1[i][i] + 1;
		for (int i=0; i<i_num_class2_samples; i++) I2[i][i] = I2[i][i] + 1;

		// The following set of for loops does this:
		//Matrix<double> N = K1*I1*~K1+K2*I2*~K2;
		
		Matrix<double> N1(i_num_samples,i_num_class1_samples,0);
		for (int i=0; i<i_num_samples; i++)
			for (int j=0; j<i_num_class1_samples; j++)
				N1[i][j] = dot_product(&K1,i,true,&I1,j,false);
		Matrix<double> N2(i_num_samples,i_num_class2_samples,0);
		for (int i=0; i<i_num_samples; i++)
			for (int j=0; j<i_num_class2_samples; j++)
				N2[i][j] = dot_product(&K2,i,true,&I2,j,false);
		Matrix<double> N(i_num_samples,i_num_samples,0);
		for (int i=0; i<i_num_samples; i++)
			for (int j=0; j<i_num_samples; j++)
				N[i][j] = dot_product(&N1,i,true,&K1,j,true)+dot_product(&N2,i,true,&K2,j,true);
		
		// regularize N matrix
		for (int i=0; i<i_num_samples; i++)
			N[i][i] = N[i][i]+_f_reg_gamma;

		// clear memory
		K1 = Matrix<double>();
		K2 = Matrix<double>();
		I1 = Matrix<double>();
		I2 = Matrix<double>();
		N1 = Matrix<double>();
		N2 = Matrix<double>();

	// ---------------------------------------------------------------------------
	
	// convert to fortran format
	double* d_N = convert_to_fortran(N);
	double* d_M = convert_to_fortran(M);

		// clear memory
		N = Matrix<double>();
	
	// now solve generalized eigen problem: M*alpha = lambda*N*alpha

		// generalized eigenvalue decomposition
		int rows = i_num_samples;
		int cols = i_num_samples;
		int info = 0;
		int itype = 1;
		char jobz = 'V';
		char range = 'I';
		char uplo = 'L';
		double vl = 0;
		double vu = 0;
		int il = i_num_samples;
		int iu = i_num_samples;
		char ch = 'S';
		double abstol = 2*dlamch_(&ch);
		int num_eigenvals = 0;
		double* eigenvalues = new double[i_num_samples];
		double* eigenvectors = new double[i_num_samples];
		int ldz = i_num_samples;
		double* work = new double[i_num_samples*8];
		int lwork = i_num_samples*8;
		int* iwork = new int[5*i_num_samples];
		int* ifail = new int[i_num_samples];
	
		dsygvx_(&itype, &jobz, &range, &uplo, &rows, d_M, &rows, d_N, &rows, &vl, &vu, &il, &iu, &abstol, &num_eigenvals, eigenvalues, eigenvectors, &ldz, work, &lwork, iwork, ifail, &info);
		if (info > 0) cerr << "eigen decomp error: " << info << endl;

	// --------------------------------------------------------------

	_mat_alpha = Matrix<double>(i_num_samples, 1);
	convert_to_matrix(eigenvectors, _mat_alpha);
	_vec_alpha_index = vector<int>(y->size(), 0);
	for (int i=0; i<i_num_samples; i++)
		_vec_alpha_index[vec_all_samples[i]] = i;

	_i_sign = 1;
	_d_bias = 0;

	// compute the bias
	double tempbias1 = 0;
	double tempbias2 = 0;
	for (int i=0; i<i_num_class1_samples; i++)
		tempbias1 += kernel_predict_point_in_matrix(x, x, vec_index, vec2_index_class[0][i]);
	for (int i=0; i<i_num_class2_samples; i++)
		tempbias2 += kernel_predict_point_in_matrix(x, x, vec_index, vec2_index_class[1][i]);
	tempbias1 = tempbias1/(double)i_num_class1_samples;
	tempbias2 = tempbias2/(double)i_num_class2_samples;

	_d_bias = -0.5*(tempbias1+tempbias2);
	if (tempbias1 > tempbias2) _i_sign = 1; else _i_sign = -1;

	// free memory
	delete[] d_N;
	delete[] d_M;
	delete[] work;
	delete[] eigenvalues;
	delete[] eigenvectors;
	delete[] iwork;
	delete[] ifail;
}

void LC::train_linear_SDF(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
	// count the number of samples in each class
	int i_num_samples = y->size();
	int i_dimensions = x->getHeight();

	int i_num_class1_samples = 0;
	int i_num_class2_samples = 0;

	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);

	vector< vector<int> > vec2_index_class(2, vector<int>());
	vector<int> vec_all_samples;
	vector<int> vec_all_samples_class;
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0) {
			if ((int)(*y)[i] == 1) {
				i_num_class1_samples++;
				vec2_index_class[0].push_back(i);
				vec_all_samples.push_back(i);
				vec_all_samples_class.push_back(i_num_class1_samples-1);
			} else {
				i_num_class2_samples++;
				vec2_index_class[1].push_back(i);
				vec_all_samples.push_back(i);
				vec_all_samples_class.push_back(i_num_class2_samples-1);
			}
		}
	}
	i_num_samples = i_num_class1_samples+i_num_class2_samples;

	// compute distance matrix and find shortest distances between classes
	vector<int> vec_class1_index(i_num_class1_samples,0);
	vector<int> vec_class2_index(i_num_class2_samples,0);
	vector<double> vec_class1_shortest(i_num_class1_samples,0);
	vector<double> vec_class2_shortest(i_num_class2_samples,0);

	for (int i=0; i<i_num_class1_samples; i++) {
		for (int j=0; j<i_num_class2_samples; j++) {
			double d_dist = 0;
			for (int k=0; k<i_dimensions; k++) {
				double d_diff = (*x)[k][vec2_index_class[0][i]]-(*x)[k][vec2_index_class[1][j]];
				d_dist += d_diff*d_diff;
			}
			d_dist = sqrt(d_dist);
			if (vec_class1_index[i] > 0) {
				if (d_dist < vec_class1_shortest[i]) vec_class1_shortest[i] = d_dist;
			} else {
				vec_class1_index[i] = 1;
				vec_class1_shortest[i] = d_dist;
			}
			if (vec_class2_index[j] > 0) {
				if (d_dist < vec_class2_shortest[j]) vec_class2_shortest[j] = d_dist;
			} else {
				vec_class2_index[j] = 1;
				vec_class2_shortest[j] = d_dist;
			}
		}
	}

	// compute the signed distance function values, this is also the B matrix for least squares
	int i_max_size = i_num_samples;
	if (i_max_size < i_dimensions+1) i_max_size = i_dimensions+1;
	Matrix<double> sd(i_max_size, 1, 0);
	for (int i=0; i<i_num_samples; i++) {
		if ((int)(*y)[vec_all_samples[i]] == 1) {
			sd[i][0] = (*y)[vec_all_samples[i]]*vec_class1_shortest[vec_all_samples_class[i]]*0.5;
		} else {
			sd[i][0] = (*y)[vec_all_samples[i]]*vec_class2_shortest[vec_all_samples_class[i]]*0.5;
		}
	}

	// compute the A matrix for least squares, bias is the last dimension value
	Matrix<double> A(i_num_samples, i_dimensions+1, 1);
	for (int i=0; i<i_num_samples; i++)
		for (int j=0; j<i_dimensions; j++)
			A[i][j] = (*x)[j][vec_all_samples[i]];

	double* d_B = convert_to_fortran(sd);
	double* d_A = convert_to_fortran(A);

	char trans = 'N';
	int rows = i_num_samples;
	int cols = i_dimensions+1;
	int nrhs = 1;
	int lwork = 2*rows*cols;
	double* work = new double[lwork];
	int info = 0;

	// least squares
	dgels_(&trans, &rows, &cols, &nrhs, d_A, &rows, d_B, &i_max_size, work, &lwork, &info);

	convert_to_matrix(d_B, sd);

	// extract the solution
	_mat_alpha = sd(-1,i_dimensions-1,0,0);
	_d_bias = sd[i_dimensions][0];
	_i_sign = 1;

	delete[] work;
	delete[] d_B;
	delete[] d_A;
}

void LC::train_kernel_SDF(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) {
	// count the number of samples in each class
	int i_num_samples = y->size();
	int i_dimensions = x->getHeight();

	int i_num_class1_samples = 0;
	int i_num_class2_samples = 0;
	if (vec_index.size() == 0) vec_index = vector<int>(i_num_samples, 0);

	vector< vector<int> > vec2_index_class(2, vector<int>());
	vector<int> vec_all_samples;
	vector<int> vec_all_samples_class;
	for (int i=0; i<i_num_samples; i++) {
		if (vec_index[i] == 0) {
			if ((*y)[i] > 0) {
				vec2_index_class[0].push_back(i);
				i_num_class1_samples++;
				vec_all_samples.push_back(i);
				vec_all_samples_class.push_back(i_num_class1_samples-1);
			} else {
				vec2_index_class[1].push_back(i);
				i_num_class2_samples++;
				vec_all_samples.push_back(i);
				vec_all_samples_class.push_back(i_num_class2_samples-1);
			}
		}
	}
	i_num_samples = i_num_class1_samples+i_num_class2_samples;

	// compute distance matrix and find shortest distances between classes
	vector<int> vec_class1_index(i_num_class1_samples,0);
	vector<int> vec_class2_index(i_num_class2_samples,0);
	vector<double> vec_class1_shortest(i_num_class1_samples,0);
	vector<double> vec_class2_shortest(i_num_class2_samples,0);
	for (int i=0; i<i_num_class1_samples; i++) {
		for (int j=0; j<i_num_class2_samples; j++) {
			double d_dist = 0;
			for (int k=0; k<i_dimensions; k++) {
				double d_diff = (*x)[k][vec2_index_class[0][i]]-(*x)[k][vec2_index_class[1][j]];
				d_dist += d_diff*d_diff;
			}
			d_dist = sqrt(d_dist);
			if (vec_class1_index[i] > 0) {
				if (d_dist < vec_class1_shortest[i]) vec_class1_shortest[i] = d_dist;
			} else {
				vec_class1_index[i] = 1;
				vec_class1_shortest[i] = d_dist;
			}
			if (vec_class2_index[j] > 0) {
				if (d_dist < vec_class2_shortest[j]) vec_class2_shortest[j] = d_dist;
			} else {
				vec_class2_index[j] = 1;
				vec_class2_shortest[j] = d_dist;
			}
		}
	}

	// compute the signed distance function values, this is also the B matrix for least squares
	Matrix<double> sd(i_num_samples, 1, 0);
	for (int i=0; i<i_num_samples; i++) {
		if ((int)(*y)[vec_all_samples[i]] == 1) {
			sd[i][0] = (*y)[vec_all_samples[i]]*vec_class1_shortest[vec_all_samples_class[i]]*0.5;
		} else {
			sd[i][0] = (*y)[vec_all_samples[i]]*vec_class2_shortest[vec_all_samples_class[i]]*0.5;
		}
	}

	// compute the kernel matrix , K+N*reg_gamma*I
	Matrix<double> K(i_num_samples, i_num_samples, 0);
	for (int i=0; i<i_num_samples; i++) {
		for (int j=0; j<=i; j++) {
			K[i][j] = (this->*_kernel[_i_kernel_type])(x, x, vec_all_samples[i], vec_all_samples[j]);
			if (i == j) K[i][j] += i_num_samples*_f_reg_gamma;
		}
	}

	double* d_K = convert_to_fortran(K);
	double* d_B = convert_to_fortran(sd);

	char uplo = 'L';
	int N = i_num_samples;
	int nrhs = 1;
	int info = 0;
	
	// solve symmetric system of equations
	dposv_(&uplo, &N, &nrhs, d_K, &N, d_B, &N, &info);

	//if (info > 0) cerr << "##### linear system info: " << info << endl;

	_mat_alpha = Matrix<double>(i_num_samples, 1);
	_vec_alpha_index = vector<int>(y->size(), 0);
	for (int i=0; i<i_num_samples; i++)
		_vec_alpha_index[vec_all_samples[i]] = i;
	convert_to_matrix(d_B, _mat_alpha);
	_d_bias = 0;
	_i_sign = 1;

	delete[] d_K;
	delete[] d_B;
}
