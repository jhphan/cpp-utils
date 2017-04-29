#ifndef __LC_H__
#define __LC_H__

#include <miblab/classifiers/classifier.h>

#define LINEAR_FD	0
#define KERNEL_FD	1
#define LINEAR_SDF	2
#define KERNEL_SDF	3

// --- LAPACK fortran routines ---------------------------------------------------------------------------
// generalized symmetric eigenvalue decomposition
//extern "C" void ssygvx_(int*, char*, char*, char*, int*, float*, int*, float*, int*, float*, float*, int*, int*, float*, int*, float*, float*, int*, float*, int*, int*, int*, int*);
extern "C" void dsygvx_(int*, char*, char*, char*, int*, double*, int*, double*, int*, double*, double*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*, int*);
// linear least squares
extern "C" void dgels_(char*, int*, int*, int*, double*, int*, double*, int*, double*, int*, int*);
// symmetric system of equations
extern "C" void dposv_(char*, int*, int*, double*, int*, double*, int*, int*);
// determine machine parameters
extern "C" double dlamch_(char*);

class LC : public Classifier {

public:

	LC(
		int i_classifier_type = LINEAR_FD,
		float f_reg_gamma = 0.01,
		int i_kernel_type = KERNEL_LINEAR,
		float f_gamma = 0,
		float f_coef = 0,
		float f_degree = 0
	);
	~LC();
	void set_lc_params(int i_classifier_type, float f_reg_gamma);
	void set_kernel_params(int i_kernel_type, float f_gamma, float f_coef, float f_degree);

	int get_classifier_type() { return _i_classifier_type; };
	float get_reg_gamma() { return _f_reg_gamma; };
	int get_kernel_type() { return _i_kernel_type; };
	float get_gamma() { return _f_gamma; };
	float get_coef() { return _f_coef; };
	float get_degree() { return _f_degree; };

	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	float predict_point_in_matrix(Matrix<float>* x, int i_index);
	void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold=0);

private:
	// classifier and kernel parameters
	int _i_classifier_type;
	float _f_reg_gamma;
	int _i_kernel_type;
	float _f_gamma;
	float _f_coef;
	float _f_degree;

	// model parameters after training
	Matrix<double> _mat_alpha;
	vector<int> _vec_alpha_index;
	double _d_bias;
	int _i_sign;

	// if kernelized, store training data
	Matrix<float> _mat_train_data;
	vector<int> _vec_train_index;

	// kernel pointers
	typedef double (LC::*kernel_function)(Matrix<float>*, Matrix<float>*, int, int);
	kernel_function _kernel[4];

	// kernel functions
	double kernel_linear(Matrix<float>* x1, Matrix<float>* x2, int i_index1, int i_index2);
	double kernel_rbf(Matrix<float>* x1, Matrix<float>* x2, int i_index1, int i_index2);
	double kernel_poly(Matrix<float>* x1, Matrix<float>* x2, int i_index1, int i_index2);
	double kernel_sigmoid(Matrix<float>* x1, Matrix<float>* x2, int i_index1, int i_index2);
	void init_kernel_pointers();

	// utility functions	
	double* convert_to_fortran(Matrix<double>& mat_data);
	void convert_to_matrix(double* d_fortran_data, Matrix<double>& data);

	double dot_product(Matrix<double>* matrix1, int i_index1, bool b_row1, Matrix<double>* matrix2, int i_index2, bool b_row2);
	void compute_mean_vector(Matrix<float>* x, vector<int>& vec_index, Matrix<double>& mat_mean);
	void compute_class_mean_vector(Matrix<float>* x, vector<float>* y, vector<int>& vec_index, int i_class, Matrix<double>& mat_mean);

	// base training functions
	void train_linear_FD(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	void train_kernel_FD(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	void train_linear_SDF(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	void train_kernel_SDF(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	
	// prediction functions, called by the public prediction functions
	float linear_predict_point_in_matrix(Matrix<float>* x, int i_index);
	float kernel_predict_point_in_matrix(Matrix<float>* mat_train_data, Matrix<float>* x, vector<int>& vec_train_index, int i_index);

	// definition of pure virtual function required
	void free_model();
};

#endif
