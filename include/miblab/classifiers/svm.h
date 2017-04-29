#ifndef __SVM_H__
#define __SVM_H__

#include <miblab/classifiers/classifier.h>
#include <miblab/classifiers/libsvm.h>

class SVM : public Classifier {

public:

	SVM(
		int i_kernel_type = KERNEL_LINEAR,
		float f_eps = 0.001,
		float f_cost = 1,
		float f_gamma = 0,
		float f_coef = 0,
		float f_degree = 0
	);
	~SVM();

	void set_svm_params(float f_eps, float f_cost);
	void set_kernel_params(int i_kernel_type, float f_gamma, float f_coef, float f_degree);

	int get_kernel_type() { return _i_kernel_type; };
	float get_eps() { return _f_eps; };
	float get_cost() { return _f_cost; };
	float get_gamma() { return _f_gamma; };
	float get_coef() { return _f_coef; };
	float get_degree() { return _f_degree; };

	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	float predict_point_in_matrix(Matrix<float>* x, int i_index);
	void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold=0);

private:
	struct svm_model* _model;
	struct svm_node* _x_space;

	int _i_kernel_type;
	float _f_eps;
	float _f_cost;
	float _f_gamma;
	float _f_coef;
	float _f_degree;

	void free_model();
};

#endif
