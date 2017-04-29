#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

#include <iostream>
#include <vector>
#include <miblab/machine.h>
#include <miblab/matrix.h>
#include <miblab/classifiers/classifier_shared.h>

using namespace std;

class Classifier {

public:
	Classifier();
	virtual ~Classifier() {};

	virtual void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index) = 0;
	virtual float predict_point_in_matrix(Matrix<float>* x, int i_index) = 0;
	virtual void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) = 0;
	virtual void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists) = 0;
	virtual float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold=0) = 0;

	float resubstitution(Matrix<float>* x, vector<float>* y, int i_metric, float f_threshold=0);
	//float loo_cross_validation(Matrix<float>*, vector<float>* y);
	float leave_n_out(Matrix<float>* x, vector<float>* y, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_metric, float f_threshold=0);
	float cross_validation(Matrix<float>* x, vector<float>* y, int i_folds, int i_iterations, int i_metric, float f_threshold=0);
	float bootstrap(Matrix<float>* x, vector<float>* y, int i_bs_type, int i_iterations, int i_metric, float f_threshold=0);

protected:
	virtual void free_model() = 0;

};

#endif
