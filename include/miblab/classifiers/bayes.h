#ifndef __BAYES_H__
#define __BAYES_H__

#include <miblab/classifiers/classifier.h>
#include <opencv/ml.h>

class Bayes : public Classifier {

public:

	Bayes(bool b_pooled=false, int i_cov_type=0);
	~Bayes();

	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index, Matrix<float> ** mat_prodsums);
	float predict_point_in_matrix(Matrix<float>* x, int i_index);
	void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold=0);
	float log_likelihood_points_in_matrix(Matrix<float>* x, vector<float>* y, vector<int>& vec_index, vector<float>& vec_dists); //RMP
	float log_likelihood_point_in_matrix(Matrix<float>* x, vector<float>* y, int i_index); // RMP
private:
	CvNormalBayesClassifier* _model;

	int _i_num_classes;
	bool _b_multiclass;
	bool _b_pooled; // RMP: 0=unpooled, 1=pooled
	int _i_cov_type; // RMP: 0=spherical, 1=diagonal, 2=full
	void free_model();
};

#endif
