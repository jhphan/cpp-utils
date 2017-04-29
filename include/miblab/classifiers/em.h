#ifndef __EM_H__
#define __EM_H__

#include <miblab/classifiers/classifier.h>
#include <opencv/ml.h>

#define COV_SPHERICAL	CvEM::COV_MAT_SPHERICAL
#define COV_DIAGONAL	CvEM::COV_MAT_DIAGONAL
#define COV_GENERIC	CvEM::COV_MAT_GENERIC

class EM : public Classifier {

public:

	EM( int i_clusters = 1, int i_cov_mat_type = COV_SPHERICAL, int i_max_iter = 100, float f_eps = 0.01 );
	~EM();

	void set_clusters(int i_clusters);
	void set_cov_mat_type(int i_cov_mat_type);
	void set_max_iter(int i_max_iter);
	void set_eps(float f_eps);

	int get_clusters() { return _i_clusters; };
	int get_cov_mat_type() { return _i_cov_mat_type; };
	int get_max_iter() { return _i_max_iter; };
	float get_eps() { return _f_eps; };

	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	float predict_point_in_matrix(Matrix<float>* x, int i_index);
	void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold=0);

private:
	int _i_clusters;
	int _i_cov_mat_type;
	int _i_max_iter;
	float _f_eps;
	int _i_num_classes;

	CvEM** _model;

	void free_model();
};

#endif
