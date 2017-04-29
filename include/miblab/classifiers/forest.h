#ifndef __FOREST_H__
#define __FOREST_H__

#include <miblab/classifiers/classifier.h>
#include <opencv/ml.h>

class Forest : public Classifier {

public:

	Forest( int i_max_trees = 50, float f_max_oob_error = 0.1, int i_max_depth = 10, int i_min_sample_count = 10 );
	~Forest();

	int get_max_trees() { return _i_max_trees; };
	float get_max_oob_error() { return _f_max_oob_error; };
	int get_max_depth() { return _i_max_depth; };
	int get_min_sample_count() { return _i_min_sample_count; };

	void set_max_trees(int i_max_trees);
	void set_max_oob_error(float f_max_oob_error);
	void set_max_depth(int i_max_depth);
	void set_min_sample_count(int i_min_sample_count);

	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	float predict_point_in_matrix(Matrix<float>* x, int i_index);
	void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold);

private:
	CvRTrees* _model;

	int _i_max_trees;
	float _f_max_oob_error;
	int _i_max_depth;
	int _i_min_sample_count;
	bool _b_multiclass;

	void free_model();
};

#endif
