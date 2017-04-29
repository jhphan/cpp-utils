#ifndef __DTREE_H__
#define __DTREE_H__

#include <miblab/classifiers/classifier.h>
#include <opencv/ml.h>

class DTree : public Classifier {

public:

	DTree( int i_max_depth = 10, int i_min_sample_count = 10, int i_cv_folds = 0 );
	~DTree();

	int get_max_depth() { return _i_max_depth; };
	int get_min_sample_count() { return _i_min_sample_count; };
	int get_cv_folds() { return _i_cv_folds; };

	void set_max_depth(int i_max_depth);
	void set_min_sample_count(int i_min_sample_count);
	void set_cv_folds(int i_cv_folds);

	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	float predict_point_in_matrix(Matrix<float>* x, int i_index);
	void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold=0);

private:
	CvDTree* _model;

	int _i_max_depth;
	int _i_min_sample_count;
	int _i_cv_folds;
	bool _b_multiclass;

	void free_model();
};

#endif
