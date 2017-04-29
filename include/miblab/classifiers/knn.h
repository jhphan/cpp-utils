#ifndef __KNN_H__
#define __KNN_H__

#include <miblab/classifiers/classifier.h>
#include <opencv/ml.h>

class KNN : public Classifier {

public:

	KNN( int i_k = 3 );
	~KNN();

	int get_k() { return _i_k; };
	void set_k(int i_k);

	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	float predict_point_in_matrix(Matrix<float>* x, int i_index);
	void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold=0);

private:
	CvKNearest* _model;

	int _i_k;
	bool _b_multiclass;

	void free_model();
	float compute_distance_weighted_decision(CvMat* cvmat_nearest, CvMat* cvmat_dists, int i_index);
	float compute_decision(CvMat* cvmat_nearest, int i_index);
};

#endif
