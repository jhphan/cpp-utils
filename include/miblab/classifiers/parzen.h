#ifndef __PARZEN_H__
#define __PARZEN_H__

#include <miblab/classifiers/classifier.h>
#include <opencv/ml.h>
#include <math.h>

#define SQRT2PI 2.506628274631

class Parzen : public Classifier {

public:

	Parzen( float f_h = 1, float f_min_val = 0 );
	~Parzen();

	float get_h() { return _f_h; };
	float get_min_val() { return _f_min_val; };
	void set_h(float f_h);
	void set_min_val(float f_min_val);

	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	float predict_point_in_matrix(Matrix<float>* x, int i_index);
	void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold=0);

private:
	CvKNearest* _model;

	float _f_h;
	float _f_min_val;
	int _i_num_samples;

	void free_model();
	float compute_parzen_response(CvMat* cvmat_nearest, CvMat* cvmat_dists, int i_index);
};

#endif
