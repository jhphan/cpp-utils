#ifndef __LR_H__
#define __LR_H__

#include <miblab/classifiers/classifier.h>
extern "C" {
	#include <miblab/classifiers/lr/liblr.h>
}

class LR : public Classifier {

public:

	LR();
	~LR();

	void train(Matrix<float>* x, vector<float>* y, vector<int>& vec_index);
	float predict_point_in_matrix(Matrix<float>* x, int i_index);
	void predict_points_in_matrix(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	void predict_points_in_matrix_scaled(Matrix<float>* x, vector<int>& vec_index, vector<float>& vec_dists);
	float predict_metric(Matrix<float>* test_x, vector<float>* test_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, int i_metric, float f_threshold=0);

private:

	lr_train* _lrt;
	lr_predict* _lrp;

	void free_model();
};

#endif
