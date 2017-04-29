#ifndef __CLASSIFIER_SHARED_H__
#define __CLASSIFIER_SHARED_H__

#include <math.h>
#include <miblab/matrix.h>
#include <miblab/machine.h>
#include <miblab/rand.h>

#define CLS_SVM		0
#define CLS_KNN		1
#define CLS_DTREE	2
#define CLS_FOREST	3
#define CLS_PARZEN	4
#define CLS_GMM		5
#define CLS_BAYES	6
#define CLS_GD		7
#define CLS_LR		8

#define TEST_RESUB	0
#define TEST_CV		1
#define TEST_BS		2

#define BS_REG		0
#define BS_0632		1
#define BS_0632P	2

#define KERNEL_LINEAR	0
#define KERNEL_POLY	1
#define KERNEL_RBF	2
#define KERNEL_SIGMOID	3

#define METRIC_ACC	0
#define METRIC_AUC	1
#define METRIC_MCC	2
#define METRIC_BAUC	3

void index_samples(vector<float>* y, vector<vector<int> >& vec2_index_class);
float compute_accuracy_multi(vector<float>* y, vector<float>& vec_responses, vector<int>& vec_index);
float compute_accuracy_metric(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, float f_threshold=0);
float compute_AUC_metric(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, vector<vector<int> >& vec2_index_class);
float compute_MCC_metric(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, float f_threshold=0);
float compute_BAUC_metric(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, float f_threshold=0);
void compute_metrics(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int& tp, int& tn, int& fp, int& fn, float f_threshold=0);
void stratified_cv(int i_folds, vector<vector<int> >& vec2_index_class, vector< vector<int> >& vec2_folds);
void stratified_bs(vector<vector<int> >& vec2_index_class, vector<int>& vec_index);
float compute_bs_metric(float f_metric_resub, float f_metric_bs, int i_bs_type);

#endif
