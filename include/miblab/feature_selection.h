#ifndef __FEATURE_SELECTION_H__
#define __FEATURE_SELECTION_H__

#include <float.h>
#include <math.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include <miblab/quicksort.h>
#include <miblab/machine.h>

#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/fisher_f.hpp>

using namespace std;
using namespace boost::math;

// --- utility functions
//	vec_data and vec_sort_index are outputs
//	of the quicksort functions
void compute_ranks(
	vector<int>& vec_sort_index,
	vector<int>& vec_ranks
);
void compute_ranks_w_ties(
	vector<float>& vec_data,
	vector<int>& vec_sort_index,
	vector<float>& vec_ranks
);
int n_choose_k(
	int n,
	int k
);
float indexed_mean(
	vector<float>& vec_data,
	vector<int>& vec_index
);
float indexed_mean(
	vector<float>& vec_data,
	vector<int>& vec_index,
	vector<int>& vec_index_class
);
float indexed_stdev(
	vector<float>& vec_data,
	vector<int>& vec_index,
	int i_bias = 0
);
float indexed_stdev(
	vector<float>& vec_data,
	vector<int>& vec_index,
	vector<int>& vec_index_class,
	int i_bias = 0
);


// --- fold change feature selection functions
void fold_change_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
);
// this fold change function assumes data are in log scale
// so fold change is absolute difference
float compute_fold_change(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row
);

// --- t-test feature selection functions
void t_test_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
);
float compute_t_stat(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	int i_num_class2
);
float compute_t_stat_p_value(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	int i_num_class2
);

// --- Significance Analysis of Microarrays
void sam_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
);
void compute_sam_stats(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_num_class1,
	int i_num_class2,
	vector<float>& vec_mean_class1,
	vector<float>& vec_mean_class2,
	vector<float>& vec_pooled_stdev
);

// --- Wilcoxon Rank Sum Test
void ranksum_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index,
	bool b_exact
);
int ranksum_dist(
	int n1,
	int n2,
	int u
);
float compute_ranksum_stat(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	int i_num_class2,
	int& i_num_small
);
float compute_ranksum_stat_p_value(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	int i_num_class2,
	bool b_exact
);

// --- minimum Redundancy Maximum Relevance feature selection
void mrmr_index(
	Matrix<float>& mat_data,
	vector<float>& vec_labels,
	vector<int>& vec_index,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index,
	int i_max_num, bool b_miq
);
int discretize_vector(
	vector<float>& vec_float,
	vector<int>& vec_int,
	vector<int>& vec_index
);
void compute_joint_prob(
	Matrix<float>& mat_joint_prob,
	vector<float>& vec_f1,
	vector<float>& vec_f2,
	vector<int>& vec_index
);
float compute_mutual_info(
	vector<float>& vec_f1,
	vector<float>& vec_f2,
	vector<int>& vec_index
);

// --- Rank Products feature selection
void rankprod_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
);
void rankprod_precompute(
	Matrix<float>& mat_data,
	vector<vector<int> >& vec2_index_class,
	Matrix<vector<float> >& mat_scores
);
void rankprod_index_quick(
	Matrix<vector<float> >& mat_scores,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
);

// --- Choi et al. meta-analysis ranking method
// single dataset
void choi_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
);
// multi-dataset
void choi_index(
	vector<Matrix<float> >& vec_mat_data,
	vector<vector<int> >& vec2_index,
	vector<vector<vector<int> > >& vec3_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
);
float choi_t_stat(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	 int i_num_class2
);
void choi_dstar(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_num_class1,
	int i_num_class2,
	 vector<float>& vec_dstar
);
void choi_sigmad(
	vector<float>& vec_dstar,
	int i_num_class1,
	int i_num_class2,
	vector<float>& vec_sigmad
);
void choi_Q(
	vector<vector<float> >& vec2_means,
	vector<vector<float> >& vec2_vars,
	vector<float>& vec_q
);
void choi_tau2DL(
	vector<float>& vec_q,
	vector<vector<float> >& vec2_vars,
	vector<float>& vec_tau2DL
);
void choi_mutau2(
	vector<vector<float> >& vec2_means,
	vector<vector<float> >& vec2_vars,
	vector<float>& vec_tau2,
	vector<float>& vec_mu
);

void wang_index(
	Matrix<float >& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
);
void wang_index(
	vector<Matrix<float> >& vec_mat_data,
	vector<vector<int> >& vec2_index,
	vector<vector<vector<int> > >& vec3_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
);
void wang_means(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_means
);
void wang_vars(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_num_class1,
	int i_num_class2,
	vector<float>& vec_vars
);

// --- other feature selection functions
void random_feature_indexes(
	int i_seed,
	int i_num_features,
	vector<int>& vec_features
);



//rmp
void jbks_test_index(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index);
void spline(const double x[], const double y[], int n, double yp1, double ypn, double y2[]);
void splint(const double xa[], const double ya[], double y2a[], int n, double x, double *y);

float compute_ks_stat(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_row, int i_num_class1, int i_num_class2);
float probks(float alam);
void kstwo(vector<float> &v1, vector<float> &v2, float *d, float *prob);

float indexed_skewness(vector<float>& vec_data, vector<int>& vec_index, vector<int>& vec_index_class);
float indexed_kurtosis(vector<float>& vec_data, vector<int>& vec_index, vector<int>& vec_index_class);
float indexed_compute_jb_test(vector<float>& vec_data, vector<int>& vec_index, vector<int>& vec_index_class);

#define _RMP_N_ALPHAS 17
#define _RMP_N_SAMPLESIZES 33
const double _RMP_ALPHAS[_RMP_N_ALPHAS] = {0.001,0.0016681,0.0027826,0.0046416,0.0077426,0.01,0.012915,0.021544,0.025,0.035938,0.05,0.059948,0.1,0.15,0.2,0.25,0.50};
const double _RMP_REV_ALPHAS[_RMP_N_ALPHAS] = {0.50,0.25,0.2,0.15,0.1,0.059948,0.05,0.035938,0.025,0.021544,0.012915,0.01,0.0077426,0.0046416,0.0027826,0.0016681,0.001};
const double _RMP_SAMPLESIZES[_RMP_N_SAMPLESIZES] = {3,4,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,175,200,250,300,400,500,800,1000,1200,1400,1600,1800,2000,FLT_MAX};
const double _RMP_CRITICAL_VALUES[_RMP_N_ALPHAS][_RMP_N_SAMPLESIZES]={
{ 0.531200,0.960600,1.828900,10.971900,19.542500,25.072200,28.488500,30.610600,31.934300,32.751400,33.227300,33.482500,33.500900,33.261000,32.874200,32.341800,31.888400,30.608900,29.429000,28.422500,27.514000,26.023900,24.835300,23.077100,21.817000,19.553900,18.670700,18.012600,17.519000,17.130500,16.815800,16.558500,13.815500},
{ 0.531200,0.959000,1.805300,9.843000,16.720700,21.000100,23.633200,25.250100,26.269500,26.902500,27.268600,27.476500,27.545800,27.373800,27.097500,26.727800,26.399000,25.474800,24.604300,23.858200,23.191400,22.076400,21.175800,19.854200,18.906800,17.174800,16.487900,15.983000,15.603600,15.302900,15.060000,14.863300,12.792100},
{ 0.531200,0.956400,1.772700,8.675100,14.045400,17.298100,19.281800,20.524200,21.310500,21.801600,22.103000,22.286300,22.379000,22.284800,22.113500,21.877000,21.640700,21.010000,20.397900,19.865800,19.392200,18.587700,17.938400,16.966900,16.266600,14.986500,14.474500,14.099000,13.814900,13.588300,13.410200,13.265700,11.768700},
{ 0.531200,0.952000,1.727600,7.481500,11.552700,13.976200,15.463500,16.408800,17.017600,17.414500,17.659400,17.817200,17.943100,17.920900,17.830500,17.697700,17.551000,17.146000,16.742500,16.384600,16.063500,15.512300,15.063300,14.384300,13.890300,12.976500,12.614300,12.349300,12.150100,11.988300,11.868200,11.766600,10.745400},
{ 0.531200,0.944800,1.666100,6.292700,9.281400,11.060200,12.165800,12.880500,13.357200,13.674900,13.884800,14.034500,14.178800,14.209200,14.191100,14.128300,14.049100,13.827100,13.587800,13.366400,13.164300,12.812400,12.520600,12.074300,11.746900,11.144100,10.907200,10.736200,10.607200,10.505900,10.429600,10.363600,9.722000},
{ 0.531200,0.939600,1.627500,5.707700,8.236500,9.753100,10.705800,11.326300,11.748800,12.035500,12.230200,12.373900,12.525500,12.580100,12.584100,12.554200,12.506700,12.356500,12.180400,12.013900,11.860200,11.592300,11.365300,11.015700,10.760400,10.291400,10.110600,9.980300,9.881100,9.807200,9.747800,9.698100,9.210300},
{ 0.531100,0.932900,1.582800,5.135000,7.261300,8.548900,9.365800,9.906300,10.281800,10.539900,10.720100,10.858600,11.018200,11.088400,11.115500,11.111200,11.088200,10.997600,10.877900,10.758900,10.649400,10.454200,10.283900,10.022600,9.831800,9.485200,9.352100,9.258700,9.186900,9.133300,9.091100,9.054900,8.698700},
{ 0.531000,0.913300,1.472300,4.045600,5.523800,6.440100,7.038900,7.447200,7.739500,7.952000,8.109800,8.234000,8.396800,8.492100,8.552000,8.585000,8.600400,8.602000,8.571900,8.530700,8.490400,8.411400,8.338400,8.226600,8.145400,8.002700,7.950100,7.912700,7.882800,7.861100,7.844600,7.829600,7.675300},
{ 0.530900,0.905600,1.434300,3.747400,5.072900,5.899800,6.446300,6.822200,7.094200,7.294900,7.446900,7.566100,7.728700,7.828600,7.893800,7.936100,7.958900,7.981600,7.971800,7.949600,7.925900,7.875100,7.827600,7.752600,7.698400,7.604000,7.569100,7.544100,7.523900,7.508900,7.497900,7.487200,7.377800},
{ 0.530500,0.881700,1.329400,3.069100,4.077300,4.717600,5.150100,5.457800,5.685600,5.859800,5.994500,6.104500,6.262000,6.369600,6.446200,6.502800,6.542900,6.607100,6.638700,6.657100,6.668800,6.680300,6.684200,6.687700,6.688200,6.685800,6.684700,6.683200,6.680000,6.678900,6.677700,6.676300,6.651900},
{ 0.529700,0.851900,1.218500,2.523900,3.298500,3.801100,4.149400,4.403900,4.597300,4.748100,4.868900,4.969700,5.120300,5.230500,5.313400,5.379600,5.431400,5.527700,5.591900,5.640800,5.678300,5.734300,5.772800,5.824800,5.858100,5.909600,5.928200,5.940800,5.948200,5.954600,5.960000,5.963500,5.991500},
{ 0.529000,0.831600,1.151600,2.255500,2.921500,3.359600,3.668400,3.896800,4.073400,4.212600,4.325800,4.421400,4.568800,4.680300,4.766200,4.837800,4.895700,5.007100,5.086300,5.147400,5.195700,5.268100,5.319400,5.388900,5.433600,5.504300,5.529600,5.547100,5.558700,5.567700,5.575500,5.580600,5.628600},
{ 0.525100,0.755300,0.944200,1.623100,2.053300,2.350500,2.570700,2.743100,2.882700,2.998700,3.097300,3.183400,3.324600,3.437400,3.529200,3.607100,3.673000,3.802500,3.898700,3.973200,4.032700,4.122400,4.187300,4.274800,4.332000,4.424600,4.458000,4.481400,4.497900,4.510500,4.521400,4.528900,4.605200},
{ 0.517600,0.672100,0.794500,1.282100,1.596500,1.823900,1.998600,2.139000,2.254700,2.352400,2.436100,2.509700,2.631600,2.729800,2.810400,2.878800,2.937200,3.052300,3.138400,3.205000,3.258400,3.340200,3.398800,3.479000,3.531800,3.618000,3.650000,3.671800,3.687600,3.699700,3.710100,3.717300,3.794200},
{ 0.507400,0.630300,0.730200,1.123500,1.377900,1.563100,1.706300,1.821600,1.917200,1.998000,2.067400,2.128300,2.229700,2.311500,2.378900,2.436100,2.485100,2.581900,2.654300,2.710600,2.755900,2.825000,2.874900,2.943400,2.988600,3.063100,3.090700,3.109900,3.123700,3.134500,3.143300,3.149900,3.218900},
{ 0.494600,0.594700,0.687800,1.019800,1.233600,1.388500,1.507900,1.604000,1.683500,1.750800,1.808500,1.859200,1.943400,2.011400,2.067400,2.115100,2.155700,2.236300,2.296400,2.343600,2.381200,2.438800,2.480800,2.538200,2.576000,2.638800,2.662400,2.678700,2.690400,2.699600,2.707200,2.712900,2.772600},
{ 0.406300,0.473900,0.528500,0.695100,0.791600,0.857700,0.907100,0.945700,0.977100,1.003300,1.025600,1.044900,1.076800,1.102300,1.123100,1.140800,1.155700,1.185200,1.207200,1.224300,1.238200,1.259100,1.274400,1.295600,1.309700,1.333400,1.342100,1.348400,1.353000,1.356400,1.359500,1.361900,1.386300},
};


#endif
