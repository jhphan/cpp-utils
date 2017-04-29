#include <iostream>
#include <miblab/matrix.h>
#include <miblab/rand.h>
#include <miblab/stringtokenizer.h>
#include <miblab/classifiers/svm.h>
#include <miblab/classifiers/knn.h>
#include <miblab/classifiers/em.h>
#include <miblab/classifiers/lc.h>
#include <miblab/classifiers/dtree.h>
#include <miblab/classifiers/forest.h>
#include <miblab/classifiers/bayes.h>
#include <miblab/classifiers/parzen.h>
#include <miblab/classifiers/lr.h>
#include <miblab/datafile.h>
#include <miblab/strconv.h>

using namespace std;

void generate_linear_data(
//	int i_cluster_samples, 
//	float f_stdev, 
//	float f_dist,
	Matrix<float>& x,
	vector<float>& y
) {

	// write results to file
cerr << "reading data" << endl;
	DataFile<float> df(
		string("./pair_linear_data3"),
		'\t'
		);
	df.readDataFile(x);
	
	Matrix<float> Y;
cerr << "reading labels" << endl;

	DataFile<float> df2(
		string("./pair_linear_labels3"),
		'\t'
		);
	df2.readDataFile(Y);
	y = Y[0];

/*	// total number of training samples
	int i_total_samples = 2*i_cluster_samples;
	// distance between the center of each grid cell
	x = Matrix<float>(2, i_total_samples);
	y = vector<float>(i_total_samples);

	for (int i=0; i<i_cluster_samples; i++) {
		x[0][i] = 4+nrand()*f_stdev-f_dist;
		x[1][i] = 4+nrand()*f_stdev+f_dist;
		y[i] = -1;

		x[0][i+i_cluster_samples] = 4+nrand()*f_stdev+f_dist;
		x[1][i+i_cluster_samples] = 3+nrand()*f_stdev-f_dist;
		y[i+i_cluster_samples] = 1;
	}
*/
}

int main() {
	srand48(time(NULL));

	float f_min_val = -3+4;
	float f_max_val = 3+4;
	int i_cluster_samples = 100;	// number of samples per grid cell or cluster
	int i_grid_res = 300;		// resolution of surface map
	float f_stdev = 1;		// std dev of gaussian in each grid cell
	float f_dist = 0.905;

	Matrix<float> x;
	vector<float> y;

	generate_linear_data(
//		i_cluster_samples, 
//		f_stdev, 
//		f_dist,
		x,
		y
		);
	f_min_val = 3.3205;
	f_max_val = 11.32;	
	i_cluster_samples = y.size();
	i_grid_res = 100;
	

	// define classifier parameters
	int i_num_classifiers = 13;
	vector<vector<float> > vec2_params(i_num_classifiers, vector<float>());

	const float f_rbf_vals[] = {0,0.1,1,10,100};
	vec2_params[0] = vector<float>(f_rbf_vals, f_rbf_vals+5);
	const float f_k_vals[] = {1, 11, 21, 31};
	vec2_params[1] = vector<float>(f_k_vals, f_k_vals+4);
	const float f_clust_vals[] = {2, 3, 4, 5, 20};
	vec2_params[2] = vector<float>(f_clust_vals, f_clust_vals+5);
	vec2_params[3] = vector<float>(f_rbf_vals, f_rbf_vals+5);
	vec2_params[4] = vector<float>(f_rbf_vals, f_rbf_vals+5);
	const float f_dtree_depth[] = {2,5,7,10,100};
	vec2_params[5] = vector<float>(f_dtree_depth, f_dtree_depth+5);
	const float f_forest_trees[] = {10,50,100};
	vec2_params[6] = vector<float>(f_forest_trees, f_forest_trees+3);
	vec2_params[7] = vector<float>(1, 0);
	const float f_degree[] = {2,3,4,5};
	vec2_params[8] = vector<float>(f_degree, f_degree+4);
	vec2_params[9] = vector<float>(f_degree, f_degree+4);
	vec2_params[10] = vector<float>(f_degree, f_degree+4);
	const float f_h[] = {0.1,1,10};
	vec2_params[11] = vector<float>(f_h, f_h+3);
	vec2_params[12] = vector<float>(1, 0);

	const char* ch_classifier_names[] = {"SVM", "KNN", "EM", "LDA", "SDF", "DTree", "Forest", "Bayes", "PolySVM", "PolyLDA", "PolySDF", "Parzen", "LR"};
	vector<string> vec_classifier_names(ch_classifier_names, ch_classifier_names+i_num_classifiers);

	float f_grid_step = (f_max_val-f_min_val)/(float)i_grid_res;
	Matrix<float> mat_results(i_grid_res, i_grid_res);
	vector<int> index;


	DataFile<float> df_points(string("./output_points"), '\t');
	df_points.writeDataFile(x);

	// ------------------------------------------------------------
	Classifier* cls = NULL;

	for (int c=0; c<1; /*vec_classifier_names.size();*/ c++) {
		for (int k=0; k<1; /*vec2_params[c].size();*/ k++) {
			cout << vec_classifier_names[c] << ": " << vec2_params[c][k] << endl;
			// initialize classifier
			if (c == 0) {
				if (vec2_params[c][k] > 0) {
					cls = new SVM(KERNEL_RBF, 0.001, 1, vec2_params[c][k], 0, 0);
				} else {
					cls = new SVM(KERNEL_LINEAR, 0.001, 1, 0, 0, 0);
				}
			} else if (c == 1) {
				cls = new KNN((int)vec2_params[c][k]);
			} else if (c == 2) {
				cls = new EM((int)vec2_params[c][k]);
			} else if (c == 3) {
				if (vec2_params[c][k] > 0) {
					cls = new LC(KERNEL_FD, 0.01, KERNEL_RBF, vec2_params[c][k], 0, 0);
				} else {
					cls = new LC(LINEAR_FD, 0.01, KERNEL_LINEAR, 0, 0, 0);
				}
			} else if (c == 4) {
				if (vec2_params[c][k] > 0) {
					cls = new LC(KERNEL_SDF, 0.01, KERNEL_RBF, vec2_params[c][k], 0, 0);
				} else {
					cls = new LC(LINEAR_SDF, 0.01, KERNEL_LINEAR, 0, 0, 0);
				}
			} else if (c == 5) {
				cls = new DTree((int)vec2_params[c][k], 1, 0);
			} else if (c == 6) {
				cls = new Forest((int)vec2_params[c][k], 0.1, 5, 1);
			} else if (c == 7) {
				cls = new Bayes();
			} else if (c == 8) {
				cls = new SVM(KERNEL_POLY, 0.001, 1, 1, 0.1, (int)vec2_params[c][k]);
			} else if (c == 9) {
				cls = new LC(KERNEL_FD, 0.01, KERNEL_POLY, 1, 0.1, (int)vec2_params[c][k]);
			} else if (c == 10) {
				cls = new LC(KERNEL_SDF, 0.01, KERNEL_POLY, 1, 0.1, (int)vec2_params[c][k]);
			} else if (c == 11) {
				cls = new Parzen(vec2_params[c][k], 0);
			} else if (c == 12) {
				cls = new LR();
			}
			// train classifier
cerr << "train classifier" << endl;
			cls->train(&x, &y, index);

			// predict each point in the grid
			Matrix<float> mat_sample(2, i_grid_res*i_grid_res);
			int i_cur = 0;
			for (int i=0; i<i_grid_res; i++) {
				for (int j=0; j<i_grid_res; j++) {
					mat_sample[0][i_cur] = f_min_val+i*f_grid_step;
					mat_sample[1][i_cur] = f_min_val+j*f_grid_step;
					i_cur++;
				}
			}
cerr << "predict points" << endl;
			vector<float> vec_dists(i_grid_res*i_grid_res,0);
			vector<int> vec_index;
			cls->predict_points_in_matrix_scaled(&mat_sample, vec_index, vec_dists);
			i_cur = 0;
			for (int i=0; i<i_grid_res; i++) {
				for (int j=0; j<i_grid_res; j++) {
					mat_results[i][j] = vec_dists[i_cur];
					i_cur++;
				}
			}

			// write results to file
			DataFile<float> df(
				string("./output_pair_")
					+vec_classifier_names[c]
					+string("_")
					+conv2str(vec2_params[c][k]),
				'\t'
				);
			df.writeDataFile(mat_results);

			delete cls;
		}
	}

	return 0;

}
