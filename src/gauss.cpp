#include <iostream>
#include <miblab/matrix.h>
#include <miblab/rand.h>
#include <miblab/classifiers/svm.h>
#include <miblab/classifiers/knn.h>
#include <miblab/classifiers/em.h>
#include <miblab/classifiers/lc.h>
#include <miblab/classifiers/dtree.h>
#include <miblab/classifiers/forest.h>
#include <miblab/classifiers/bayes.h>
#include <miblab/datafile.h>
#include <miblab/strconv.h>

using namespace std;

void generate_gauss_data(
	vector<float>& m1,	
	vector<float>& m2,
	Matrix<float>& w1,
	Matrix<float>& w2,
	Matrix<float>& x,
	vector<float>& y
) {
	// total number of training samples per class
	int i_samples_per_class = 100;
	int i_total_samples = 2*i_samples_per_class;

	x = Matrix<float>(2, i_total_samples);
	y = vector<float>(i_total_samples);
	
	// generate samples for class '-1'
	float r1,r2;
	for (int i=0; i<i_samples_per_class; i++){
		r1=nrand();
		r2=nrand();
		x[0][i] = r1*w1[0][0] + r2*w1[1][0] + m1[0];
		x[1][i] = r1*w1[0][1] + r2*w1[1][1] + m1[1];
		y[i] = -1;
	}
	// generate samples for class '+1'
	for (int i=i_samples_per_class; i<2*i_samples_per_class; i++){
		r1=nrand();
		r2=nrand();
		x[0][i] = r1*w2[0][0] + r2*w2[1][0] + m2[0];
		x[1][i] = r1*w2[0][1] + r2*w2[1][1] + m2[1];
		y[i] = 1;
	}
}

int main() {
	srand48(time(NULL));

	int i_grid_size = 3;		// grid size: 3x3
	float f_min_val = 0;		// minimum value for each grid dimension
	float f_max_val = 7.68;		// maximum value for each grid dimension
	int i_cluster_samples = 100;	// number of samples per grid cell or cluster
	int i_grid_res = 300;		// resolution of surface map
	float f_stdev = 0.7;		// std dev of gaussian in each grid cell

	Matrix<float> x;
	vector<float> y;
	const float f_m1[] = {2,4};
	const float f_m2[] = {4,2};
	
	vector<float> m1(f_m1,f_m1+2);
	vector<float> m2(f_m2,f_m2+2);

	// w1--> cov = [1 .2; .2 1];
	Matrix<float> w1(2,2);
	w1[0][0] = 1.0000; w1[0][1] = 0.2000;
	w1[1][0] = 0.0000; w1[1][1] = 0.9798;

	// w2--> cov = [.5 -.2; -.2 .5];
	Matrix<float> w2(2,2);
	w2[0][0] = 0.7071; w2[0][1] = -0.2828;
	w2[1][0] = 0.0000; w2[1][1] =  0.6481;
	
	generate_gauss_data(
		m1,
		m2,
		w1,
		w2,
		x, 
		y
		);

	// define classifier parameters
	int i_num_classifiers = 14;
	vector<vector<float> > vec2_params(i_num_classifiers, vector<float>());
	const float f_rbf_vals[] = {0.1, 1, 10, 100};
	vec2_params[0] = vector<float>(f_rbf_vals, f_rbf_vals+4);
	const float f_k_vals[] = {1,3,5,7,9,11, 21, 31};
	vec2_params[1] = vector<float>(f_k_vals, f_k_vals+8);
	const float f_clust_vals[] = {2, 3, 4, 5, 20};
	vec2_params[2] = vector<float>(f_clust_vals, f_clust_vals+5);
	const float f_0[] = {0};
	vec2_params[3] = vector<float>(f_0, f_0+1);
	const float f_dtree_depth[] = {2,5,7,10,100};
	vec2_params[4] = vector<float>(f_dtree_depth, f_dtree_depth+5);
	const float f_forest_trees[] = {10,50,100};
	vec2_params[5] = vector<float>(f_forest_trees, f_forest_trees+3);
	vec2_params[6] = vector<float>(f_0, f_0+1);
	vec2_params[7] = vector<float>(f_0, f_0+1);
	vec2_params[8] = vector<float>(f_0, f_0+1);
	vec2_params[9] = vector<float>(f_0, f_0+1);
	vec2_params[10] = vector<float>(f_0, f_0+1);
	vec2_params[11] = vector<float>(f_0, f_0+1);
	vec2_params[12] = vector<float>(f_0, f_0+1);
	vec2_params[13] = vector<float>(f_0, f_0+1);
	
	const char* ch_classifier_names[] = {"SVM", "KNN", "EM", "LDA0", "DTree", "Forest","QSDA","NC","QUDA","LUDA","QDA","LDA","LDA_LC","LDA_Bayes"};
	vector<string> vec_classifier_names(ch_classifier_names, ch_classifier_names+i_num_classifiers);

	float f_grid_step = (f_max_val-f_min_val)/(float)i_grid_res;
	Matrix<float> mat_results(i_grid_res, i_grid_res);
	vector<int> index;

	// ------------------------------------------------------------
	Classifier* cls = NULL;

	//for (int c=0; c<vec_classifier_names.size(); c++) {
	for (int c=0; c<14/*vec_classifier_names.size()*/; c++) {
		for (int k=0; k<vec2_params[c].size(); k++) {
			cout << vec_classifier_names[c] << ": " << vec2_params[c][k] << endl;
			// initialize classifier
			if (c == 0) {
				cls = new SVM(KERNEL_RBF, 0.001, 1, vec2_params[c][k], 0, 0);
			} else if (c == 1) {
				cls = new KNN((int)vec2_params[c][k]);
			} else if (c == 2) {
				cls = new EM((int)vec2_params[c][k]);
			} else if (c == 3) {
				cls = new LC(LINEAR_FD,0.0,KERNEL_LINEAR, 0, 0, 0); // LDA
			} else if (c == 4) {
				cls = new DTree((int)vec2_params[c][k], 10, 0);
			} else if (c == 5) {
				cls = new Forest((int)vec2_params[c][k], 0.1, 5, 10);
			} else if (c == 6) {
				cls = new Bayes(false,0); // QSDA = spherical covariance
			} else if (c == 7) {
				cls = new Bayes(true,0); // NC = nearest centroid, pooled spherical covariance
			} else if (c == 8) {
				cls = new Bayes(false,1); // QUDA = diagonal covariance (uncorrelated features) 
			} else if (c == 9) {
				cls = new Bayes(true,1); // LUDA = pooled diagonal covariance
			} else if (c == 10) {
				cls = new Bayes(false,2); // QDA = Full covariance
			} else if (c == 11) {
				cls = new Bayes(true,2); // LDA = Pooled full covariance
			} else if (c == 12) {
				cls = new LC(LINEAR_FD,0.0,KERNEL_LINEAR, 0, 0, 0); // LDA 0
			} else if (c == 13) {
				cls = new Bayes(true,2); // LDA = pooled full covariance
			}
			// train classifier
			cout << "training..." << endl;
			cls->train(&x, &y, index);

			cout << "predicting..." << endl;
			// predict each point in the grid
			Matrix<float> mat_sample(2, 1);
			for (int i=0; i<i_grid_res; i++) {
				for (int j=0; j<i_grid_res; j++) {
					mat_sample[0][0] = i*f_grid_step;
					mat_sample[1][0] = j*f_grid_step;
					mat_results[i][j] = cls->predict_point_in_matrix(&mat_sample, 0);
				}
			}

			cout << "writing..." << endl;
			// write results to file
			DataFile<float> df(
				string("./output_")
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
