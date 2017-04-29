#include <iostream>
#include <miblab/matrix.h>
#include <miblab/rand.h>
#include <miblab/classifiers/knn.h>
#include <miblab/classifiers/em.h>
#include <miblab/classifiers/bayes.h>
#include <miblab/classifiers/dtree.h>
#include <miblab/classifiers/forest.h>
#include <miblab/datafile.h>
#include <miblab/strconv.h>

using namespace std;

void generate_checkerboard_data(
	int i_grid_size, 
	float f_min_val, 
	float f_max_val, 
	int i_cluster_samples, 
	float f_stdev, 
	Matrix<float>& x,
	vector<float>& y
) {
	// total number of training samples
	int i_total_samples = i_grid_size*i_grid_size*i_cluster_samples;
	// distance between the center of each grid cell
	float f_clust_step = (f_max_val-f_min_val)/(float)i_grid_size;

	x = Matrix<float>(2, i_total_samples);
	y = vector<float>(i_total_samples);

	int i_cur_sample = 0;
	for (int i=0; i<i_grid_size; i++) {
		for (int j=0; j<i_grid_size; j++) {
			// compute cluster centers
			float f_clust_pos_x = i*f_clust_step+f_clust_step/2;
			float f_clust_pos_y = j*f_clust_step+f_clust_step/2;
			// determine class of center
			int i_clust_class;
			if ( (i+j)%2 == 0 ) i_clust_class = 0; else i_clust_class = 1;
			if ( (i==0 && j==1) || (i==1 && j==1) || (i==2 && j==0) ) i_clust_class = 2;
			// generate random points
			for (int k=0; k<i_cluster_samples; k++) {
				x[0][i_cur_sample] = nrand()*f_stdev+f_clust_pos_x;
				x[1][i_cur_sample] = nrand()*f_stdev+f_clust_pos_y;
				y[i_cur_sample] = i_clust_class;
				i_cur_sample++;
			}
		}
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

	generate_checkerboard_data(
		i_grid_size,
		f_min_val,
		f_max_val,
		i_cluster_samples,
		f_stdev,
		x,
		y
		);

	// define classifier parameters
	int i_num_classifiers = 5;
	vector<vector<float> > vec2_params(i_num_classifiers, vector<float>());

	vec2_params[0] = vector<float>(1,0);	// bayes
	const float f_k_vals[] = {1, 11, 21, 31};
	vec2_params[1] = vector<float>(f_k_vals, f_k_vals+4);
	const float f_clust_vals[] = {2, 3, 4, 5};
	vec2_params[2] = vector<float>(f_clust_vals, f_clust_vals+4);
	const float f_dtree_depth[] = {2,5,7,10,100};
	vec2_params[3] = vector<float>(f_dtree_depth, f_dtree_depth+5);
	const float f_forest_trees[] = {10,50,100};
	vec2_params[4] = vector<float>(f_forest_trees, f_forest_trees+3);

	const char* ch_classifier_names[] = {"Bayes", "KNN", "EM", "DTree", "Forest"};
	vector<string> vec_classifier_names(ch_classifier_names, ch_classifier_names+i_num_classifiers);

	float f_grid_step = (f_max_val-f_min_val)/(float)i_grid_res;
	Matrix<float> mat_results(i_grid_res, i_grid_res);
	vector<int> index;

	// ------------------------------------------------------------
	Classifier* cls = NULL;

	for (int c=0; c<vec_classifier_names.size(); c++) {
		for (int k=0; k<vec2_params[c].size(); k++) {
			cout << vec_classifier_names[c] << ": " << vec2_params[c][k] << endl;
			// initialize classifier
			if (c == 0) {
				cls = new Bayes();
			} else if (c == 1) {
				cls = new KNN((int)vec2_params[c][k]);
			} else if (c == 2) {
				cls = new EM((int)vec2_params[c][k]);
			} else if (c == 3) {
				cls = new DTree((int)vec2_params[c][k], 1, 0);
			} else if (c == 4) {
				cls = new Forest((int)vec2_params[c][k], 0.1, 5, 1);
			}
			// train classifier
			cls->train(&x, &y, index);

			cout << "\tDrawing Grid..." << endl;
			// predict each point in the grid
			Matrix<float> mat_sample(2, 1);
			for (int i=0; i<i_grid_res; i++) {
				for (int j=0; j<i_grid_res; j++) {
					mat_sample[0][0] = i*f_grid_step;
					mat_sample[1][0] = j*f_grid_step;
					mat_results[i][j] 
						= cls->predict_point_in_matrix(&mat_sample, 0);
				}
			}

			// write results to file
			DataFile<float> df(
				string("./output_")
					+vec_classifier_names[c]
					+string("_multi_")
					+conv2str(vec2_params[c][k]),
				'\t'
				);
			df.writeDataFile(mat_results);
			delete cls;
		}
	}

	return 0;

}
