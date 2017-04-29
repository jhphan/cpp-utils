#include <iostream>
#include <miblab/matrix.h>
#include <miblab/rand.h>
#include <miblab/commandline.h>
#include <miblab/feature_selection.h>
#include <miblab/classifiers/lr.h>
#include <miblab/datafileheader.h>

using namespace std;


// compute train and test indexes for stratified cross validation
// vec2_folds is pre-allocated with 1s
void indexed_stratified_cv(int i_folds, vector<vector<int> >& vec2_index_class, vector<int>& vec_index, vector<vector<int> >& vec2_folds_train, vector<vector<int> >& vec2_folds_test) {
        int i_num_classes = vec2_index_class.size();
        vector<int> vec_base_num_samples(i_num_classes,0);
        vector<int> vec_rem_samples(i_num_classes,0);
        vector<int> vec_used_class_sizes(i_num_classes, 0);

        // compute the number of samples in each fold
        for (int i=0; i<i_num_classes; i++) {
                for (int j=0; j<vec2_index_class[i].size(); j++)
                        if (vec_index[vec2_index_class[i][j]] == 0) vec_used_class_sizes[i]++;
                vec_base_num_samples[i] = vec_used_class_sizes[i] / i_folds;
                vec_rem_samples[i] = vec_used_class_sizes[i] % i_folds;
        }

        // keep track of number of samples assigned to each fold
        vector<vector<int> > vec2_max_fold_count(i_num_classes, vector<int>(i_folds,0));
        for (int i=0; i<i_num_classes; i++) {
                for (int j=0; j<i_folds; j++) {
                        vec2_max_fold_count[i][j] = vec_base_num_samples[i];
                        if (j < vec_rem_samples[i]) vec2_max_fold_count[i][j]++;
                }
        }

        // randomly assign samples to each fold
        vector<vector<int> > vec_cur_fold_count(i_num_classes, vector<int>(i_folds,0));
        for (int j=0; j<i_num_classes; j++) {
                for (int k=0; k<vec2_index_class[j].size(); k++) {
                        if (vec_index[vec2_index_class[j][k]] == 0) {
                                int i_fold_num = (int)(drand48()*i_folds);
                                while (vec_cur_fold_count[j][i_fold_num] >= vec2_max_fold_count[j][i_fold_num])
                                        i_fold_num = (int)(drand48()*i_folds);
                                vec_cur_fold_count[j][i_fold_num]++;
                                for (int el=0; el<i_folds; el++)
                                        if (i_fold_num == el)
                                                vec2_folds_test[el][vec2_index_class[j][k]] = 0;
                                        else
                                                vec2_folds_train[el][vec2_index_class[j][k]] = 0;
                        }
                }
        }
}

void extract_features(Matrix<float>& mat_x, vector<int>& vec_sort_index, int i_feature_size, Matrix<float>& mat_out_x) {
        mat_out_x = Matrix<float>();
        for (int i=0; i<i_feature_size; i++)
                mat_out_x.push_back(mat_x[vec_sort_index[i]]);
}

int main(int argc, char* argv[]) {
	srand48(time(NULL));

	string str_data_file;

	CommandLine cl(argc, argv);

	if (!cl.getArg(str_data_file, "f")) {
		cout << "missing -f option" << endl;
		return 1;
	}

	Matrix<float> x;
	vector<float> y;

	DataFile<float> df_data(str_data_file, '\t');
	df_data.readDataFile(x);

	y = x[0];
	x = x(1,-1,-1,-1);

	int i_num_samples = y.size();

	vector<int> vec_index(i_num_samples, 0);
	vector<vector<int> > vec2_index_class;
	index_samples(&y, vec2_index_class);
	int i_num_classes = vec2_index_class.size();
	int i_num_folds = 2;

	vector<vector<int> > vec2_folds_train(i_num_folds, vector<int>(i_num_samples, 1));
	vector<vector<int> > vec2_folds_test(i_num_folds, vector<int>(i_num_samples, 1));
	indexed_stratified_cv(i_num_folds, vec2_index_class, vec_index, vec2_folds_train, vec2_folds_test);

		Classifier* cls = new LR();

		//cls->train(&x, &y, vec_index);
		cls->train(&x, &y, vec2_folds_train[0]);

		vector<float> vec_dec_vals;
		//cls->predict_points_in_matrix_scaled(&x, vec_index, vec_dec_vals);
		cls->predict_points_in_matrix_scaled(&x, vec2_folds_test[0], vec_dec_vals);

		for (int j=0; j<i_num_samples; j++) {
			if (vec2_folds_test[0][j] == 0)
				cout << vec_dec_vals[j] << "(" << y[j] << "),";
		}
		cout << endl;

		float f_auc = compute_AUC_metric(&y, vec_dec_vals, vec2_folds_test[0], vec2_index_class);
		//float f_auc = compute_AUC_metric(&y, vec_dec_vals, vec_index, vec2_index_class);
		cout << "AUC: " << f_auc << endl;
		delete cls;


/*

	// ------------------------------------------------------------
	Classifier* cls = NULL;

				cls = new LR();
			// train classifier
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
				string("./output_linear_")
					+vec_classifier_names[c]
					+string("_")
					+conv2str(vec2_params[c][k]),
				'\t'
				);
			df.writeDataFile(mat_results);

			delete cls;
		}
	}

*/
	return 0;

}
