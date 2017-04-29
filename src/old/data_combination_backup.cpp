#include <iostream>
#include <miblab/matrix.h>
#include <miblab/rand.h>
#include <miblab/commandline.h>
#include <miblab/feature_selection.h>
#include <miblab/classifiers/lr.h>
#include <miblab/datafileheader.h>
#include <miblab/stringtokenizer.h>
#include <miblab/normalize.h>

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

void load_data(string& str_data_file, Matrix<float>& mat_x, vector<float>& vec_y) {
	DataFileHeader<float> df_data(str_data_file, '\t');
	df_data.readDataFile(mat_x, true, true);

	vec_y = mat_x[0];
	mat_x = mat_x(1,-1,-1,-1);
}

// combine data from multiple datasets			
void combine_data(Matrix<float>& combine_x, vector<float>& combine_y, vector<Matrix<float> >& data_x, vector<vector<float> >& data_y, vector<vector<vector<int> > >& vec3_folds_train, int i_fold_num) {
	int i_num_datasets = data_x.size();
	int i_num_features = data_x[0].getHeight();
	// count number of samples in each dataset
	int i_total_samples = 0;
	for (int i=0; i<i_num_datasets; i++) {
		int i_num_samples = data_x[i].getWidth();
		for (int j=0; j<i_num_samples; j++)
			if (vec3_folds_train[i][i_fold_num][j] == 0)
				i_total_samples++;
	}
	// allocate memory of combined data
	combine_x = Matrix<float>(i_num_features, i_total_samples);
	combine_y = vector<float>(i_total_samples);
	int i_cur_sample = 0;
	for (int i=0; i<i_num_datasets; i++) {
		int i_num_samples = data_x[i].getWidth();
		for (int j=0; j<i_num_samples; j++)
			if (vec3_folds_train[i][i_fold_num][j] == 0) {
				combine_y[i_cur_sample] = data_y[i][j];
				for (int k=0; k<i_num_features; k++)
					combine_x[k][i_cur_sample] = data_x[i][k][j];
				i_cur_sample++;
			}
	}
}

int main(int argc, char* argv[]) {
	srand48(time(NULL));

	string str_data_file_list;
	string str_data_file_val;
	vector<string> vec_data_file_list;
	
	CommandLine cl(argc, argv);

	if (!cl.getArg(str_data_file_list, "f")) {
		cerr << "missing -f option" << endl;
		return 1;
	}
	StringTokenizer st(str_data_file_list,',');
	vec_data_file_list = st.split();
	int i_num_datasets = vec_data_file_list.size();

	if (!cl.getArg(str_data_file_val, "v")) {
		cerr << "missing -v option" << endl;
		return 1;
	}

	vector<Matrix<float> > data_x(i_num_datasets);
	vector<vector<float> > data_y(i_num_datasets);
	Matrix<float> data_val_x;
	vector<float> data_val_y;

	// load data
	for (int i=0; i<i_num_datasets; i++) {
		//cout << vec_data_file_list[i] << endl;
		load_data(vec_data_file_list[i], data_x[i], data_y[i]);
	}

	load_data(str_data_file_val, data_val_x, data_val_y);
	vector<vector<int> > vec2_val_index_class;
	index_samples(&data_val_y, vec2_val_index_class);

	int i_num_iterations = 100;
	int i_num_folds = 3;
	int i_max_features = 50;
	int i_num_features = data_x[0].getHeight();

	//Matrix<Matrix<vector<float> > > results(i_num_iterations, i_num_folds, Matrix<vector<float> >(i_num_datasets,4,vector<float>(i_max_features,0)));
	vector<int> vec_val_index;

	for (int i=0; i<i_num_iterations; i++) {
		// partition into folds
		vector<vector<int> > vec2_index(i_num_datasets);
		vector<vector<vector<int> > > vec3_index_class(i_num_datasets);
		vector<vector<vector<int> > > vec3_folds_train(i_num_datasets);
		vector<vector<vector<int> > > vec3_folds_test(i_num_datasets);
		for (int ds=0; ds<i_num_datasets; ds++) {
			index_samples(&data_y[ds], vec3_index_class[ds]);

			int i_num_samples = data_y[ds].size();
			vec2_index[ds] = vector<int>(i_num_samples,0);
			vec3_folds_train[ds] = vector<vector<int> >(i_num_folds, vector<int>(i_num_samples, 1));
			vec3_folds_test[ds] = vector<vector<int> >(i_num_folds, vector<int>(i_num_samples, 1));
			indexed_stratified_cv(i_num_folds, vec3_index_class[ds], vec2_index[ds], vec3_folds_train[ds], vec3_folds_test[ds]);
		}

		for (int j=0; j<i_num_folds; j++) {
			vector<vector<int> > vec2_feature_index(i_num_datasets);
			vector<vector<float> > vec2_scores(i_num_datasets);
			vector<vector<int> > vec2_feature_ranks(i_num_datasets);

			// $$ <quantile>
			vector<Matrix<float> > norm_data_x(i_num_datasets);
			vector<vector<float> > quant_ref(i_num_datasets);
			vector<Matrix<float> > norm_val_data_x(i_num_datasets);
			// $$ </quantile>
			
			// combine data
			Matrix<float> combine_x;
			vector<float> combine_y;
			combine_data(combine_x, combine_y, data_x, data_y, vec3_folds_train, j);
			vector<vector<int> > vec2_combine_index_class;
			index_samples(&combine_y, vec2_combine_index_class);
			vector<int> vec_combine_index(combine_y.size(),0);
			vector<float> vec_combine_scores;
			vector<int> vec_combine_sort_index;

			// ### select features individually, train individually
			for (int ds=0; ds<i_num_datasets; ds++) {

				// $$ <quantile>
					// create a quantile reference for the training portion of each dataset
					//cout << "quantile reference" << endl;
					quantile_reference(data_x[ds], vec3_folds_train[ds][j], quant_ref[ds]);
					// quantile normalize
					//cout << "quantile normalize 1" << endl;
					quantile_normalize_matrix(data_x[ds], norm_data_x[ds], vec3_folds_train[ds][j], quant_ref[ds]);
					// quantile normalize the validation data
					//cout << "quantile normalize 2" << endl;
					quantile_normalize_matrix(data_val_x, norm_val_data_x[ds], vec_val_index, quant_ref[ds]);
					//cout << "done" << endl;
				// $$ </quantile>

				// $$ <quantile>
					t_test_index(norm_data_x[ds], vec3_folds_train[ds][j], vec3_index_class[ds], vec2_scores[ds], vec2_feature_index[ds]);
					// $$ -----
					//fold_change_index(data_x[ds], vec3_folds_train[ds][j], vec3_index_class[ds], vec2_scores[ds], vec2_feature_index[ds]);
					//t_test_index(data_x[ds], vec3_folds_train[ds][j], vec3_index_class[ds], vec2_scores[ds], vec2_feature_index[ds]);
				// $$ </quantile>

				compute_ranks(vec2_feature_index[ds], vec2_feature_ranks[ds]);
				for (int fs=1; fs<=i_max_features; fs++) {

					Matrix<float> sub_x;
					// && <quantile>
						extract_features(norm_data_x[ds], vec2_feature_index[ds], fs, sub_x);
						// $$ -----
						//extract_features(data_x[ds], vec2_feature_index[ds], fs, sub_x);
					// && </quantile>

					// classify with individually selected features
					Classifier* cls = new LR();
					cls->train(&sub_x, &data_y[ds], vec3_folds_train[ds][j]);
					//vector<float> vec_dec_vals;
					//cls->predict_points_in_matrix_scaled(&sub_x, vec3_folds_test[ds][j], vec_dec_vals);
					//float f_auc = compute_AUC_metric(&data_y[ds], vec_dec_vals, vec3_folds_test[ds][j], vec3_index_class[ds]);
					//results[i][j][ds][0][fs-1] = f_auc;
					//cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t0\t0\t" << f_auc << endl;

					// classify external validation data
					Matrix<float> sub_val_x;
					// $$ <quantile>
						extract_features(norm_val_data_x[ds], vec2_feature_index[ds], fs, sub_val_x);
						// -----
						//extract_features(data_val_x, vec2_feature_index[ds], fs, sub_val_x);
					// $$ </quantile>
					vector<float> vec_val_dec_vals;
					cls->predict_points_in_matrix_scaled(&sub_val_x, vec_val_index, vec_val_dec_vals);
					float f_val_auc = compute_AUC_metric(&data_val_y, vec_val_dec_vals, vec_val_index, vec2_val_index_class);
					cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t0\t" << f_val_auc << endl;
				}
			}

			// #### rank-combine the features selected, train individually
			vector<int> vec_combined_ranks(i_num_features);
			for (int fs=0; fs<i_num_features; fs++) {
				for (int ds=0; ds<i_num_datasets; ds++)
					vec_combined_ranks[fs] += vec2_feature_ranks[ds][fs];
			}
			vector<int> vec_combined_index;
			quicksort_i(vec_combined_ranks, vec_combined_index, SORT_ASCENDING);

			for (int ds=0; ds<i_num_datasets; ds++) {
				for (int fs=1; fs<=i_max_features; fs++) {
					Matrix<float> sub_x;
					// $$ <quantile>
						extract_features(norm_data_x[ds], vec_combined_index, fs, sub_x);
						// -----
						//extract_features(data_x[ds], vec_combined_index, fs, sub_x);
					// $$ </quantile>
					// classify with individually selected features
					Classifier* cls = new LR();
					cls->train(&sub_x, &data_y[ds], vec3_folds_train[ds][j]);
					//vector<float> vec_dec_vals;
					//cls->predict_points_in_matrix_scaled(&sub_x, vec3_folds_test[ds][j], vec_dec_vals);
					//float f_auc = compute_AUC_metric(&data_y[ds], vec_dec_vals, vec3_folds_test[ds][j], vec3_index_class[ds]);
					//results[i][j][ds][1][fs-1] = f_auc;
					//cout << "1: iteration: " << i << ",fold: " << j << ",ds: " << ds << ",fs: " << fs << " > " << f_auc << endl;
					//cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t1\t0\t" << f_auc << endl;

					Matrix<float> sub_val_x;
					// $$ <quantile>
						extract_features(norm_val_data_x[ds], vec_combined_index, fs, sub_val_x);
						// -----
						//extract_features(data_val_x, vec_combined_index, fs, sub_val_x);
					// $$ </quantile>
					vector<float> vec_val_dec_vals;
					cls->predict_points_in_matrix_scaled(&sub_val_x, vec_val_index, vec_val_dec_vals);
					float f_val_auc = compute_AUC_metric(&data_val_y, vec_val_dec_vals, vec_val_index, vec2_val_index_class);
					cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t1\t" << f_val_auc << endl;
				}
			}
			
			// ### combine samples and select features, train individually
			Matrix<float> combine_x;
			vector<float> combine_y;
			combine_data(combine_x, combine_y, data_x, data_y, vec3_folds_train, j);
			vector<vector<int> > vec2_combined_index_class;
			index_samples(&combine_y, vec2_combined_index_class);
			vector<int> vec_comb_index(combine_y.size(),0);
			vector<float> vec_comb_scores;
			vector<int> vec_comb_sort_index;

			// $$ <quantile>
				Matrix<float> combine_norm_x;
				vector<float> combine_quant_ref;
				//Matrix<float> combine_val_norm_x;
				quantile_reference(combine_x, vec_comb_index, combine_quant_ref);
				// quantile normalize
				quantile_normalize_matrix(combine_x, combine_norm_x, vec_comb_index, combine_quant_ref);
				// quantile normalize the validation data
				//quantile_normalize_matrix(data_val_x, combine_val_norm_x, vec_val_index, combine_quant_ref);
				// -----
			// $$ </quantile>

			//fold_change_index(combine_x, vec_comb_index, vec2_combined_index_class, vec_comb_scores, vec_comb_sort_index);
			t_test_index(combine_norm_x, vec_comb_index, vec2_combined_index_class, vec_comb_scores, vec_comb_sort_index);

			for (int ds=0; ds<i_num_datasets; ds++) {
				for (int fs=1; fs<=i_max_features; fs++) {
					Matrix<float> sub_x;
					//extract_features(data_x[ds], vec_comb_sort_index, fs, sub_x);
					extract_features(norm_data_x[ds], vec_comb_sort_index, fs, sub_x);
					// classify with individually selected features
					Classifier* cls = new LR();
					cls->train(&sub_x, &data_y[ds], vec3_folds_train[ds][j]);
					//vector<float> vec_dec_vals;
					//cls->predict_points_in_matrix_scaled(&sub_x, vec3_folds_test[ds][j], vec_dec_vals);
					//float f_auc = compute_AUC_metric(&data_y[ds], vec_dec_vals, vec3_folds_test[ds][j], vec3_index_class[ds]);
					//results[i][j][ds][2][fs-1] = f_auc;
					//cout << "2: iteration: " << i << ",fold: " << j << ",ds: " << ds << ",fs: " << fs << " > " << f_auc << endl;
					//cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t2\t0\t" << f_auc << endl;

					Matrix<float> sub_val_x;
					extract_features(norm_val_data_x[ds], vec_comb_sort_index, fs, sub_val_x);
					vector<float> vec_val_dec_vals;
					cls->predict_points_in_matrix_scaled(&sub_val_x, vec_val_index, vec_val_dec_vals);
					float f_val_auc = compute_AUC_metric(&data_val_y, vec_val_dec_vals, vec_val_index, vec2_val_index_class);
					cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t2\t" << f_val_auc << endl;
				}
			}

			// ### quantile normalize the combined data
		}
	}

	// print everything out

	return 0;
}
