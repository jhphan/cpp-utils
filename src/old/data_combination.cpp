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

void usage() {
	cout << "data_combination:" << endl;
	cout << "\t-data [list of data files, comma separated]" << endl;
	cout << "\t-val [validation data]" << endl;
	cout << "\t-i [number of iterations]" << endl;
	cout << "\f-f [number of folds]" << endl;
	cout << "\t-fs [max feature size]" << endl;
	cout << "\t-fsm [feature selection method]" << endl;
	cout << "\t-cls [classification method]" << endl;
}

void select_features(int i_fs_method, Matrix<float>& mat_data, vector<float>& vec_data_y, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<int>& vec_feature_index, int i_max_features) {
	vector<float> vec_scores;
	if (i_fs_method == 0) {
		t_test_index(mat_data, vec_index, vec2_index_class, vec_scores, vec_feature_index);
	} else if (i_fs_method == 1) {
		fold_change_index(mat_data, vec_index, vec2_index_class, vec_scores, vec_feature_index);
	} else if (i_fs_method == 2) {
		bool b_exact = false;
		if (mat_data.getWidth() < 50) b_exact = true;
		ranksum_index(mat_data, vec_index, vec2_index_class, vec_scores, vec_feature_index, b_exact);
	} else if (i_fs_method == 3) {
		mrmr_index(mat_data, vec_data_y, vec_index, vec_scores, vec_feature_index, i_max_features, false);
	} else if (i_fs_method == 4) {
		mrmr_index(mat_data, vec_data_y, vec_index, vec_scores, vec_feature_index, i_max_features, true);
	}
}

float classify(int i_cls_method, Matrix<float>& train_data_x, vector<float>& train_data_y, vector<int>& vec_train_index, Matrix<float>& test_data_x, vector<float>& test_data_y, vector<int>& vec_test_index, vector<vector<int> >& vec2_test_index_class, vector<int>& vec_feature_index, int i_fs) {
	Matrix<float> sub_x;
	extract_features(train_data_x, vec_feature_index, i_fs, sub_x);

	// classify with individually selected features
	Classifier* cls = new LR();
	cls->train(&sub_x, &train_data_y, vec_train_index);

	Matrix<float> sub_val_x;
	extract_features(test_data_x, vec_feature_index, i_fs, sub_val_x);

	vector<float> vec_val_dec_vals;
	cls->predict_points_in_matrix_scaled(&sub_val_x, vec_test_index, vec_val_dec_vals);

	float f_val_auc = compute_AUC_metric(&test_data_y, vec_val_dec_vals, vec_test_index, vec2_test_index_class);
	return f_val_auc;
}

int main(int argc, char* argv[]) {
	srand48(time(NULL));

	string str_data_file_list;
	string str_data_file_val;
	vector<string> vec_data_file_list;

	int i_num_iterations = 10;
	int i_num_folds = 3;
	int i_max_features = 50;
	int i_fs_method = 0;
	int i_cls_method = 0;	

	CommandLine cl(argc, argv);

	if (!cl.getArg(str_data_file_list, "data")) {
		usage();
		return 1;
	}
	StringTokenizer st(str_data_file_list,',');
	vec_data_file_list = st.split();
	int i_num_datasets = vec_data_file_list.size();

	if (!cl.getArg(str_data_file_val, "val")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_num_iterations, "i")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_num_folds, "f")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_max_features, "fs")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_fs_method, "fsm")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_cls_method, "cls")) {
		usage();
		return 1;
	}

	// individual datasets
	vector<Matrix<float> > data_x(i_num_datasets);
	vector<vector<float> > data_y(i_num_datasets);
	vector<vector<vector<int> > > vec3_index_class(i_num_datasets);
	vector<vector<int> > vec2_index(i_num_datasets);	// use all samples of data
	vector<int> vec_num_samples(i_num_datasets);
	vector<float> vec_sample_weight(i_num_datasets);
	int i_total_samples = 0;

	for (int i=0; i<i_num_datasets; i++) {
		//cout << vec_data_file_list[i] << endl;
		load_data(vec_data_file_list[i], data_x[i], data_y[i]);
		index_samples(&data_y[i], vec3_index_class[i]);
		vec_num_samples[i] = data_y[i].size();
		i_total_samples += vec_num_samples[i];
		vec2_index[i] = vector<int>(vec_num_samples[i],0);	// use all samples of dataset
	}
	for (int i=0; i<i_num_datasets; i++)
		vec_sample_weight[i] = vec_num_samples[i]/(float)i_total_samples;

	// validation data
	Matrix<float> data_val_x;
	vector<float> data_val_y;
	vector<vector<int> > vec2_val_index_class;
	vector<int> vec_val_index;

	load_data(str_data_file_val, data_val_x, data_val_y);
	index_samples(&data_val_y, vec2_val_index_class);
	vec_val_index = vector<int>(data_val_y.size(),0);

	int i_num_features = data_x[0].getHeight();

	for (int i=0; i<i_num_iterations; i++) {
		// for each iteration, partition into random folds
		vector<vector<vector<int> > > vec3_folds_train(i_num_datasets);
		vector<vector<vector<int> > > vec3_folds_test(i_num_datasets);
		for (int ds=0; ds<i_num_datasets; ds++) {
			int i_num_samples = data_y[ds].size();
			vec3_folds_train[ds] = vector<vector<int> >(i_num_folds, vector<int>(i_num_samples, 1));
			vec3_folds_test[ds] = vector<vector<int> >(i_num_folds, vector<int>(i_num_samples, 1));
			indexed_stratified_cv(i_num_folds, vec3_index_class[ds], vec2_index[ds], vec3_folds_train[ds], vec3_folds_test[ds]);
		}

		for (int j=0; j<i_num_folds; j++) {
			// variables for individual dataset feature selection
			vector<vector<int> > vec2_feature_index(i_num_datasets);
			vector<vector<int> > vec2_feature_ranks(i_num_datasets);

			// variables for individual dataset normalization
			vector<Matrix<float> > norm_data_x(i_num_datasets);
			vector<vector<float> > quant_ref(i_num_datasets);
			// validation data normalized to each individual dataset
			vector<Matrix<float> > norm_data_val_x(i_num_datasets);
			
			// sample-combine training folds of each dataset
			Matrix<float> data_scomb_x;
			vector<float> data_scomb_y;
				combine_data(data_scomb_x, data_scomb_y, data_x, data_y, vec3_folds_train, j);
			vector<vector<int> > vec2_scomb_index_class;
				index_samples(&data_scomb_y, vec2_scomb_index_class);
			vector<int> vec_scomb_index(data_scomb_y.size(),0);
			vector<int> vec_scomb_feature_index;
			Matrix<float> norm_data_scomb_x;
			vector<float> scomb_quant_ref;
			Matrix<float> norm_data_scomb_val_x;

			// variables for rank-combined results
			vector<float> vec_rcomb_average_ranks(i_num_features);
			vector<int> vec_rcomb_feature_index;

			//  $$$$$$ create quantile normalized data
			for (int ds=0; ds<i_num_datasets; ds++) {
				// create a quantile reference for the training portion of each dataset
				quantile_reference(data_x[ds], vec3_folds_train[ds][j], quant_ref[ds]);
				quantile_normalize_matrix(data_x[ds], norm_data_x[ds], vec3_folds_train[ds][j], quant_ref[ds]);
				// quantile normalize the validation data to each individual dataset
				quantile_normalize_matrix(data_val_x, norm_data_val_x[ds], vec_val_index, quant_ref[ds]);
			}
			// quantile the scomb data
			quantile_reference(data_scomb_x, vec_scomb_index, scomb_quant_ref);
			quantile_normalize_matrix(data_scomb_x, norm_data_scomb_x, vec_scomb_index, scomb_quant_ref);
			// quantile normalize validation data to scomb data
			quantile_normalize_matrix(data_val_x, norm_data_scomb_val_x, vec_val_index, scomb_quant_ref);

			// $$$$$$ feature selection
			for (int ds=0; ds<i_num_datasets; ds++) {
				select_features(
					i_fs_method,
					norm_data_x[ds],
					data_y[ds],
					vec3_folds_train[ds][j],
					vec3_index_class[ds],
					vec2_feature_index[ds],
					i_max_features
				);
				compute_ranks(vec2_feature_index[ds], vec2_feature_ranks[ds]);	
			}
			// -- rank-combine features
			for (int fs=0; fs<i_num_features; fs++)
				for (int ds=0; ds<i_num_datasets; ds++)
					vec_rcomb_average_ranks[fs] += vec2_feature_ranks[ds][fs]*vec_sample_weight[ds];
			quicksort_i(vec_rcomb_average_ranks, vec_rcomb_feature_index, SORT_ASCENDING);
			// -- sample-combined features
			select_features(
				i_fs_method,
				norm_data_scomb_x,
				data_scomb_y,
				vec_scomb_index,
				vec2_scomb_index_class,
				vec_scomb_feature_index,
				i_max_features
			);
			
			// $$$$$$ classification
			// ### select features individually, train individually
			for (int fs=1; fs<=i_max_features; fs++) {
				for (int ds=0; ds<i_num_datasets; ds++) {
					// select feature individually, train individually
					float f_val = classify(
							i_cls_method,
							norm_data_x[ds],
							data_y[ds],
							vec3_folds_train[ds][j],
							norm_data_val_x[ds],
							data_val_y,
							vec_val_index,
							vec2_val_index_class,
							vec2_feature_index[ds],
							fs
						);
					cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t0\t" << f_val << endl;
					
					// rank-combine features, train individually
					f_val = classify(
							i_cls_method,
							norm_data_x[ds],
							data_y[ds],
							vec3_folds_train[ds][j],
							norm_data_val_x[ds],
							data_val_y,
							vec_val_index,
							vec2_val_index_class,
							vec_rcomb_feature_index,
							fs
						);
					cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t1\t" << f_val << endl;
					
					// sample-combined features, train individually
					f_val = classify(
							i_cls_method,
							norm_data_x[ds],
							data_y[ds],
							vec3_folds_train[ds][j],
							norm_data_val_x[ds],
							data_val_y,
							vec_val_index,
							vec2_val_index_class,
							vec_scomb_feature_index,
							fs
						);
					cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t2\t" << f_val << endl;

					// individual features, train with combined data
					f_val = classify(
							i_cls_method,
							norm_data_scomb_x,
							data_scomb_y,
							vec_scomb_index,
							norm_data_scomb_val_x,
							data_val_y,
							vec_val_index,
							vec2_val_index_class,
							vec2_feature_index[ds],
							fs
						);
					cout << i << "\t" << j << "\t" << ds << "\t" << fs << "\t3\t" << f_val << endl;

				}
				
				// rank-combine features, train with combined data
				float f_val = classify(
						i_cls_method,
						norm_data_scomb_x,
						data_scomb_y,
						vec_scomb_index,
						norm_data_scomb_val_x,
						data_val_y,
						vec_val_index,
						vec2_val_index_class,
						vec_rcomb_feature_index,
						fs
					);
				cout << i << "\t" << j << "\t" << -1 << "\t" << fs << "\t4\t" << f_val << endl;
				
				// sample-combined features, train with combined data
				f_val = classify(
						i_cls_method,
						norm_data_scomb_x,
						data_scomb_y,
						vec_scomb_index,
						norm_data_scomb_val_x,
						data_val_y,
						vec_val_index,
						vec2_val_index_class,
						vec_scomb_feature_index,
						fs
					);
				cout << i << "\t" << j << "\t" << -1 << "\t" << fs << "\t5\t" << f_val << endl;
			}
		}
	}

	return 0;
}
