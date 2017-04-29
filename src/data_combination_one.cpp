#include <iostream>
#include <miblab/matrix.h>
#include <miblab/rand.h>
#include <miblab/commandline.h>
#include <miblab/datafileheader.h>
#include <miblab/stringtokenizer.h>
#include <miblab/feature_selection.h>
#include <miblab/classifiers/classifier_shared.h>
#include <miblab/classifiers/classifier.h>
#include <miblab/classifiers/bayes.h>
#include <miblab/classifiers/lr.h>
#include <miblab/classifiers/svm.h>

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

// randomly permute data
void permute_data(Matrix<float>& mat_x, int i_num_passes) {
	int i_num_features = mat_x.getHeight();
	int i_num_samples = mat_x.getWidth();
	
	for (int i=0; i<i_num_features; i++) {
		for (int k=0; k<i_num_passes; k++)
			for (int j=0; j<i_num_samples; j++) {
				// select random number
				int i_num = (int)(drand48()*i_num_samples);
				// switch sample j with i_num
				float f_temp = mat_x[i][j];
				mat_x[i][j] = mat_x[i][i_num];
				mat_x[i][i_num] = f_temp;
			}
	}
}

// combine data from multiple datasets			
void combine_samples(
	Matrix<float>& combine_x,
	vector<float>& combine_y,
	vector<Matrix<float> >& data_x,
	vector<vector<float> >& data_y
) {
	int i_num_datasets = data_x.size();
	int i_num_features = data_x[0].getHeight();

	// count number of samples in each dataset
	int i_total_samples = 0;
	for (int i=0; i<i_num_datasets; i++) {
		int i_num_samples = data_x[i].getWidth();
		i_total_samples += i_num_samples;
	}
	// allocate memory of combined data
	combine_x = Matrix<float>(i_num_features, i_total_samples);
	combine_y = vector<float>(i_total_samples);
	
	// combine data
	int i_cur_sample = 0;
	for (int i=0; i<i_num_datasets; i++) {
		int i_num_samples = data_x[i].getWidth();
		for (int j=0; j<i_num_samples; j++) {
			combine_y[i_cur_sample] = data_y[i][j];
			for (int k=0; k<i_num_features; k++)
				combine_x[k][i_cur_sample] = data_x[i][k][j];
			i_cur_sample++;
		}
	}
}

// combine data from multiple datasets			
void combine_data(
	vector<vector<int> >& vec2_scomb_folds_train,
	vector<vector<int> >& vec2_scomb_folds_test,
	vector<Matrix<float> >& data_x,
	vector<vector<float> >& data_y,
	vector<vector<vector<int> > >& vec3_folds_train,
	vector<vector<vector<int> > >& vec3_folds_test
) {
	int i_num_datasets = data_x.size();
	int i_num_features = data_x[0].getHeight();
	int i_num_folds = vec3_folds_train[0].size();

	// count number of samples in each dataset
	int i_total_samples = 0;
	for (int i=0; i<i_num_datasets; i++) {
		int i_num_samples = data_x[i].getWidth();
		i_total_samples += i_num_samples;
	}
	// allocate memory of combined data
	vec2_scomb_folds_train = vector<vector<int> >(i_num_folds, vector<int>(i_total_samples,0));
	vec2_scomb_folds_test = vector<vector<int> >(i_num_folds, vector<int>(i_total_samples,0));

	// combine data
	int i_cur_sample = 0;
	for (int i=0; i<i_num_datasets; i++) {
		int i_num_samples = data_x[i].getWidth();
		for (int j=0; j<i_num_samples; j++) {
			for (int k=0; k<i_num_folds; k++) {
				vec2_scomb_folds_train[k][i_cur_sample] = vec3_folds_train[i][k][j];
				vec2_scomb_folds_test[k][i_cur_sample] = vec3_folds_test[i][k][j];
			}
			i_cur_sample++;
		}
	}
}

void usage() {
	cout << "data_combination:" << endl;
	cout << "\t-data [list of data files, comma separated]" << endl;
	cout << "\t-f [number of folds]" << endl;
	cout << "\t-fs [max feature size]" << endl;
	cout << "\t-cls [classification method]" << endl;
}

void select_features(int i_fs_method, Matrix<float>& mat_data, vector<float>& vec_data_y, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<int>& vec_feature_index, vector<float>& vec_scores, int i_max_features, Matrix<vector<float> >& mat_scores) {
	if (i_fs_method == 0) {
		t_test_index(mat_data, vec_index, vec2_index_class, vec_scores, vec_feature_index);
	} else if (i_fs_method == 1) {
		fold_change_index(mat_data, vec_index, vec2_index_class, vec_scores, vec_feature_index);
	} else if (i_fs_method == 2) {
		bool b_exact = false;
		if (mat_data.getWidth() < 30) b_exact = true;
		ranksum_index(mat_data, vec_index, vec2_index_class, vec_scores, vec_feature_index, b_exact);
	} else if (i_fs_method == 3) {
		mrmr_index(mat_data, vec_data_y, vec_index, vec_scores, vec_feature_index, i_max_features, false);
	} else if (i_fs_method == 4) {
		mrmr_index(mat_data, vec_data_y, vec_index, vec_scores, vec_feature_index, i_max_features, true);
	} else if (i_fs_method == 5) {
		// sam
		sam_index(mat_data, vec_index, vec2_index_class, vec_scores, vec_feature_index);
	} else if (i_fs_method == 6) {
		// rank prod
		rankprod_index_quick(mat_scores, vec_index, vec2_index_class, vec_scores, vec_feature_index);
	}
}

float classify(
	int i_cls_method,
	Matrix<float>& train_data_x,
	vector<float>& train_data_y,
	vector<int>& vec_train_index,
	Matrix<float>& test_data_x,
	vector<float>& test_data_y,
	vector<int>& vec_test_index,
	vector<vector<int> >& vec2_test_index_class,
	vector<int>& vec_feature_index,
	int i_fs
) {
	Classifier* cls = NULL;
	if (i_cls_method == 0) { // logistic regression
		cls = new LR();
	} else if (i_cls_method == 1) { // diagonal quadratic discriminant
		cls = new Bayes(false, 1);
	} else if (i_cls_method == 2) { // DLDA
		cls = new Bayes(true, 1);
	} else if (i_cls_method == 3) {	// linear SVM
		cls = new SVM();
	}

	Matrix<float> sub_x;
	extract_features(train_data_x, vec_feature_index, i_fs, sub_x);
	cls->train(&sub_x, &train_data_y, vec_train_index);

	Matrix<float> sub_val_x;
	extract_features(test_data_x, vec_feature_index, i_fs, sub_val_x);
	vector<float> vec_val_dec_vals;
	cls->predict_points_in_matrix_scaled(&sub_val_x, vec_test_index, vec_val_dec_vals);

	float f_val_auc = compute_AUC_metric(&test_data_y, vec_val_dec_vals, vec_test_index, vec2_test_index_class);

	delete cls;
	return f_val_auc;
}

int optimize_fs_method(
		Matrix<float>& data_train_x,
		vector<float>& data_train_y,
		vector<vector<int> >& vec2_train_index,
		vector<vector<int> >& vec2_train_index_class,
		Matrix<float>& data_test_x,
		vector<float>& data_test_y,
		vector<vector<int> >& vec2_test_index,
		vector<vector<int> >& vec2_test_index_class,
		int i_max_features,
		int i_cls_method,
		int i_num_folds,
		vector<vector<int> >& vec2_opt_feature_index,
		Matrix<vector<float> >& mat_scores
) {

	vector<vector<vector<int> > > vec3_feature_index(i_num_folds, vector<vector<int> >(7));
	vector<vector<vector<float> > > vec3_perf(i_num_folds, vector<vector<float> >(7, vector<float>(i_max_features, 0)));

	vector<float> vec_scores;

	// select feature for each fold, using each of the 5 different methods
	for (int i=0; i<i_num_folds; i++) {
		for (int j=0; j<7; j++) {
			select_features(
				j,
				data_train_x,
				data_train_y,
				vec2_train_index[i],
				vec2_train_index_class,
				vec3_feature_index[i][j],
				vec_scores,
				i_max_features,
				mat_scores
			);
		}
	}
	
	vector<float> vec_avg_perf(7,0);

	// compute classification performance for each set of selected features
	for (int i=0; i<i_num_folds; i++) {
		for (int j=0; j<7; j++) {
			for (int fs=1; fs<=i_max_features; fs++) {
				float f_val = classify(
					i_cls_method,
					data_train_x,
					data_train_y,
					vec2_train_index[i],
					data_test_x,
					data_test_y,
					vec2_test_index[i],
					vec2_test_index_class,
					vec3_feature_index[i][j],
					fs
				);
				vec3_perf[i][j][fs-1] = f_val;
				vec_avg_perf[j] += f_val;
			}
		}
	}

	//vector<float> vec_smallest(5,0);
	//int i_best_fs_method = 0;
	//for (int i=0; i<5; i++) {
	//	for (int fs=0; fs<i_max_features; fs++) {
	//		float f_avg_perf = 0;
	//		for (int j=0; j<i_num_folds; j++)
	//			f_avg_perf += vec3_perf[j][i][fs];
	//		f_avg_perf /= (float)i_num_folds;
	//		float f_perf_dist = (1-f_avg_perf)*(1-f_avg_perf)+fs*fs/16.0;
	//		if (f_perf_dist < vec_smallest[i]) vec_smallest[i] = f_perf_dist;
	//	}
	//	if (vec_smallest[i] < vec_smallest[i_best_fs_method]) i_best_fs_method = i;
	//}

	// average the performance of each feature selection method over all folds and all feature sizes
	// also find the max performing fs method
	int i_max_fs_method = 0;
	for (int i=0; i<7; i++) {
		vec_avg_perf[i] /= (float)(i_num_folds*i_max_features);
		if (vec_avg_perf[i] > vec_avg_perf[i_max_fs_method]) i_max_fs_method = i;
	}

	cerr << "optimal fs method: " << i_max_fs_method << ", avg perf: " << vec_avg_perf[i_max_fs_method] << endl;
	//cerr << "optimal fs method: " << i_best_fs_method << ", avg perf: " << vec_smallest[i_best_fs_method] << endl;

	// return optimal feature lists for each fold
	vec2_opt_feature_index = vector<vector<int> >(i_num_folds);
	for (int i=0; i<i_num_folds; i++)
		vec2_opt_feature_index[i] = vec3_feature_index[i][i_max_fs_method];

	return i_max_fs_method;
}

int main(int argc, char* argv[]) {
	srand48(time(NULL));

	string str_data_file_list;
	vector<string> vec_data_file_list;

	int i_num_folds = 3;
	int i_max_features = 50;
	int i_cls_method = 0;	

	CommandLine cl(argc, argv);

	if (!cl.getArg(str_data_file_list, "data")) {
		usage();
		return 1;
	}
	StringTokenizer st(str_data_file_list,',');
	vec_data_file_list = st.split();
	int i_num_datasets = vec_data_file_list.size();

	if (!cl.getArg(i_num_folds, "f")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_max_features, "fs")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_cls_method, "cls")) {
		usage();
		return 1;
	}

	// individual datasets
	vector<Matrix<float> > data_x(i_num_datasets);
	vector<Matrix<float> > data_perm_x(i_num_datasets);
	vector<vector<float> > data_y(i_num_datasets);
	vector<vector<vector<int> > > vec3_index_class(i_num_datasets);
	vector<vector<int> > vec2_index(i_num_datasets);	// use all samples of data
	vector<int> vec_num_samples(i_num_datasets);
	vector<float> vec_sample_weight(i_num_datasets);
	int i_total_samples = 0;
	vector<Matrix<vector<float> > > data_rankprod_precompute(i_num_datasets);
	vector<Matrix<vector<float> > > data_perm_rankprod_precompute(i_num_datasets);

	for (int i=0; i<i_num_datasets; i++) {
		load_data(vec_data_file_list[i], data_x[i], data_y[i]);
		data_perm_x[i] = data_x[i];
		permute_data(data_perm_x[i], 2);
		index_samples(&data_y[i], vec3_index_class[i]);
		vec_num_samples[i] = data_y[i].size();
		i_total_samples += vec_num_samples[i];
		vec2_index[i] = vector<int>(vec_num_samples[i],0);	// use all samples of dataset
		rankprod_precompute(data_x[i], vec3_index_class[i], data_rankprod_precompute[i]);
		rankprod_precompute(data_perm_x[i], vec3_index_class[i], data_perm_rankprod_precompute[i]);
	}
	for (int i=0; i<i_num_datasets; i++) {
		vec_sample_weight[i] = vec_num_samples[i]/(float)i_total_samples;
		cerr << "sample weight: " << i << ", " << vec_sample_weight[i] << endl;
	}

	int i_num_features = data_x[0].getHeight();

	//for (int i=0; i<i_num_iterations; i++) {
		// for each iteration, partition into random folds
		vector<vector<vector<int> > > vec3_folds_train(i_num_datasets);
		vector<vector<vector<int> > > vec3_folds_test(i_num_datasets);
		for (int ds=0; ds<i_num_datasets; ds++) {
			int i_num_samples = data_y[ds].size();
			vec3_folds_train[ds] = vector<vector<int> >(i_num_folds, vector<int>(i_num_samples, 1));
			vec3_folds_test[ds] = vector<vector<int> >(i_num_folds, vector<int>(i_num_samples, 1));
			indexed_stratified_cv(i_num_folds, vec3_index_class[ds], vec2_index[ds], vec3_folds_train[ds], vec3_folds_test[ds]);
		}
		
		vector<vector<int> > vec2_feature_index(i_num_datasets);
		vector<vector<float> > vec2_feature_score(i_num_datasets);
		vector<vector<int> > vec2_perm_feature_index(i_num_datasets);
		vector<vector<float> > vec2_perm_feature_score(i_num_datasets);
		vector<vector<vector<int> > > vec3_opt_feature_index(i_num_datasets);
		vector<int> vec_opt_fs_method(i_num_datasets, 0);
		for (int ds=0; ds<i_num_datasets; ds++) {
			cerr << "ds: " << ds << ": optimizing..." << endl;
			vec_opt_fs_method[ds] = optimize_fs_method(
				data_x[ds],
				data_y[ds],
				vec3_folds_train[ds],
				vec3_index_class[ds],
				data_x[ds],
				data_y[ds],
				vec3_folds_test[ds],
				vec3_index_class[ds],
				i_max_features,
				i_cls_method,
				i_num_folds,
				vec3_opt_feature_index[ds],
				data_rankprod_precompute[ds]
			);
			select_features(
				vec_opt_fs_method[ds],
				data_x[ds],
				data_y[ds],
				vec2_index[ds],
				vec3_index_class[ds],
				vec2_feature_index[ds],
				vec2_feature_score[ds],
				i_max_features,
				data_rankprod_precompute[ds]
			);
			select_features(
				vec_opt_fs_method[ds],
				data_perm_x[ds],
				data_y[ds],
				vec2_index[ds],
				vec3_index_class[ds],
				vec2_perm_feature_index[ds],
				vec2_perm_feature_score[ds],
				i_max_features,
				data_perm_rankprod_precompute[ds]
			);
		}

		// -- rank-combine features
		vector<vector<int> > vec2_opt_feature_ranks(i_num_datasets);
		for (int ds=0; ds<i_num_datasets; ds++)
			compute_ranks(vec2_feature_index[ds], vec2_opt_feature_ranks[ds]);
		vector<float> vec_rcomb_average_ranks(i_num_features, 0);
		vector<int> vec_rcomb_feature_index;
		for (int ds=0; ds<i_num_datasets; ds++)
			for (int fs=0; fs<i_num_features; fs++)
				vec_rcomb_average_ranks[fs] += vec2_opt_feature_ranks[ds][fs]*vec_sample_weight[ds];

	for (int i=0; i<i_num_datasets; i++) {
		cout << "\t" << vec_opt_fs_method[i];
	}
	cout << endl;
	for (int i=0; i<i_num_features; i++) {
		cout << i;
		cout << "\t" << vec_rcomb_average_ranks[i];
		for (int j=0; j<i_num_datasets; j++) {
			cout << "\t" << vec2_feature_score[j][vec2_opt_feature_ranks[j][i]];
			cout << " (" << vec2_opt_feature_ranks[j][i] << ")";
		}
		cout << endl;
	}

		quicksort_i(vec_rcomb_average_ranks, vec_rcomb_feature_index, SORT_ASCENDING);
		// sort the scores 
		for (int i=0; i<i_num_datasets; i++) {
			if (vec_opt_fs_method[i] == 0 || vec_opt_fs_method[i] == 2 || vec_opt_fs_method[i] == 6) {
				quicksort(vec2_feature_score[i], SORT_ASCENDING);
			} else {
				quicksort(vec2_feature_score[i], SORT_DESCENDING);
			}
		}
		// original score order
		vector<vector<float> > vec2_orig_score(i_num_datasets, vector<float>(i_num_features));
		for (int i=0; i<i_num_datasets; i++) {
			for (int j=0; j<i_num_features; j++) {
				vec2_orig_score[i][vec2_feature_index[i][j]] = vec2_feature_score[i][j];
			}
		}

		// -- rank combine permuted features
		vector<vector<int> > vec2_opt_perm_feature_ranks(i_num_datasets);
		for (int ds=0; ds<i_num_datasets; ds++)
			compute_ranks(vec2_perm_feature_index[ds], vec2_opt_perm_feature_ranks[ds]);
		vector<float> vec_perm_rcomb_average_ranks(i_num_features, 0);
		vector<int> vec_perm_rcomb_feature_index;
		for (int ds=0; ds<i_num_datasets; ds++)
			for (int fs=0; fs<i_num_features; fs++)
				vec_perm_rcomb_average_ranks[fs] += vec2_opt_perm_feature_ranks[ds][fs]*vec_sample_weight[ds];
		quicksort_i(vec_perm_rcomb_average_ranks, vec_perm_rcomb_feature_index, SORT_ASCENDING);
		// sort the scores
		for (int i=0; i<i_num_datasets; i++) {
			if (vec_opt_fs_method[i] == 0 || vec_opt_fs_method[i] == 2 || vec_opt_fs_method[i] == 6) {
				quicksort(vec2_perm_feature_score[i], SORT_ASCENDING);
			} else {
				quicksort(vec2_perm_feature_score[i], SORT_DESCENDING);
			}
		}

		// get the adjusted thresholds/interp random vals
		Matrix<float> mat_thresh(i_num_features, i_num_datasets, 0);
		Matrix<float> mat_rand(i_num_features, i_num_datasets, 0);
		for (int i=0; i<i_num_datasets; i++) {
			for (int j=0; j<i_num_features; j++) {
				mat_thresh[j][i] = vec2_feature_score[i][(int)vec_rcomb_average_ranks[j]];
				mat_rand[j][i] = vec2_perm_feature_score[i][(int)vec_perm_rcomb_average_ranks[j]];
			}
		}

		Matrix<int> mat_i_fdr(i_num_features, i_num_datasets, 0);
		Matrix<float> mat_f_fdr(i_num_features, i_num_datasets, 0);
		for (int i=0; i<i_num_datasets; i++) {
			int i_cur = 0;
			for (int j=0; j<i_num_features; j++) {
				if (j > 0) mat_i_fdr[j][i] = mat_i_fdr[j-1][i];
				// count number in mat_rand[:][i] that are less than mat_thresh[j][i]
				while (i_cur < i_num_features) {
					if (vec_opt_fs_method[i] == 0 || vec_opt_fs_method[i] == 2 || vec_opt_fs_method[i] == 6) {
						if (mat_rand[i_cur][i] < mat_thresh[j][i]) {
							mat_i_fdr[j][i]++;
							i_cur++;
						} else {
							break;
						}
					} else {
						if (mat_rand[i_cur][i] > mat_thresh[j][i]) {
							mat_i_fdr[j][i]++;
							i_cur++;
						} else {
							break;
						}
					}
				}
				mat_f_fdr[j][i] = mat_i_fdr[j][i]/(float)(j+1);
				if (mat_f_fdr[j][i] > 1) mat_f_fdr[j][i] = 1;
			}
		}

	cout << endl << endl;
	for (int i=0; i<i_num_features; i++) {
		cout << vec_rcomb_feature_index[i];
		for (int j=0; j<i_num_datasets; j++) {
			cout << "\t" << mat_f_fdr[i][j];
		}
		cout << endl;
	}

	return 0;
}
