#include <iostream>
#include <miblab/matrix.h>
#include <miblab/rand.h>
#include <miblab/commandline.h>
#include <miblab/datafileheader.h>
#include <miblab/stringtokenizer.h>
#include <miblab/normalize.h>
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

// combine data from multiple datasets			
void combine_data(
	Matrix<float>& combine_x,
	vector<float>& combine_y,
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
	combine_x = Matrix<float>(i_num_features, i_total_samples);
	combine_y = vector<float>(i_total_samples);
	vec2_scomb_folds_train = vector<vector<int> >(i_num_folds, vector<int>(i_total_samples,0));
	vec2_scomb_folds_test = vector<vector<int> >(i_num_folds, vector<int>(i_total_samples,0));

	// combine data
	int i_cur_sample = 0;
	for (int i=0; i<i_num_datasets; i++) {
		int i_num_samples = data_x[i].getWidth();
		for (int j=0; j<i_num_samples; j++) {
			combine_y[i_cur_sample] = data_y[i][j];
			for (int k=0; k<i_num_features; k++)
				combine_x[k][i_cur_sample] = data_x[i][k][j];
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
	cout << "\t-val [validation data]" << endl;
	cout << "\t-i [number of iterations]" << endl;
	cout << "\t-f [number of folds]" << endl;
	cout << "\t-fs [max feature size]" << endl;
	cout << "\t-cls [classification method]" << endl;
	cout << "\t-quant" << endl;
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

void fs_method_single(
		vector<Matrix<float> >& data_train_x,
		vector<vector<int> >& vec2_train_index,
		vector<vector<int> >& vec2_train_index_class,
		int i_num_folds,
		vector<vector<int> >& vec2_opt_feature_index	
) {
	vec2_opt_feature_index = vector<vector<int> >(i_num_folds);

	// select features for each fold
	for (int i=0; i<i_num_folds; i++) {
		vector<float> vec_scores;
		choi_index(data_train_x[i], vec2_train_index[i], vec2_train_index_class, vec_scores, vec2_opt_feature_index[i]);
	}
}

int main(int argc, char* argv[]) {
	srand48(time(NULL));

	string str_data_file_list;
	string str_data_file_val;
	vector<string> vec_data_file_list;

	int i_num_iterations = 10;
	int i_num_folds = 3;
	int i_max_features = 50;
	int i_cls_method = 0;	
	bool b_quant = false;

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
	if (!cl.getArg(i_cls_method, "cls")) {
		usage();
		return 1;
	}
	b_quant = cl.getArg("quant");

	// individual datasets
	vector<Matrix<float> > data_x(i_num_datasets);
	vector<vector<float> > data_y(i_num_datasets);
	vector<vector<vector<int> > > vec3_index_class(i_num_datasets);
	vector<vector<int> > vec2_index(i_num_datasets);	// use all samples of data
	vector<int> vec_num_samples(i_num_datasets);
	vector<float> vec_sample_weight(i_num_datasets);
	int i_total_samples = 0;

	for (int i=0; i<i_num_datasets; i++) {
		load_data(vec_data_file_list[i], data_x[i], data_y[i]);
		index_samples(&data_y[i], vec3_index_class[i]);
		vec_num_samples[i] = data_y[i].size();
		i_total_samples += vec_num_samples[i];
		vec2_index[i] = vector<int>(vec_num_samples[i],0);	// use all samples of dataset
	}
	for (int i=0; i<i_num_datasets; i++) {
		vec_sample_weight[i] = vec_num_samples[i]/(float)i_total_samples;
		cerr << "sample weight: " << i << ", " << vec_sample_weight[i] << endl;
	}

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

		// sample-combine training folds of each dataset
		Matrix<float> data_scomb_x;
		vector<float> data_scomb_y;
		vector<vector<int> > vec2_scomb_folds_train;
		vector<vector<int> > vec2_scomb_folds_test;
		vector<vector<int> > vec2_scomb_index_class;
		vector<int> vec_scomb_index;

		combine_data(data_scomb_x, data_scomb_y, vec2_scomb_folds_train, vec2_scomb_folds_test, data_x, data_y, vec3_folds_train, vec3_folds_test);

		//cout << "x size: " << data_scomb_x.getWidth() << endl;
		//cout << "y size: " << data_scomb_y.size() << endl;
		//for (int ind=0; ind<vec2_scomb_folds_train[0].size(); ind++) {
		//	cout << vec2_scomb_folds_train[0][ind] << ",";
		//}
		//cout << endl;

		index_samples(&data_scomb_y, vec2_scomb_index_class);
		vec_scomb_index = vector<int>(data_scomb_y.size(), 0);

		// variables for individual dataset normalization
		vector<vector<Matrix<float> > > norm_data_x(i_num_datasets, vector<Matrix<float> >(i_num_folds));
		vector<vector<Matrix<float> > > norm_data_val_x(i_num_datasets, vector<Matrix<float> >(i_num_folds));
		vector<vector<Matrix<float> > > norm_data_test_x(i_num_datasets, vector<Matrix<float> >(i_num_folds));

		// variables for sample-combined dataset normalization
		vector<Matrix<float> > norm_data_scomb_x(i_num_folds);
		vector<Matrix<float> > norm_data_scomb_val_x(i_num_folds);
		vector<Matrix<float> > norm_data_scomb_test_x(i_num_folds);

		cerr << i << ": quantile normalizing..." << endl;
		// quantile normalize data
		for (int j=0; j<i_num_folds; j++) {
			for (int ds=0; ds<i_num_datasets; ds++) {
				vector<float> quant_ref;
				// create a quantile reference for the training portion of each dataset
				if (b_quant) quantile_reference(data_x[ds], vec3_folds_train[ds][j], quant_ref);
				// quantile normalize the training data to the reference
				if (b_quant) quantile_normalize_matrix(data_x[ds], norm_data_x[ds][j], vec3_folds_train[ds][j], quant_ref);
				else norm_data_x[ds][j] = data_x[ds];
				// quantile normalize the validation data to each individual dataset
				if (b_quant) quantile_normalize_matrix(data_val_x, norm_data_val_x[ds][j], vec_val_index, quant_ref);
				else norm_data_val_x[ds][j] = data_val_x;
				// quantile normalize the test data
				if (b_quant) quantile_normalize_matrix(data_x[ds], norm_data_test_x[ds][j], vec3_folds_test[ds][j], quant_ref);
				else norm_data_test_x[ds][j] = data_x[ds];
			}
			vector<float> scomb_quant_ref;
			// create a quantile reference for the training portion of combined data
			if (b_quant) quantile_reference(data_scomb_x, vec2_scomb_folds_train[j], scomb_quant_ref);
			// quantile normalize the training data
			if (b_quant) quantile_normalize_matrix(data_scomb_x, norm_data_scomb_x[j], vec2_scomb_folds_train[j], scomb_quant_ref);
			else norm_data_scomb_x[j] = data_scomb_x;
			// quantile normalize validation data to scomb data
			if (b_quant) quantile_normalize_matrix(data_val_x, norm_data_scomb_val_x[j], vec_val_index, scomb_quant_ref);
			else norm_data_scomb_val_x[j] = data_val_x;
			// quantile normalize the test data
			if (b_quant) quantile_normalize_matrix(data_scomb_x, norm_data_scomb_test_x[j], vec2_scomb_folds_test[j], scomb_quant_ref);
			else norm_data_scomb_test_x[j] = data_scomb_x;
		}

		// 1. optimize fs method for each individual dataset

		// 2. using optimized fs method, train w/ individual dataset and classify validation

		// 3. optimize fs method for sample-combined dataset

		// 4. using sample-combined fs results, train w/ individual dataset and classify validation

		// 5. using sample-combined fs results, train w/ sample-combined dataset and classify validation

		// 6. rank-combine optimal fs results from each dataset

		// 7. using rank-combined fs results, train w/ individual dataset and classify validation

		// 8. using rank-combined fs results, train w/ sample-combined dataset and classify validation


		// 1)
		vector<vector<vector<int> > > vec3_opt_feature_index(i_num_datasets);
		for (int ds=0; ds<i_num_datasets; ds++) {
			cerr << i << ": ds: " << ds << ": optimizing..." << endl;
			fs_method_single(
				norm_data_x[ds],
				vec3_folds_train[ds],
				vec3_index_class[ds],
				i_num_folds,
				vec3_opt_feature_index[ds]
			);
		}

		// 2)
		// ### select features individually, train individually
		for (int ds=0; ds<i_num_datasets; ds++) {
			cerr << i << ": ds: " << ds << ": train individual" << endl;
			for (int fold=0; fold<i_num_folds; fold++) {
				for (int fs=1; fs<=i_max_features; fs++) {
					float f_val = classify(
						i_cls_method,
						norm_data_x[ds][fold],
						data_y[ds],
						vec3_folds_train[ds][fold],
						norm_data_val_x[ds][fold],
						data_val_y,
						vec_val_index,
						vec2_val_index_class,
						vec3_opt_feature_index[ds][fold],
						fs
					);
					cout << i << "\t" << fold << "\t" << ds << "\t" << fs << "\t0\t" << f_val << endl;
				}
			}
		}
		
		// 3)

		vector<vector<int> > vec2_scomb_opt_feature_index;
		cerr << i << ": ds: scomb " << ": optimizing..." << endl;
		fs_method_single(
			norm_data_scomb_x,
			vec2_scomb_folds_train,
			vec2_scomb_index_class,
			i_num_folds,
			vec2_scomb_opt_feature_index
		);
		
		// 4)
		// ### select features from combined data, train individually
		for (int ds=0; ds<i_num_datasets; ds++) {
			cerr << i << ": ds: " << ds << ": scomb fs, train individual" << endl;
			for (int fold=0; fold<i_num_folds; fold++) {
				for (int fs=1; fs<=i_max_features; fs++) {
					float f_val = classify(
						i_cls_method,
						norm_data_x[ds][fold],
						data_y[ds],
						vec3_folds_train[ds][fold],
						norm_data_val_x[ds][fold],
						data_val_y,
						vec_val_index,
						vec2_val_index_class,
						vec2_scomb_opt_feature_index[fold],
						fs
					);
					cout << i << "\t" << fold << "\t" << ds << "\t" << fs << "\t1\t" << f_val << endl;
				}
			}
		}
	
		// 5)
		// ### select features from combined data, train w/ combined data
		cerr << i << ": scomb fs, train combined" << endl;
		for (int fold=0; fold<i_num_folds; fold++) {
			for (int fs=1; fs<=i_max_features; fs++) {
				float f_val = 0;
				//float f_val = classify(
				//	i_cls_method,
				//	norm_data_scomb_x[fold],
				//	data_scomb_y,
				//	vec2_scomb_folds_train[fold],
				//	norm_data_scomb_val_x[fold],
				//	data_val_y,
				//	vec_val_index,
				//	vec2_val_index_class,
				//	vec2_scomb_opt_feature_index[fold],
				//	fs
				//);
				cout << i << "\t" << fold << "\t" << -1 << "\t" << fs << "\t2\t" << f_val << endl;
			}
		}

		// 6)
		// -- rank-combine features, choi method
		vector<vector<int> > vec2_rcomb_feature_index(i_num_folds);
		for (int fold=0; fold<i_num_folds; fold++) {
			vector<Matrix<float> > tmp_vec_mat_data(i_num_datasets);
			vector<vector<int> > tmp_vec2_index(i_num_datasets);
			vector<vector<vector<int> > > tmp_vec3_index_class(i_num_datasets);
			for (int ds=0; ds<i_num_datasets; ds++) {
				tmp_vec_mat_data[ds] = norm_data_x[ds][fold];
				tmp_vec2_index[ds] = vec3_folds_train[ds][fold];
				tmp_vec3_index_class[ds] = vec3_index_class[ds];
			}
			vector<float> tmp_vec_scores;
			choi_index(tmp_vec_mat_data, tmp_vec2_index, tmp_vec3_index_class, tmp_vec_scores, vec2_rcomb_feature_index[fold]);
		}
			
		// 7)
		// ### rank-combined feature selection, train individually
		for (int ds=0; ds<i_num_datasets; ds++) {
			cerr << i << ": ds: " << ds << ": rcomb fs, train individual" << endl;
			for (int fold=0; fold<i_num_folds; fold++) {
				for (int fs=1; fs<=i_max_features; fs++) {
					float f_val = classify(
						i_cls_method,
						norm_data_x[ds][fold],
						data_y[ds],
						vec3_folds_train[ds][fold],
						norm_data_val_x[ds][fold],
						data_val_y,
						vec_val_index,
						vec2_val_index_class,
						vec2_rcomb_feature_index[fold],
						fs
					);
					cout << i << "\t" << fold << "\t" << ds << "\t" << fs << "\t3\t" << f_val << endl;
				}
			}
		}
		
		// 8)
		// ### rank-combined feature selection, train w/ combined data
		cerr << i << ": rcomb fs, train combined" << endl;
		for (int fold=0; fold<i_num_folds; fold++) {
			for (int fs=1; fs<=i_max_features; fs++) {
				float f_val = 0;
				//float f_val = classify(
				//	i_cls_method,
				//	norm_data_scomb_x[fold],
				//	data_scomb_y,
				//	vec2_scomb_folds_train[fold],
				//	norm_data_scomb_val_x[fold],
				//	data_val_y,
				//	vec_val_index,
				//	vec2_val_index_class,
				//	vec2_rcomb_feature_index[fold],
				//	fs
				//);
				cout << i << "\t" << fold << "\t" << -1 << "\t" << fs << "\t4\t" << f_val << endl;
			}
		}
	}

	return 0;
}
