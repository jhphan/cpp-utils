#include <iostream>
#include <miblab/matrix.h>
#include <miblab/rand.h>
#include <miblab/commandline.h>
#include <miblab/classifiers/svm.h>
#include <miblab/classifiers/knn.h>
#include <miblab/classifiers/em.h>
#include <miblab/classifiers/lc.h>
#include <miblab/classifiers/dtree.h>
#include <miblab/classifiers/forest.h>
#include <miblab/datafile.h>
#include <miblab/strconv.h>

#include <mysql/mysql.h>

using namespace std;

void usage() {
	cout << "generate_grid " << endl;
	cout << "\t-analysis_id [file with list of analysis ids]" << endl;
	cout << "\t-pair [gene pair number]" << endl;
	cout << "\t-xmin [grid left]" << endl;
	cout << "\t-xrange [grid width]" << endl;
	cout << "\t-ymin [grid bottom]" << endl;
	cout << "\t-yrange [grid height]" << endl;
	cout << "\t-res [grid resolution]" << endl;
}

bool get_analysis_record(MYSQL* mysql, string& str_analysis_id, string& str_fold_set_id, string& str_train_data_id, string& str_cls_method) {
	string query = string("select fold_set_id,train_data_id,classification_methods from analysis where id = '")
			+str_analysis_id
			+string("'");

	if (mysql_real_query(mysql, query.c_str(), query.size())) {
		cerr << "!error: generate_grid: get_analysis_record(): cannot query for analysis record: " << query << endl;
		return false;
	}
	
	MYSQL_RES* result = mysql_store_result(mysql);
	if (!result) {
		cerr << "!error: generate_grid: get_analysis_record(): error storing mysql result for analysis record" << endl;
		return false;
	}
	MYSQL_ROW row = mysql_fetch_row(result);
	if (!row) {
		cerr << "!error: generate_grid: get_analysis_record(): empty analysis record" << endl;
		return false;
	}
	str_fold_set_id = string(row[0]);
	str_train_data_id = string(row[1]);
	str_cls_method = string(row[2]);

	mysql_free_result(result);

	return true;
}

bool get_fold_masks(MYSQL* mysql, string& str_fold_set_id, int i_number, int i_iteration, string& str_outer_fold_id, string& str_train_mask, string& str_test_mask) {
	string str_number = conv2str(i_number);
	string str_iteration = conv2str(i_iteration);
	string query = string("select outer_fold.id,outer_fold.train_mask,outer_fold.test_mask from outer_fold,subsample where ")
			+string("subsample.fold_set_id = '")+str_fold_set_id+string("' ")
			+string(" and outer_fold.subsample_id = subsample.id ")
			+string(" and outer_fold.number = ")+str_number+string(" and outer_fold.iteration = ")+str_iteration;
	if (mysql_real_query(mysql, query.c_str(), query.size())) {
		cerr << "!error: generate_grid: get_fold_masks(): cannot query for outer fold info: " << query << endl;
		return false;
	}
	MYSQL_RES* result = mysql_store_result(mysql);
	if (!result) {
		cerr << "!error: generate_grid: get_fold_masks(): error storing mysql result for fold masks" << endl;
		return false;
	}
	MYSQL_ROW row mysql_fetch_row(result);
	if (!row) {
		cerr << "!error: generate_grid: get_fold_masks(): empty fold record" << endl;
		return false;
	}

	str_outer_fold_id = string(row[0]);
	str_train_mask = string(row[1]);
	str_test_mask = string(row[2]);

	mysql_free_result(result);

	return true;
}

int main(int argc, char* argv[]) {
	srand48(time(NULL));

	string str_analysis_id;
	int i_pair_num = 0;
	float f_xmin = 0;
	float f_xrange = 0;
	float f_ymin = 0;
	float f_yrange = 0;
	float f_res = 0;
	int i_pair1 = 0;
	int i_pair2 = 1;
	int i_number = 0;
	int i_iteration = 0;
	string str_db_host = "hermes";
	string str_db_user = "omni";
	string str_db_pass = "omnipass";
	string str_db_name = "simpledap";

	CommandLine cl(argc, argv);
	if (!cl.getArg(str_analysis_id, "analysis_id")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_pair_num, "pair")) {
		usage();
		return 1;
	}
	if (!cl.getArg(f_xmin, "xmin")) {
		usage();
		return 1;
	}
	if (!cl.getArg(f_xrange, "xrange")) {
		usage();
		return 1;
	}
	if (!cl.getArg(f_ymin, "ymin")) {
		usage();
		return 1;
	}
	if (!cl.getArg(f_yrange, "yrange")) {
		usage();
		return 1;
	}
	if (!cl.getArg(f_res, "res")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_number, "num")) {
		usage();
		return 1;
	}
	if (!cl.getArg(i_iteration, "iter")) {
		usage();
		return 1;
	}
	cl.getArg(str_db_host, "dbhost");
	cl.getArg(str_db_user, "dbuser");
	cl.getArg(str_db_pass, "dbpass");
	cl.getArg(str_db_name, "dbname");

	// open mysql connection
	mysql_library_init(0, NULL, NULL);
	MYSQL* mysql = mysql_init(NULL);
	if (mysql == NULL) {
		cerr << "!error: generate_grid: cannot init mysql" << endl;
		mysql_library_end();
		return 1;
	}
	if (!mysql_real_connect(mysql, str_db_host.c_str(), str_db_user.c_str(), str_db_pass.c_str(), str_db_name.c_str(), 0, NULL, 0)) {
		cerr << "!error: generate_grid: cannot connect to mysql database" << endl;
		mysql_library_end();
		return 1;
	}

	// get analysis record
	string str_fold_set_id;
	string str_train_data_id;
	string str_cls_method;
	if (!get_analysis_record(mysql, str_analysis_id, str_fold_set_id, str_train_data_id, str_cls_method)) {
		cerr << "!error: generate_grid: cannot get analysis record: " << str_analysis_id << endl;
		mysql_close(mysql);
		mysql_library_end();
		return 1;
	}

	string str_outer_fold_id;
	string str_train_mask;
	string str_test_mask;
	if (!get_fold_masks(mysql, str_fold_set_id, i_number, i_iteration, str_outer_fold_id, str_train_mask, str_test_mask)) {
		cerr << "!error: generate_grid: cannot get fold record: " << str_fold_set_id << endl;
		mysql_close(mysql);
		mysql_library_end();
		return 1;
	}
	
	// download dataset
	Matrix<float> mat_train;
	if (!download_data(mysql, str_train_data_id, mat_train)) {
		cerr << "!error: generage_grid: cannot download train data" << endl;
		mysql_close(mysql);
		mysql_library_end();
		return 1;
	}

	// compute random pair numbers
	




	// parse classification params


	// get folds

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
	int i_num_classifiers = 7;
	vector<vector<float> > vec2_params(i_num_classifiers, vector<float>());

	const float f_rbf_vals[] = {0.1, 1, 10, 100};
	vec2_params[0] = vector<float>(f_rbf_vals, f_rbf_vals+4);
	const float f_k_vals[] = {1,3,5,7,9,11, 21, 31};
	vec2_params[1] = vector<float>(f_k_vals, f_k_vals+8);
	const float f_clust_vals[] = {2, 3, 4, 5, 20};
	vec2_params[2] = vector<float>(f_clust_vals, f_clust_vals+5);
	vec2_params[3] = vector<float>(f_rbf_vals, f_rbf_vals+4);
	vec2_params[4] = vector<float>(f_rbf_vals, f_rbf_vals+4);
	const float f_dtree_depth[] = {2,5,7,10,100};
	vec2_params[5] = vector<float>(f_dtree_depth, f_dtree_depth+5);
	const float f_forest_trees[] = {10,50,100};
	vec2_params[6] = vector<float>(f_forest_trees, f_forest_trees+3);

	const char* ch_classifier_names[] = {"SVM", "KNN", "EM", "LDA", "SDF", "DTree", "Forest"};
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
				cls = new SVM(KERNEL_RBF, 0.001, 1, vec2_params[c][k], 0, 0);
			} else if (c == 1) {
				cls = new KNN((int)vec2_params[c][k]);
			} else if (c == 2) {
				cls = new EM((int)vec2_params[c][k]);
			} else if (c == 3) {
				cls = new LC(KERNEL_FD, 0.01, KERNEL_RBF, vec2_params[c][k], 0, 0);
			} else if (c == 4) {
				cls = new LC(KERNEL_SDF, 0.01, KERNEL_RBF, vec2_params[c][k], 0, 0);
			} else if (c == 5) {
				cls = new DTree((int)vec2_params[c][k], 10, 0);
			} else if (c == 6) {
				cls = new Forest((int)vec2_params[c][k], 0.1, 5, 10);
			}
			// train classifier
			cls->train(&x, &y, index);

			// predict each point in the grid
			Matrix<float> mat_sample(2, 1);
			for (int i=0; i<i_grid_res; i++) {
				for (int j=0; j<i_grid_res; j++) {
					mat_sample[0][0] = i*f_grid_step;
					mat_sample[1][0] = j*f_grid_step;
					mat_results[i][j] = cls->predict_point_in_matrix(&mat_sample, 0);
				}
			}

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
