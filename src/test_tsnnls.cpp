#include <iostream>
#include <vector>
#include <miblab/matrix.h>
#include <miblab/commandline.h>
#include <miblab/datafile.h>

#include <libtsnnls/tsnnls.h>

using namespace std;

int main(int argc, char* argv[]) {

	string str_prob_file;
	string str_x_file;

	CommandLine cl(argc, argv);
	if (!cl.getArg(str_prob_file, "p")) {
		cerr << "error: missing -p option" << endl;
		return 1;
	}
	if (!cl.getArg(str_x_file, "x")) {
		cerr << "error: missing -x option" << endl;
		return 1;
	}

	Matrix<double> P;
	Matrix<double> X;

	DataFile<double> df_prob(str_prob_file, '\t');
	df_prob.readDataFile(P);

	DataFile<double> df_x(str_x_file, '\t');
	df_x.readDataFile(X);

	//cout << P << endl;
	//cout << endl << X << endl;

	// -----

	taucs_ccs_matrix* A;
	int i_num_vals = X.getHeight();
	int i_num_elems = P.getHeight();
	int i_num_vals_used = 0;

	vector<int> vec_use(i_num_vals,0);
	vector<int> vec_map(i_num_vals,0);
	// remove zeros and map X values to new coordinates
	for (int i=0; i<i_num_vals; i++) {
		if (X[i][1] > 0) {
			vec_use[i] = 1;
			vec_map[i] = i_num_vals_used;
			i_num_vals_used++;
		}
	}

	cout << "num vals used: " << i_num_vals_used << endl;

	double* _p = NULL;
	//_p = (double*)(calloc(i_num_vals*i_num_vals, sizeof(double)));
	_p = (double*)(calloc(i_num_vals_used*i_num_vals_used, sizeof(double)));
	for (int i=0; i<i_num_elems; i++) {
		int i_row = (int)P[i][1]-1;
		int i_col = (int)P[i][0]-1;
		if (vec_use[i_row] && vec_use[i_col]) {
			int i_map_row = vec_map[i_row];
			int i_map_col = vec_map[i_col];
			if (i_map_row == i_map_col) {
				_p[i_map_row*i_num_vals_used+i_map_col] = P[i][2]+0.01;
			} else {
				_p[i_map_row*i_num_vals_used+i_map_col] = P[i][2];
			}
		}
	}

	cout << "constructing matrix" << endl;
	A = taucs_construct_sorted_ccs_matrix(_p, i_num_vals_used, i_num_vals_used);
	free(_p);

	double* _x = NULL;

	_x = (double*)(calloc(i_num_vals_used, sizeof(double)));
	for (int i=0; i<i_num_vals; i++) {
		if (vec_use[i] == 1) {
			_x[vec_map[i]] = X[i][1];
		}
	}

	// ----

	double d_residual;
	double* _y = NULL;

	tsnnls_verbosity(10);	
	_y = t_snnls(A, _x, &d_residual, -1, 1);

	if (_y == NULL) {
		cout << "nnls failed" << endl;
		char errstr[1024] = "";
		int code = tsnnls_error((char**)&errstr);
		cout << "Error code: " << code << endl;
		cout << "Error: " << errstr << endl;
		
	} else {
		for (int i=0; i<i_num_vals; i++) {
			if (vec_use[i] == 1) {
				cout << _x[vec_map[i]] << " : " << _y[vec_map[i]] << endl;
			}
		}
		free(_y);
	}

	taucs_ccs_free(A);
	free(_x);

	return 0;
}
