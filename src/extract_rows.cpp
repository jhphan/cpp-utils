
#include <iostream>
#include <miblab/matrix.h>
#include <miblab/commandline.h>
#include <miblab/datafileheader.h>
#include <map>

using namespace std;

int main(int argc, char* argv[]) {

	string str_data_file;
	string str_row_file;
	int i_col = 0;

	CommandLine cl(argc, argv);
	if (!cl.getArg(str_data_file, "f")) {
		cout << "missing -f option" << endl;
		return 1;
	}
	if (!cl.getArg(str_row_file, "r")) {
		cout << "missing -r option" << endl;
		return 1;
	}
	if (!cl.getArg(i_col, "c")) {
		cout << "missing -c option" << endl;
		return 1;
	}

	// read data file
	Matrix<float> x;
	vector<float> y;

	DataFileHeader<float> df_data(str_data_file, '\t');
	df_data.readDataFile(x, true, true);
	vector<string> vec_col_headers = df_data.getColHeaders();
	vector<string> vec_row_headers = df_data.getRowHeaders();	

	y = x[0];
	x = x(1,-1,-1,-1);
	
	// create map of row numbers
	map<string, int> map_rows;
	for (int i=1; i<x.getHeight(); i++) {
		map_rows[vec_row_headers[i]] = i-1;
	}

	// read row file
	Matrix<string> mat_rows;
	DataFile<string> df_rows(str_row_file, '\t');
	df_rows.readDataFile(mat_rows);
	
	// print new file
	for (int i=0; i<vec_col_headers.size(); i++)
		cout << "\t" << vec_col_headers[i];		
	cout << endl;
	cout << "Class";
	for (int i=0; i<y.size(); i++)
		cout << "\t" << y[i];
	cout << endl;

	for (int i=0; i<mat_rows.getHeight(); i++) {
		map<string,int>::iterator it;
		it = map_rows.find(mat_rows[i][i_col]);
		if (it != map_rows.end()) {
			int i_row = it->second;
			cout << vec_row_headers[i_row+1];
			for (int j=0; j<y.size(); j++)
				cout << "\t" << x[i_row][j];
			cout << endl;
		} else {
			cerr << "row not found: " << mat_rows[i][i_col] << endl;
		}
	}

	return 0;
}
