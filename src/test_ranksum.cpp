
#include <iostream>
#include <miblab/matrix.h>
#include <miblab/rand.h>
#include <miblab/commandline.h>
#include <miblab/feature_selection.h>
#include <miblab/classifiers/classifier_shared.h>
#include <miblab/datafileheader.h>

using namespace std;

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

	DataFileHeader<float> df(str_data_file, '\t');
	df.readDataFile(x, true, true);

	y = x[0];
	x = x(1,-1,-1,-1);

	int i_num_samples = y.size();
	vector<int> vec_index(i_num_samples, 0);
	vector<vector<int> > vec2_index_class;
	index_samples(&y, vec2_index_class);

	vector<float> vec_scores;
	vector<int> vec_sort_index;
	//ranksum_index(x, vec_index, vec2_index_class, vec_scores, vec_sort_index, false);
	mrmr_index(x, y, vec_index, vec_scores, vec_sort_index, 1000, true);

	vector<float> vec_unsort(vec_scores.size());
	
	for (int i=0; i<x.getHeight(); i++) {
		vec_unsort[vec_sort_index[i]] = vec_scores[i];
	}

	for (int i=0; i<x.getHeight(); i++) {
		//cout << vec_unsort[i] << endl;
		cout << vec_scores[i] << endl;
	}

	return 0;
}
