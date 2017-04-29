
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

	DataFile<float> df(str_data_file, '\t');
	df.readDataFile(x);

	y = x[0];
	y[0] = -1;
	x = x(1,10,-1,-1);

	int i_num_samples = y.size();
	vector<int> vec_index(i_num_samples, 0);
	//for (int i=0; i<i_num_samples; i++)
	//	if (i%2 == 1) vec_index[i] = 1;
	vector<vector<int> > vec2_index_class;
	index_samples(&y, vec2_index_class);

/*	int i_num_class1 = 0;
	int i_num_class2 = 0;
	for (int i=0; i<i_num_samples; i++) {
		if (y[i] < 0) i_num_class1++; else i_num_class2++;
		cout << y[i] << "\t";
	}
	cout << endl;
	int i_num_small;

	cout << x(0,0,-1,-1) << endl;
	float stat = compute_ranksum_stat_p_value(x, vec_index, vec2_index_class, 0, i_num_class1, i_num_class2, true);
	cout << stat << endl;


	for (int i=0; i<=26; i++) {
		int test = ranksum_dist(4,5,i);
		cout << i << ":" << test << endl;
	}

	return 0;
*/


	for (int i=0; i<x.getWidth(); i++) {
		x[0][i] = 0;
	}

//	for (int i=0; i<x.getHeight(); i++) {
//		x[i][1] = x[i][0];
//	}

	vector<float> vec_scores;
	vector<int> vec_sort_index;
	ranksum_index(x, vec_index, vec2_index_class, vec_scores, vec_sort_index, false);

	vector<float> vec_unsort(vec_scores.size());
	
	for (int i=0; i<x.getHeight(); i++) {
		vec_unsort[vec_sort_index[i]] = vec_scores[i];
	}

	for (int i=0; i<x.getHeight(); i++) {
		cout << vec_unsort[i] << endl;
	}

	return 0;
}
