#include <iostream>
#include <miblab/matrix.h>
#include <miblab/datafileheader.h>
#include <miblab/strconv.h>
#include <miblab/feature_selection.h>
#include <miblab/commandline.h>
#include <miblab/classifiers/classifier_shared.h>

using namespace std;

int main(int argc, char* argv[]) {
	srand48(time(NULL));

	vector<Matrix<float> > x(2);
	vector<vector<float> > y(2);

	CommandLine cl(argc, argv);
	string str_file1;
	string str_file2;
	cl.getArg(str_file1, "f1");
	cl.getArg(str_file2, "f2");

	DataFileHeader<float> dfh1(str_file1, '\t');
	dfh1.readDataFile(x[0], true, true);
	DataFileHeader<float> dfh2(str_file2, '\t');
	dfh2.readDataFile(x[1], true, true);

	y[0] = x[0][0];
	x[0] = x[0](1,-1,-1,-1);
	y[1] = x[1][0];
	x[1] = x[1](1,-1,-1,-1);

	vector<vector<int> > vec2_index(2);
	vec2_index[0] = vector<int>(y[0].size(), 0);
	vec2_index[1] = vector<int>(y[1].size(), 0);

	vector<vector<vector<int> > > vec3_index_class(2);
	index_samples(&y[0], vec3_index_class[0]);
	index_samples(&y[1], vec3_index_class[1]);

	vector<float> vec_scores;
	vector<int> vec_feature_index;
	cout << "start..." << endl;
	choi_index(x, vec2_index, vec3_index_class, vec_scores, vec_feature_index);
	cout << "done" << endl;

	//int i_num_features = vec_scores.size();
	//for (int i=0; i<i_num_features; i++) {
	//	cout << vec_scores[i] << "\t" << vec_feature_index[i] << endl;
	//}

	return 0;

}
