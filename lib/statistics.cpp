#include <miblab/statistics.h>

// sample standard deviation
float stdev(vector<float>& vec) {
	int vec_size = vec.size();
	float mean_sq = 0;
	float sum_sq = 0;
	for (int i=0; i<vec_size; i++) {
		mean_sq += vec[i];
		sum_sq += vec[i]*vec[i];
	}
	mean_sq = mean_sq/(float)vec_size;
	return sqrt( (1/(float)(vec_size-1)) * sum_sq - mean_sq*mean_sq );
}


