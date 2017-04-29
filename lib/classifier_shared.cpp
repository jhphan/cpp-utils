#include <miblab/classifiers/classifier_shared.h>

void index_samples(vector<float>* y, vector<vector<int> >& vec2_index_class) {
	int i_max_class = 0;
	for (int i=0; i<y->size(); i++)
		if ( (*y)[i] > i_max_class ) i_max_class = (int)(*y)[i];
	int i_num_classes = i_max_class+1;
	vec2_index_class = vector<vector<int> >(i_num_classes, vector<int>());
	for (int i=0; i<y->size(); i++)	
		if ( (*y)[i] > 0 ) vec2_index_class[(int)(*y)[i]].push_back(i); else vec2_index_class[0].push_back(i);
}

float compute_accuracy_multi(vector<float>* y, vector<float>& vec_responses, vector<int>& vec_index) {
	float f_accuracy = 0;
	int i_num_samples = y->size();
	int i_total_samples = 0;
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0) {
			if ( (int)(vec_responses[i]) == (int)(*y)[i] ) f_accuracy++;
			i_total_samples++;
		}
	return f_accuracy/(float)i_total_samples;
}

// compute accuracy
float compute_accuracy_metric(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, float f_threshold) {
	float f_accuracy = 0;
	int i_num_samples = y->size();
	int i_total_samples = 0;
	for (int i=0; i<i_num_samples; i++)
		if (vec_index[i] == 0) {
			if ( (vec_dists[i] > f_threshold && (int)(*y)[i] > 0) || (vec_dists[i] <= f_threshold && (int)(*y)[i] < 0) ) f_accuracy++;
			i_total_samples++;
		}
	return f_accuracy/(float)i_total_samples;	// accuracy
}

// compute auc
float compute_AUC_metric(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, vector<vector<int> >& vec2_index_class) {
	float f_AUC = 0;
	int i_total_tests = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++) {
		if (vec_index[vec2_index_class[0][i]] == 0) {
			for (int j=0; j<vec2_index_class[1].size(); j++) {
				if (vec_index[vec2_index_class[1][j]] == 0) {
					i_total_tests++;
					if (fabs(vec_dists[vec2_index_class[0][i]]-vec_dists[vec2_index_class[1][j]]) < FLT_EPS) {	// dists are equal
						f_AUC += 0.5;
					} else {
						if (vec_dists[vec2_index_class[0][i]] < vec_dists[vec2_index_class[1][j]]) f_AUC++;
					}
				}
			}
		}
	}
	return f_AUC/(float)i_total_tests;	// auc
}

void compute_metrics(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int& tp, int& tn, int& fp, int& fn, float f_threshold) {
	tp = 0;
	tn = 0;
	fp = 0;
	fn = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++) {
		if (vec_index[vec2_index_class[0][i]] == 0) {
			if (vec_dists[vec2_index_class[0][i]] > f_threshold) fp++; else tn++;
		}
	}
	for (int i=0; i<vec2_index_class[1].size(); i++) {
		if (vec_index[vec2_index_class[1][i]] == 0) {
			if (vec_dists[vec2_index_class[1][i]] > f_threshold) tp++; else fn++;
		}
	}
}

// compute MCC
float compute_MCC_metric(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, float f_threshold) {
	float f_MCC = 0;
	int tp = 0;
	int tn = 0;
	int fp = 0;
	int fn = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++) {
		if (vec_index[vec2_index_class[0][i]] == 0) {
			if (vec_dists[vec2_index_class[0][i]] > f_threshold) fp++; else tn++;
		}
	}
	for (int i=0; i<vec2_index_class[1].size(); i++) {
		if (vec_index[vec2_index_class[1][i]] == 0) {
			if (vec_dists[vec2_index_class[1][i]] > f_threshold) tp++; else fn++;
		}
	}
	float f_den = sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn);
	if (f_den < FLT_EPS) {	// floating point equals 0
		f_MCC = 0;
	} else {
		f_MCC = (tp*tn-fp*fn)/f_den;
	}
	return f_MCC;
}

// compute Binary AUC
float compute_BAUC_metric(vector<float>* y, vector<float>& vec_dists, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, float f_threshold) {
	float f_BAUC = 0;
	int tp = 0;
	int tn = 0;
	int fp = 0;
	int fn = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++) {
		if (vec_index[vec2_index_class[0][i]] == 0) {
			if (vec_dists[vec2_index_class[0][i]] > f_threshold) fp++; else tn++;
		}
	}
	for (int i=0; i<vec2_index_class[1].size(); i++) {
		if (vec_index[vec2_index_class[1][i]] == 0) {
			if (vec_dists[vec2_index_class[1][i]] > f_threshold) tp++; else fn++;
		}
	}
	float f_sen = (float) tp / (tp+fn);
	float f_spe = (float) tn / (tn+fp);
	f_BAUC = (f_sen + f_spe)/2.0;
	return f_BAUC;
}

// compute indexes for stratified cross validation
// vec3_folds is pre-allocated
void stratified_cv(int i_folds, vector<vector<int> >& vec2_index_class, vector< vector<int> >& vec2_folds) {
	int i_num_classes = vec2_index_class.size();
	vector<int> vec_base_num_samples(i_num_classes,0);
	vector<int> vec_rem_samples(i_num_classes,0);

	// compute the number of samples in each fold
	for (int i=0; i<i_num_classes; i++) {
		vec_base_num_samples[i] = vec2_index_class[i].size() / i_folds;
		vec_rem_samples[i] = vec2_index_class[i].size() % i_folds;
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
			int i_fold_num = (int)(drand48()*i_folds);
			while (vec_cur_fold_count[j][i_fold_num] >= vec2_max_fold_count[j][i_fold_num])
				i_fold_num = (int)(drand48()*i_folds);
			vec_cur_fold_count[j][i_fold_num]++;
			vec2_folds[i_fold_num][vec2_index_class[j][k]] = 1;
		}
	}
}


// vec_index is pre-allocated
void stratified_bs(vector<vector<int> >& vec2_index_class, vector<int>& vec_index) {
	int i_num_classes = vec2_index_class.size();	

	bool b_passed = false;
	while (!b_passed) {
		// reset index to all 1's
		for (int i=0; i<vec_index.size(); i++)
			vec_index[i] = 1;

		// reset flags
		vector<int> vec_flagged(i_num_classes, 0);

		for (int i=0; i<i_num_classes; i++) {
			for (int j=0; j<vec2_index_class[i].size(); j++) {
				int i_sample = (int)(drand48()*vec2_index_class[i].size());
				if (vec_index[vec2_index_class[i][i_sample]] == 1) {	// new sample
					vec_flagged[i]++;
					vec_index[vec2_index_class[i][i_sample]] = 0;
				}
			}
		}
		// number of samples selected for each class should be less than total number of samples
		b_passed = true;
		//for (int i=0; i<i_num_classes; i++)
		//	if (vec_flagged[i] >= vec2_index_class[i].size()) {
		//		b_passed = false;
		//		break;
		//	}
	}
}

// compute the 0.632 or 0.632+ bootstrap
float compute_bs_metric(float f_metric_resub, float f_metric_bs, int i_bs_type) {
	if (i_bs_type == BS_REG) {
		return f_metric_bs;
	}
	
	float R = 0;
	if (i_bs_type == BS_0632P) {
		// compute R
		// lambda = 0.5
		if (f_metric_bs < f_metric_resub && 0.5 < f_metric_resub) {
			R = (f_metric_resub-f_metric_bs)/(f_metric_resub-0.5);
			if (R > 1) R = 1;
		}
	}
	// w
	float w = 0.632/(1-0.368*R);
	if (w > 1) w = 1;
	return (1-w)*f_metric_resub+w*f_metric_bs;
}

