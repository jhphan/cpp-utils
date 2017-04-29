#include <miblab/feature_selection.h>

// assume gene expression values are in the log domain
// data has no class labels row

void fold_change_index(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	int i_num_rows = mat_data.getHeight();

	// compute fold change values
	vec_scores = vector<float>(i_num_rows);
	for (int i=0; i<i_num_rows; i++)
		vec_scores[i] = compute_fold_change(mat_data, vec_index, vec2_index_class, i);

	// get the sort index
	quicksort_i(vec_scores, vec_sort_index, SORT_DESCENDING);
}

void t_test_index(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	int i_num_rows = mat_data.getHeight();

	int i_num_class1 = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++) if (vec_index[vec2_index_class[0][i]] == 0) i_num_class1++;
	int i_num_class2 = 0;
	for (int i=0; i<vec2_index_class[1].size(); i++) if (vec_index[vec2_index_class[1][i]] == 0) i_num_class2++;

	vec_scores = vector<float>(i_num_rows);
	for (int i=0; i<i_num_rows; i++)
		vec_scores[i] = compute_t_stat(mat_data, vec_index, vec2_index_class, i, i_num_class1, i_num_class2);

	quicksort_i(vec_scores, vec_sort_index, SORT_ASCENDING);
}

void sam_index(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	int i_num_rows = mat_data.getHeight();

	int i_num_class1 = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++) if (vec_index[vec2_index_class[0][i]] == 0) i_num_class1++;
	int i_num_class2 = 0;
	for (int i=0; i<vec2_index_class[1].size(); i++) if (vec_index[vec2_index_class[1][i]] == 0) i_num_class2++;
	
	vector<float> vec_mean_class1;
	vector<float> vec_mean_class2;
	vector<float> vec_pooled_stdev;
	compute_sam_stats(mat_data, vec_index, vec2_index_class, i_num_class1, i_num_class2, vec_mean_class1, vec_mean_class2, vec_pooled_stdev);

	// find the median of the pooled stdev
	vector<float> vec_tmp_stdev = vec_pooled_stdev;
	quicksort(vec_tmp_stdev);
	int i_mid_row = i_num_rows/2;
	float f_s0;
	if (i_num_rows%2 == 0) {
		f_s0 = (vec_tmp_stdev[i_mid_row]+vec_tmp_stdev[i_mid_row])/2;
	} else {
		f_s0 = vec_tmp_stdev[i_mid_row+1];
	}

	// compute sam statistic with fudge factor
	vec_scores = vector<float>(i_num_rows);
	for (int i=0; i<i_num_rows; i++)
		vec_scores[i] = fabs(vec_mean_class1[i]-vec_mean_class2[i])/(vec_pooled_stdev[i]+f_s0);

	quicksort_i(vec_scores, vec_sort_index, SORT_DESCENDING);
}

void ranksum_index(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index, bool b_exact) {
	int i_num_rows = mat_data.getHeight();
	
	int i_num_class1 = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++) if (vec_index[vec2_index_class[0][i]] == 0) i_num_class1++;
	int i_num_class2 = 0;
	for (int i=0; i<vec2_index_class[1].size(); i++) if (vec_index[vec2_index_class[1][i]] == 0) i_num_class2++;

	vec_scores = vector<float>(i_num_rows);
	for (int i=0; i<i_num_rows; i++)
		vec_scores[i] = compute_ranksum_stat(mat_data, vec_index, vec2_index_class, i, i_num_class1, i_num_class2, b_exact);

	quicksort_i(vec_scores, vec_sort_index, SORT_ASCENDING);
}


void rankprod_precompute(Matrix<float>& mat_data, vector<vector<int> >& vec2_index_class, Matrix<vector<float> >& mat_scores) {
	int i_num_class1 = vec2_index_class[0].size();
	int i_num_class2 = vec2_index_class[1].size();
	int i_num_features = mat_data.getHeight();

	mat_scores = Matrix<vector<float> >(i_num_class1, i_num_class2, vector<float>(i_num_features, 0));
	for (int i=0; i<i_num_class1; i++) {
		int i_index_class1 = vec2_index_class[0][i];
		for (int j=0; j<i_num_class2; j++) {
			int i_index_class2 = vec2_index_class[1][j];
			for (int k=0; k<i_num_features; k++) {
				mat_scores[i][j][k] = fabs(mat_data[k][i_index_class2]-mat_data[k][i_index_class1]);
			}
		}
	}

	for (int i=0; i<i_num_class1; i++) {
		for (int j=0; j<i_num_class2; j++) {
			vector<int> vec_tmp_sort_index;
			quicksort_i(mat_scores[i][j], vec_tmp_sort_index, SORT_DESCENDING);
			for (int k=0; k<i_num_features; k++)
				mat_scores[i][j][vec_tmp_sort_index[k]] = log((k+1)/(float)(i_num_features));
		}
	}

}

void rankprod_index_quick(Matrix<vector<float> >& mat_scores, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	int i_num_features = mat_scores[0][0].size();
	int i_num_class1 = vec2_index_class[0].size();
	int i_num_class2 = vec2_index_class[1].size();

	vec_scores = vector<float>(i_num_features, 0);
	for (int i=0; i<i_num_class1; i++) {
		int i_index_class1 = vec2_index_class[0][i];
		if (vec_index[i_index_class1] == 0) {
			for (int j=0; j<i_num_class2; j++) {
				int i_index_class2 = vec2_index_class[1][j];
				if (vec_index[i_index_class2] == 0) {
					for (int k=0; k<i_num_features; k++)
						vec_scores[k] += mat_scores[i][j][k];
				}
			}
		}
	}
	quicksort_i(vec_scores, vec_sort_index, SORT_ASCENDING);
}

// rank product feature select: Breitling et al. FEBS Letters, 2004
void rankprod_index(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	// count number of elements in each class
	int i_num_class1 = 0;
	int i_total_class1 = vec2_index_class[0].size();
	for (int i=0; i<i_total_class1; i++)
		if (vec_index[vec2_index_class[0][i]] == 0) i_num_class1++;
	int i_num_class2 = 0;
	int i_total_class2 = vec2_index_class[1].size();
	for (int i=0; i<i_total_class2; i++)
		if (vec_index[vec2_index_class[1][i]] == 0) i_num_class2++;

	int i_num_features = mat_data.getHeight();

	// construct matrix of pairwise fold changes
	int i_total_cols = i_num_class1*i_num_class2;
	Matrix<float> mat_scores(i_num_features,i_total_cols,0);
	for (int i=0; i<i_num_features; i++) {
		int i_col = 0;
		for (int j=0; j<i_total_class1; j++) {
			int i_index_class1 = vec2_index_class[0][j];
			if (vec_index[i_index_class1] == 0) {
				for (int k=0; k<i_total_class2; k++) {
					int i_index_class2 = vec2_index_class[1][k];
					if (vec_index[i_index_class2] == 0) {
						mat_scores[i][i_col] = fabs(mat_data[i][i_index_class2]-mat_data[i][i_index_class1]);
						i_col++;
					}
				}
			}
		}
	}

	// rank each column
	for (int i=0; i<i_total_cols; i++) {
		vector<int> vec_tmp_sort_index;
		quicksort_i(mat_scores, vec_tmp_sort_index, i, SORT_DESCENDING);
		for (int j=0; j<i_num_features; j++)
			mat_scores[vec_tmp_sort_index[j]][i] = log((j+1)/(float)(i_num_features));
	}
	vec_scores = vector<float>(i_num_features, 0);
	for (int i=0; i<i_num_features; i++)
		for (int j=0; j<i_total_cols; j++)
			vec_scores[i] += mat_scores[i][j];

	quicksort_i(vec_scores, vec_sort_index, SORT_ASCENDING);
}

void compute_ranks(vector<int>& vec_sort_index, vector<int>& vec_ranks) {
	int i_num_rows = vec_sort_index.size();
	vec_ranks = vector<int>(i_num_rows);
	for (int i=0; i<i_num_rows; i++)
		vec_ranks[vec_sort_index[i]] = i;
}

float compute_fold_change(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_row) {
	float f_mean_class1 = indexed_mean(mat_data[i_row], vec_index, vec2_index_class[0]);
	float f_mean_class2 = indexed_mean(mat_data[i_row], vec_index, vec2_index_class[1]);

	return fabs(f_mean_class1-f_mean_class2);
}

void compute_sam_stats(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_num_class1, int i_num_class2, vector<float>& vec_mean_class1, vector<float>& vec_mean_class2, vector<float>& vec_pooled_stdev) {

	int i_num_features = mat_data.getHeight();
	vec_mean_class1 = vector<float>(i_num_features, 0);
	vec_mean_class2 = vector<float>(i_num_features, 0);
	vec_pooled_stdev = vector<float>(i_num_features, 0);

	for (int i=0; i<i_num_features; i++) {
		vec_mean_class1[i] = indexed_mean(mat_data[i], vec_index, vec2_index_class[0]);
		vec_mean_class2[i] = indexed_mean(mat_data[i], vec_index, vec2_index_class[1]);

		float f_stdev_class1 = indexed_stdev(mat_data[i], vec_index, vec2_index_class[0]);
		float f_stdev_class2 = indexed_stdev(mat_data[i], vec_index, vec2_index_class[1]);
		
		vec_pooled_stdev[i] = sqrt(
					(1/(float)i_num_class1+1/(float)i_num_class2)
					*(
					  (i_num_class1-1)*f_stdev_class1*f_stdev_class1
					 +(i_num_class2-1)*f_stdev_class2*f_stdev_class2
					 )
					/(i_num_class1+i_num_class2-2)
				);
	}
}

// t statistic of samples of unequal size, but equal and unknown variances
float compute_t_stat(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_row, int i_num_class1, int i_num_class2) {
	float f_mean_class1 = indexed_mean(mat_data[i_row], vec_index, vec2_index_class[0]);
	float f_mean_class2 = indexed_mean(mat_data[i_row], vec_index, vec2_index_class[1]);

	float f_stdev_class1 = indexed_stdev(mat_data[i_row], vec_index, vec2_index_class[0]);
	float f_stdev_class2 = indexed_stdev(mat_data[i_row], vec_index, vec2_index_class[1]);

	float f_pooled_stdev = sqrt( 
					(
					 (i_num_class1-1)*f_stdev_class1*f_stdev_class1
					+(i_num_class2-1)*f_stdev_class2*f_stdev_class2
					)
					/(i_num_class1+i_num_class2-2)
				);
	float f_t_stat = fabs( ( f_mean_class1 - f_mean_class2 )
			/( f_pooled_stdev
				*sqrt(
					1/(float)i_num_class1 + 1/(float)i_num_class2
				)
			) );
	int i_df = i_num_class1+i_num_class2-2;
	students_t dist(i_df);

	float f_p = 2*cdf(complement(dist, f_t_stat));

	return f_p;
}

// recursively compute ranksum distribution
int ranksum_dist(int n, int N, int w) {
	if (n == 1) return 1;
	if (n == 2) {
		int upper = n*(N-n);
		if (w > upper/2) w = upper-w;
		return (2*w+3+((w%2==0)? 1:-1))/4;
	}
	int val = 0;
	for (int i=1; i<=(N-(n-1)); i++) {
		int new_w = w-n*(i-1);
		if (new_w >= 0 && new_w <= (n-1)*(N-i-n+1))
			val += ranksum_dist(n-1,N-i,w-n*(i-1));
	}
	return val;
}

int n_choose_k(int n, int k) {
	int num = 1;
	int den = 1;
	for (int i=n; i>(n-k); i--)
		num*=i;
	for (int i=k; i>1; i--)
		den*=i;
	return num/den;
}

float compute_ranksum_stat(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_row, int i_num_class1, int i_num_class2, bool b_exact) {
	int i_total_samples = i_num_class1+i_num_class2;
	vector<float> vec_data(i_total_samples);
	int i_cur = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++)
		if (vec_index[vec2_index_class[0][i]] == 0) {
			vec_data[i_cur] = mat_data[i_row][vec2_index_class[0][i]];
			i_cur++;
		}
	for (int i=0; i<vec2_index_class[1].size(); i++)
		if (vec_index[vec2_index_class[1][i]] == 0) {
			vec_data[i_cur] = mat_data[i_row][vec2_index_class[1][i]];
			i_cur++;
		}
	// compute the rank order of the vector
	vector<int> vec_sort_index;
	vector<int> vec_ranks;
	quicksort_i(vec_data, vec_sort_index, SORT_ASCENDING);
	compute_ranks(vec_sort_index, vec_ranks);
	for (int i=0; i<i_total_samples; i++)
		vec_ranks[i] += 1;

	//compute the ranksum statistic
	int i_stat = 0;
	for (int i=0; i<i_num_class1; i++)
		i_stat += vec_ranks[i];
	i_stat -= i_num_class1*(i_num_class1+1)/2;
	// use the smaller statistic
	int i_upper = i_num_class1*(i_total_samples-i_num_class1);
	int i_num_small = i_num_class1;
	if (i_stat > i_upper/2) {
		i_stat = i_upper-i_stat;
		i_num_small = i_num_class2;
	}
	// compute p-val
	float f_pval = 0;
	if (!b_exact) {
		// approximate with normal distribution
		float f_mu = i_num_class1*i_num_class2/2.0;
		//cout << "stat: " << i_stat << endl;
		//cout << "mu: " << f_mu << endl;
		float f_stdev = sqrt( f_mu*(i_total_samples+1)/6.0 );
		// add 0.5 for continuity correction
		float f_zstat = (i_stat+0.5-f_mu)/f_stdev;
		//cout << "stdev: " << f_stdev << endl;
		//cout << "zstat: " << f_zstat << endl;
		normal norm;
		f_pval = 2*cdf(norm, f_zstat);
	} else {
		// compute exactly
		int i_total_perms = n_choose_k(i_total_samples, i_num_class1);
		int i_sum_freq = 0;
		for (int w=0; w<=i_stat; w++)
			i_sum_freq += ranksum_dist(i_num_small, i_total_samples, w);
		f_pval = 2*i_sum_freq/(float)i_total_perms;
	}
	return f_pval;
}

// compute the mean of elements in vec_data that are indexed
float indexed_mean(vector<float>& vec_data, vector<int>& vec_index, vector<int>& vec_index_class) {
	float f_sum = 0;
	int i_num = 0;
	int i_max = vec_index_class.size();
	for (int i=0; i<i_max; i++)
		if (vec_index[vec_index_class[i]] == 0) {
			i_num++;
			f_sum+=vec_data[vec_index_class[i]];
		}
	return f_sum/(float)i_num;
}

// compute the stdev of elements in vec_data that are indexed
float indexed_stdev(vector<float>& vec_data, vector<int>& vec_index, vector<int>& vec_index_class) {
	int i_max = vec_index_class.size();
	int i_num = 0;
	float f_mean_sq = 0;
	float f_sum_sq = 0;
	for (int i=0; i<i_max; i++) {
		if (vec_index[vec_index_class[i]] == 0) {
			i_num++;
			float f_data = vec_data[vec_index_class[i]];
			f_mean_sq += f_data;
			f_sum_sq += f_data*f_data;
		}
	}
	f_mean_sq = f_mean_sq/(float)i_num;
	return sqrt( (1/(float)(i_num-1)) * f_sum_sq - (i_num/(float)(i_num-1))*f_mean_sq*f_mean_sq );
}

// --- mRMR functions

void mrmr_index(Matrix<float>& mat_data, vector<float>& vec_labels, vector<int>& vec_index, vector<float>& vec_scores, vector<int>& vec_sort_index, int i_max_num, bool b_miq) {
	int i_num_rows = mat_data.getHeight();

	// calculate mutual info of all features with labels
	vector<float> vec_mut_info(i_num_rows,0);
	for (int i=0; i<i_num_rows; i++)
		vec_mut_info[i] = compute_mutual_info(vec_labels, mat_data[i], vec_index);

	// sort 
	vector<int> vec_mut_info_sort_index;
	quicksort_i(vec_mut_info, vec_mut_info_sort_index, SORT_DESCENDING);

	// keep an index of selected features
	vector<int> vec_selected_features(i_num_rows, 0);

	vec_sort_index = vector<int>(i_num_rows,0);
	vec_sort_index[0] = vec_mut_info_sort_index[0];
	vec_scores = vector<float>(i_num_rows,0);
	vec_scores[0] = vec_mut_info[0];	// first feature is the one with highest mi
	vec_selected_features[vec_mut_info_sort_index[0]] = 1;

	Matrix<float> mat_mut_info(i_max_num, i_num_rows, -1);
	//cout << "feat: " << vec_sort_index[0] << "," << vec_scores[0];

	for (int i=1; i<i_max_num; i++) {
		// fill in the rest of the features
		//cout << "mrmr: " << i << endl;
		float f_cur_score = -10;
		int i_cur_index = 0;
		for (int j=0; j<i_num_rows; j++) {
			if (vec_selected_features[vec_mut_info_sort_index[j]] == 1)
				continue;	// skip if already in the list
			float f_rel_val = vec_mut_info[j];
			//cout << "checking: i: " << i << ", j: " << j << ", " << vec_mut_info_sort_index[j] << "," << f_rel_val << endl;	
			float f_red_val = 0;
			for (int k=0; k<i; k++) {
				int i_idx_j = vec_mut_info_sort_index[j];
				int i_idx_k = vec_sort_index[k];
				if (mat_mut_info[k][j] < 0) {
					mat_mut_info[k][j] = compute_mutual_info(mat_data[i_idx_j], mat_data[i_idx_k], vec_index);
					//cout << "new mut: " << mat_mut_info[k][j] << endl;
				}
				f_red_val += mat_mut_info[k][j];
			}
			f_red_val /= i;

			float f_score = 0;
			if (b_miq) 
				f_score = f_rel_val/(f_red_val+FLT_EPS);
			else
				f_score = f_rel_val-f_red_val;
			if (f_score > f_cur_score) {
				f_cur_score = f_score;
				i_cur_index = vec_mut_info_sort_index[j];
			}
		}
		vec_sort_index[i] = i_cur_index;
		vec_scores[i] = f_cur_score;
		vec_selected_features[i_cur_index] = 1;
	}
	// fill in the rest of the index with unused features sorted by mutual info
	int i_cur_index = i_max_num;
	for (int i=0; i<i_num_rows; i++) {
		if (vec_selected_features[vec_mut_info_sort_index[i]] == 0) {
			vec_sort_index[i_cur_index] = vec_mut_info_sort_index[i];
			vec_scores[i_cur_index] = vec_mut_info[i];
			i_cur_index++;
		}
	}
}

int discretize_vector(vector<float>& vec_float, vector<int>& vec_int, vector<int>& vec_index) {
	int i_size = vec_float.size();

	int i_index = 0;
	if (vec_index.size() == 0) {
		vec_index = vector<int>(i_size,0);
	} else {
		// find the first value with index of 0
		while (vec_index[i_index] == 1) i_index++;
	}

	int i_min, i_max;
	if (vec_float[i_index] > 0)
		i_max = i_min = int(vec_float[i_index]+0.5);
	else
		i_max = i_min = int(vec_float[i_index]-0.5);

	for (int i=0; i<i_size; i++) {
		if (vec_index[i] == 0) {
			float f_tmp = vec_float[i];
			int i_tmp = (f_tmp > 0)? int(f_tmp+0.5) : int(f_tmp-0.5);
			i_min = (i_min < i_tmp) ? i_min : i_tmp;
			i_max = (i_max > i_tmp) ? i_max : i_tmp;
			vec_int[i] = i_tmp;
		}
	}
	for (int i=0; i<i_size; i++)
		if (vec_index[i] == 0)
			vec_int[i] -= i_min;

	return (i_max-i_min+1);	// return discrete range of data
}

void compute_joint_prob(Matrix<float>& mat_joint_prob, vector<float>& vec_f1, vector<float>& vec_f2, vector<int>& vec_index) {
	int i_size = vec_f1.size();
	
	vector<int> vec_int1(i_size,0);
	vector<int> vec_int2(i_size,0);

	int i_range1 = discretize_vector(vec_f1, vec_int1, vec_index);
	int i_range2 = discretize_vector(vec_f2, vec_int2, vec_index);

	mat_joint_prob = Matrix<float>(i_range1, i_range2, 0);

	for (int i=0; i<i_size; i++)
		if (vec_index[i] == 0)
			mat_joint_prob[vec_int1[i]][vec_int2[i]] += 1;

	for (int i=0; i<i_range1; i++)
		for (int j=0; j<i_range2; j++)
			mat_joint_prob[i][j] /= i_size;
}

float compute_mutual_info(vector<float>& vec_f1, vector<float>& vec_f2, vector<int>& vec_index) {
	// compute joint probability
	Matrix<float> mat_joint_prob;
	compute_joint_prob(mat_joint_prob, vec_f1, vec_f2, vec_index);

	// compute marginal probabilities

	vector<float> vec_marg_p1(mat_joint_prob.getHeight(),0);
	vector<float> vec_marg_p2(mat_joint_prob.getWidth(),0);

	for (int i=0; i<vec_marg_p1.size(); i++) {
		for (int j=0; j<vec_marg_p2.size(); j++) {
			vec_marg_p1[i] += mat_joint_prob[i][j];
			vec_marg_p2[j] += mat_joint_prob[i][j];
		}
	}

	float f_mut_info = 0;
	for (int i=0; i<vec_marg_p1.size(); i++) {
		for (int j=0; j<vec_marg_p2.size(); j++) {
			if (mat_joint_prob[i][j] > FLT_EPS
				&& vec_marg_p1[i] > FLT_EPS
				&& vec_marg_p2[j] > FLT_EPS)
			{
				f_mut_info += mat_joint_prob[i][j]*log(mat_joint_prob[i][j]/vec_marg_p1[i]/vec_marg_p2[j]);
			}
		}
	}
	return f_mut_info/log(2);
}

// functions for method by Choi et al, Bioinformatics, 2003
void choi_index(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	int i_num_features = mat_data.getHeight();

	vector<vector<float> > vec2_means(1, vector<float>());
	vector<vector<float> > vec2_vars(1, vector<float>());

		int i_num_class1 = 0;
		for (int i=0; i<vec2_index_class[0].size(); i++) if (vec_index[vec2_index_class[0][i]] == 0) i_num_class1++;
		int i_num_class2 = 0;
		for (int i=0; i<vec2_index_class[1].size(); i++) if (vec_index[vec2_index_class[1][i]] == 0) i_num_class2++;
		
		choi_dstar(mat_data, vec_index, vec2_index_class, i_num_class1, i_num_class2, vec2_means[0]);
		choi_sigmad(vec2_means[0], i_num_class1, i_num_class2, vec2_vars[0]);

	vector<float> vec_tau2DL;
	vec_tau2DL = vector<float>(i_num_features, 0);
	choi_mutau2(vec2_means, vec2_vars, vec_tau2DL, vec_scores);

	for (int i=0; i<i_num_features; i++)
		vec_scores[i] = fabs(vec_scores[i]);

	quicksort_i(vec_scores, vec_sort_index, SORT_DESCENDING);
}

void choi_index(vector<Matrix<float> >& vec_mat_data, vector<vector<int> >& vec2_index, vector<vector<vector<int> > >& vec3_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	int i_num_studies = vec_mat_data.size();
	int i_num_features = vec_mat_data[0].getHeight();

	vector<vector<float> > vec2_means(i_num_studies, vector<float>());
	vector<vector<float> > vec2_vars(i_num_studies, vector<float>());

	for (int n=0; n<i_num_studies; n++) {
		int i_num_class1 = 0;
		for (int i=0; i<vec3_index_class[n][0].size(); i++) if (vec2_index[n][vec3_index_class[n][0][i]] == 0) i_num_class1++;
		int i_num_class2 = 0;
		for (int i=0; i<vec3_index_class[n][1].size(); i++) if (vec2_index[n][vec3_index_class[n][1][i]] == 0) i_num_class2++;
		
		choi_dstar(vec_mat_data[n], vec2_index[n], vec3_index_class[n], i_num_class1, i_num_class2, vec2_means[n]);
		choi_sigmad(vec2_means[n], i_num_class1, i_num_class2, vec2_vars[n]);
	}

	vector<float> vec_tau2DL;
	if (i_num_studies > 1) { // random effects model
		vector<float> vec_q;
		choi_Q(vec2_means, vec2_vars, vec_q);
		choi_tau2DL(vec_q, vec2_vars, vec_tau2DL);
	} else { // fixed effects model
		vec_tau2DL = vector<float>(i_num_features, 0);
	}
	choi_mutau2(vec2_means, vec2_vars, vec_tau2DL, vec_scores);
		
	for (int i=0; i<i_num_features; i++)
		vec_scores[i] = fabs(vec_scores[i]);

	quicksort_i(vec_scores, vec_sort_index, SORT_DESCENDING);
}

float choi_t_stat(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_row, int i_num_class1, int i_num_class2) {
	float f_mean_class1 = indexed_mean(mat_data[i_row], vec_index, vec2_index_class[0]);
	float f_mean_class2 = indexed_mean(mat_data[i_row], vec_index, vec2_index_class[1]);

	float f_stdev_class1 = indexed_stdev(mat_data[i_row], vec_index, vec2_index_class[0]);
	float f_stdev_class2 = indexed_stdev(mat_data[i_row], vec_index, vec2_index_class[1]);

	float f_pooled_stdev = sqrt( 
					(
					 (i_num_class1-1)*f_stdev_class1*f_stdev_class1
					+(i_num_class2-1)*f_stdev_class2*f_stdev_class2
					)
					/(i_num_class1+i_num_class2-2)
				);
	float f_t_stat = ( f_mean_class1 - f_mean_class2 )
			/( f_pooled_stdev
				*sqrt(
					1/(float)i_num_class1 + 1/(float)i_num_class2
				)
			);

	return f_t_stat*sqrt((i_num_class1+i_num_class2)/(float)(i_num_class1*i_num_class2));
}

// compute dstar, unbiased estimate of d
void choi_dstar(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_num_class1, int i_num_class2, vector<float>& vec_dstar) {
	int i_num_features = mat_data.getHeight();

	vec_dstar = vector<float>(i_num_features);
	for (int i=0; i<i_num_features; i++) {
		vec_dstar[i] = choi_t_stat(mat_data, vec_index, vec2_index_class, i, i_num_class1, i_num_class2);
		vec_dstar[i] = vec_dstar[i]-3*vec_dstar[i]/(4*(i_num_class1+i_num_class2-2)-1);
	}
}

// compute estimate of variance of unbiased d
void choi_sigmad(vector<float>& vec_dstar, int i_num_class1, int i_num_class2, vector<float>& vec_sigmad) {
	int i_num_features = vec_dstar.size();
	vec_sigmad = vector<float>(i_num_features);

	for (int i=0; i<i_num_features; i++)
		vec_sigmad[i] = 1/(float)i_num_class1+1/(float)i_num_class2+vec_dstar[i]*vec_dstar[i]/((float)2*(i_num_class1+i_num_class2));
}

void choi_Q(vector<vector<float> >& vec2_means, vector<vector<float> >& vec2_vars, vector<float>& vec_q) {
	int i_num_studies = vec2_means.size();
	int i_num_features = vec2_means[0].size();

	vec_q = vector<float>(i_num_features, 0);
	for (int i=0; i<i_num_features; i++) {
		float f_mu_num = 0;
		float f_mu_den = 0;
		vector<float> vec_w(i_num_studies, 0);
		for (int j=0; j<i_num_studies; j++) {
			vec_w[j] = 1/(float)vec2_vars[j][i];
			f_mu_num += vec2_means[j][i]*vec_w[j];
			f_mu_den += vec_w[j];
		}
		float f_mu = f_mu_num/f_mu_den;
		for (int j=0; j<i_num_studies; j++)
			vec_q[i] += vec_w[j]*(vec2_means[j][i]-f_mu)*(vec2_means[j][i]-f_mu);
	}
}

void choi_tau2DL(vector<float>& vec_q, vector<vector<float> >& vec2_vars, vector<float>& vec_tau2DL) {
	int i_num_features = vec_q.size();
	int i_num_studies = vec2_vars.size();
	vec_tau2DL = vector<float>(i_num_features, 0);

	for (int i=0; i<i_num_features; i++) {
		vector<float> vec_w(i_num_studies, 0);
		float f_tmp1 = 0;
		float f_tmp2 = 0;
		for (int j=0; j<i_num_studies; j++) {
			vec_w[j] = 1/(float)vec2_vars[j][i];
			f_tmp1 += vec_w[j];
			f_tmp2 += vec_w[j]*vec_w[j];
		}
		float f_val = (vec_q[i]-(i_num_studies-1))/(f_tmp1-(f_tmp2/f_tmp1));
		if (f_val > 0) vec_tau2DL[i] = f_val;
	}
}

void choi_mutau2(vector<vector<float> >& vec2_means, vector<vector<float> >& vec2_vars, vector<float>& vec_tau2, vector<float>& vec_mu) {
	int i_num_features = vec2_means[0].size();
	int i_num_studies = vec2_means.size();

	vec_mu = vector<float>(i_num_features, 0);
	for (int i=0; i<i_num_features; i++) {
		vector<float> vec_w(i_num_studies, 0);
		float f_tmp1 = 0;
		float f_tmp2 = 0;
		for (int j=0; j<i_num_studies; j++) {
			vec_w[j] = 1/(vec2_vars[j][i]+vec_tau2[i]);
			f_tmp1 += vec_w[j]*vec2_means[j][i];
			f_tmp2 += vec_w[j];
		}
		vec_mu[i] = f_tmp1/f_tmp2;
	}
}

// functions for method by Wang et al., Bioinformatics, 2004

void wang_index(Matrix<float >& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	int i_num_features = mat_data.getHeight();

	vector<float> vec_means;
	wang_means(mat_data, vec_index, vec2_index_class, vec_means);

	vec_scores = vector<float>(i_num_features);
	for (int i=0; i<i_num_features; i++)
		vec_scores[i] = fabs(vec_means[i]);

	quicksort_i(vec_scores, vec_sort_index, SORT_DESCENDING);
}


void wang_index(vector<Matrix<float> >& vec_mat_data, vector<vector<int> >& vec2_index, vector<vector<vector<int> > >& vec3_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	int i_num_studies = vec_mat_data.size();
	int i_num_features = vec_mat_data[0].getHeight();

	vector<vector<float> > vec2_means(i_num_studies, vector<float>());
	vector<vector<float> > vec2_vars(i_num_studies, vector<float>());

	for (int n=0; n<i_num_studies; n++) {
		int i_num_class1 = 0;
		for (int i=0; i<vec3_index_class[n][0].size(); i++) if (vec2_index[n][vec3_index_class[n][0][i]] == 0) i_num_class1++;
		int i_num_class2 = 0;
		for (int i=0; i<vec3_index_class[n][1].size(); i++) if (vec2_index[n][vec3_index_class[n][1][i]] == 0) i_num_class2++;

		wang_means(vec_mat_data[n], vec2_index[n], vec3_index_class[n], vec2_means[n]);
		wang_vars(vec_mat_data[n], vec2_index[n], vec3_index_class[n], i_num_class1, i_num_class2, vec2_vars[n]);
	}

	vec_scores = vector<float>(i_num_features);
	for (int i=0; i<i_num_features; i++) {
		float f_num = 0;
		float f_den = 0;
		for (int n=0; n<i_num_studies; n++) {
			float f_recip = 1/vec2_vars[n][i];
			f_num += f_recip*vec2_means[n][i];
			f_den += f_recip;
		}
		vec_scores[i] = fabs(f_num/f_den);
	}

	quicksort_i(vec_scores, vec_sort_index, SORT_DESCENDING);
}

void wang_means(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_means) {
	int i_num_features = mat_data.getHeight();
	vec_means = vector<float>(i_num_features);

	for (int i=0; i<i_num_features; i++) {
		float f_mean_class1 = indexed_mean(mat_data[i], vec_index, vec2_index_class[0]);
		float f_mean_class2 = indexed_mean(mat_data[i], vec_index, vec2_index_class[1]);
		vec_means[i] = f_mean_class1-f_mean_class2;
	}
}

void wang_vars(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_num_class1, int i_num_class2, vector<float>& vec_vars) {
	int i_num_features = mat_data.getHeight();
	vec_vars = vector<float>(i_num_features);

	for (int i=0; i<i_num_features; i++) {

		float f_stdev_class1 = indexed_stdev(mat_data[i], vec_index, vec2_index_class[0]);
		float f_stdev_class2 = indexed_stdev(mat_data[i], vec_index, vec2_index_class[1]);

		vec_vars[i] = f_stdev_class1*f_stdev_class1/(float)i_num_class1+f_stdev_class2*f_stdev_class2/(float)i_num_class2;
	}
}
