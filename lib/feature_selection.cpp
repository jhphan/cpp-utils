#include <miblab/feature_selection.h>

// --- utility functions
void compute_ranks(
	vector<int>& vec_sort_index,
	vector<int>& vec_ranks
) {
	int i_num_rows = vec_sort_index.size();
	vec_ranks = vector<int>(i_num_rows);
	for (int i=0; i<i_num_rows; i++)
		vec_ranks[vec_sort_index[i]] = i;
}

// compute ranks, allowing for ties
// ranks for ties are replaced with average ranks
void compute_ranks_w_ties(
	vector<float>& vec_data,
	vector<int>& vec_sort_index,
	vector<float>& vec_ranks
) {
	int i_num_vals = vec_sort_index.size();
	vector<int> vec_i_ranks = vector<int>(i_num_vals);
	vec_i_ranks[vec_sort_index[0]] = 0;
	map<int,float> map_avg_rank;

	float f_cur_val = vec_data[0];
	int i_cur_rank = 0;
	int i_num_ties = 1;
	float f_sum_rank = 0;
	for (int i=1; i<i_num_vals; i++) {
		if (fabs(vec_data[i]-f_cur_val) < FLT_EPS) {	// equal
			vec_i_ranks[vec_sort_index[i]] = i_cur_rank;

			f_sum_rank += i;
			i_num_ties++;
		} else {	// unequal
			f_cur_val = vec_data[i];
			map_avg_rank[i_cur_rank] = f_sum_rank/(float)i_num_ties;

			i_cur_rank++;
			vec_i_ranks[vec_sort_index[i]] = i_cur_rank;
			
			f_sum_rank = i;
			i_num_ties = 1;
		}
	}
	map_avg_rank[i_cur_rank] = f_sum_rank/(float)i_num_ties;

	// replace matching ranks with average ranks
	vec_ranks = vector<float>(i_num_vals);
	for (int i=0; i<i_num_vals; i++)
		vec_ranks[i] = map_avg_rank[vec_i_ranks[i]];
}

// this function is limited by the size of 'long'
int n_choose_k(int n, int k) {
	long l_num = 1;
	long l_den = 1;
	for (int i=n; i>(n-k); i--)
		l_num*=i;
	for (int i=k; i>1; i--)
		l_den*=i;
	return l_num/l_den;
}

// compute the mean of elements in vec_data that are indexed
float indexed_mean(
	vector<float>& vec_data,
	vector<int>& vec_index,
	vector<int>& vec_index_class
) {
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
float indexed_mean(
	vector<float>& vec_data,
	vector<int>& vec_index
) {
	float f_sum = 0;
	int i_num = 0;
	int i_max = vec_index.size();
	for (int i=0; i<i_max; i++)
		if (vec_index[i] == 0) {
			i_num++;
			f_sum+=vec_data[i];
		}
	return f_sum/(float)i_num;
}
// compute the stdev of elements in vec_data that are indexed
float indexed_stdev(
	vector<float>& vec_data,
	vector<int>& vec_index,
	int i_bias
) {
	int i_max = vec_index.size();
	int i_num = 0;
	float f_mean_sq = 0;
	float f_sum_sq = 0;
	for (int i=0; i<i_max; i++) {
		if (vec_index[i] == 0) {
			i_num++;
			float f_data = vec_data[i];
			f_mean_sq += f_data;
			f_sum_sq += f_data*f_data;
		}
	}
	f_mean_sq = f_mean_sq/(float)i_num;
	int i_denom = 0;
	if (i_bias)
		i_denom = i_num;
	else
		i_denom = (i_num-1);
	return sqrt(
			(1/(float)i_denom)
			*f_sum_sq
			-(i_num/(float)i_denom)
			*f_mean_sq
			*f_mean_sq
		);
}
// compute the stdev of elements in vec_data that are indexed
float indexed_stdev(
	vector<float>& vec_data,
	vector<int>& vec_index,
	vector<int>& vec_index_class,
	int i_bias
) {
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
	int i_denom = 0;
	if (i_bias)
		i_denom = i_num;
	else
		i_denom = (i_num-1);
	return sqrt(
			(1/(float)i_denom)
			*f_sum_sq
			-(i_num/(float)i_denom)
			*f_mean_sq
			*f_mean_sq
		);
}

// assume gene expression values are in the log domain
// data has no class labels row
void fold_change_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
) {
	int i_num_rows = mat_data.getHeight();

	// compute fold change values
	vec_scores = vector<float>(i_num_rows);
	for (int i=0; i<i_num_rows; i++)
		vec_scores[i] = compute_fold_change(mat_data, vec_index, vec2_index_class, i);

	// get the sort index
	quicksort_i(vec_scores, vec_sort_index, SORT_DESCENDING);
}
float compute_fold_change(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row
) {
	float f_mean_class1 = indexed_mean(mat_data[i_row], vec_index, vec2_index_class[0]);
	float f_mean_class2 = indexed_mean(mat_data[i_row], vec_index, vec2_index_class[1]);

	return fabs(f_mean_class1-f_mean_class2);
}

void t_test_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
) {
	int i_num_rows = mat_data.getHeight();

	int i_num_class1 = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++)
		if (vec_index[vec2_index_class[0][i]] == 0)
			i_num_class1++;
	int i_num_class2 = 0;
	for (int i=0; i<vec2_index_class[1].size(); i++)
		if (vec_index[vec2_index_class[1][i]] == 0)
			i_num_class2++;

	vec_scores = vector<float>(i_num_rows);
	for (int i=0; i<i_num_rows; i++)
		vec_scores[i] = compute_t_stat_p_value(
			mat_data,
			vec_index,
			vec2_index_class,
			i,
			i_num_class1,
			i_num_class2
		);

	quicksort_i(vec_scores, vec_sort_index, SORT_ASCENDING);
}

// t statistic of samples of unequal size, but equal and unknown variances
float compute_t_stat(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	int i_num_class2
) {
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
	return f_t_stat;
}

float compute_t_stat_p_value(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	int i_num_class2
) {
	float f_t_stat = compute_t_stat(
		mat_data,
		vec_index,
		vec2_index_class,
		i_row,
		i_num_class1,
		i_num_class2
	);

	int i_df = i_num_class1+i_num_class2-2;
	students_t dist(i_df);
	float f_p = 2*cdf(complement(dist, f_t_stat));

	return f_p;
}

// -- Significance Analysis of Microarrays
void sam_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
) {
	int i_num_rows = mat_data.getHeight();

	int i_num_class1 = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++)
		if (vec_index[vec2_index_class[0][i]] == 0)
			i_num_class1++;
	int i_num_class2 = 0;
	for (int i=0; i<vec2_index_class[1].size(); i++)
		if (vec_index[vec2_index_class[1][i]] == 0)
			i_num_class2++;
	
	vector<float> vec_mean_class1;
	vector<float> vec_mean_class2;
	vector<float> vec_pooled_stdev;
	compute_sam_stats(
		mat_data,
		vec_index,
		vec2_index_class,
		i_num_class1,
		i_num_class2,
		vec_mean_class1,
		vec_mean_class2,
		vec_pooled_stdev
	);

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

void compute_sam_stats(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_num_class1,
	int i_num_class2,
	vector<float>& vec_mean_class1,
	vector<float>& vec_mean_class2,
	vector<float>& vec_pooled_stdev
) {
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

// --- Rank Sum Test feature selection
void ranksum_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index,
	bool b_exact
) {
	int i_num_rows = mat_data.getHeight();
	
	int i_num_class1 = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++)
		if (vec_index[vec2_index_class[0][i]] == 0)
			i_num_class1++;
	int i_num_class2 = 0;
	for (int i=0; i<vec2_index_class[1].size(); i++)
		if (vec_index[vec2_index_class[1][i]] == 0)
			i_num_class2++;

	vec_scores = vector<float>(i_num_rows);
	int i_num_small = 0;
	for (int i=0; i<i_num_rows; i++)
		vec_scores[i] = compute_ranksum_stat_p_value(
			mat_data,
			vec_index,
			vec2_index_class,
			i,
			i_num_class1,
			i_num_class2,
			b_exact
		);

	quicksort_i(vec_scores, vec_sort_index, SORT_ASCENDING);
}
// recursively compute ranksum distribution
//int ranksum_dist(int n, int N, int w) {
int ranksum_dist(
	int n1,
	int n2,
	int u
) {
	int i_max = n1*n2;

	if (u < 0 || u > n1*n2 || n1 < 0 || n2 < 0) return 0;
	if (u == 0) return 1;
	return ranksum_dist(n1-1, n2, u-n2)+ranksum_dist(n1, n2-1, u);

	/*  ## alternative method
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
	return val;*/
}
float compute_ranksum_stat(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	int i_num_class2,
	int& i_num_small
) {
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
	vector<float> vec_ranks;
	quicksort_i(vec_data, vec_sort_index, SORT_ASCENDING);
	compute_ranks_w_ties(vec_data, vec_sort_index, vec_ranks);
	for (int i=0; i<i_total_samples; i++)
		vec_ranks[i] += 1.0;

	//compute the ranksum statistic
	float f_stat = 0;
	for (int i=0; i<i_num_class1; i++)
		f_stat += vec_ranks[i];
	f_stat -= i_num_class1*(i_num_class1+1)/2;
	// use the smaller statistic
	int i_upper = i_num_class1*(i_total_samples-i_num_class1);
	i_num_small = i_num_class1;
	if (f_stat > i_upper/2) {
		f_stat = i_upper-f_stat;
		i_num_small = i_num_class2;
	}

	return f_stat;
}

float compute_ranksum_stat_p_value(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	int i_num_class2,
	bool b_exact
) {

	int i_total_samples = i_num_class1+i_num_class2;
	int i_num_small = 0;
	float f_stat = compute_ranksum_stat(
		mat_data,
		vec_index,
		vec2_index_class,
		i_row,
		i_num_class1,
		i_num_class2,
		i_num_small
	);

	// compute p-val
	float f_pval = 0;
	if (!b_exact) {
		// approximate with normal distribution
		float f_mu = i_num_class1*i_num_class2/2.0;
		float f_stdev = sqrt( f_mu*(i_total_samples+1)/6.0 );
		// add 0.5 for continuity correction
		float f_zstat = (f_stat+0.5-f_mu)/f_stdev;
		normal norm;
		f_pval = 2*cdf(norm, f_zstat);
	} else {
		// compute exactly
		int i_total_perms = n_choose_k(i_total_samples, i_num_small);
		int i_sum_freq = 0;
		for (int w=0; w<=(int)f_stat; w++)
			i_sum_freq += ranksum_dist(i_num_small, i_total_samples-i_num_small, w);
			//i_sum_freq += ranksum_dist(i_num_small, i_total_samples, w);
		f_pval = 2*i_sum_freq/(float)i_total_perms;
	}
	if (f_pval > 1.0) f_pval = 1.0;
	return f_pval;
}

// --- minimum Redundance Maximum Relevance feature selection
void mrmr_index(
	Matrix<float>& mat_data,
	vector<float>& vec_labels,
	vector<int>& vec_index,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index,
	int i_max_num,
	bool b_miq
) {
	int i_num_rows = mat_data.getHeight();
	int i_num_cols = mat_data.getWidth();

	// z-score normalize, then discretize the data
	Matrix<float> mat_data_zscore(i_num_rows, i_num_cols);
	for (int i=0; i<i_num_rows; i++) {
		float f_mean = indexed_mean(mat_data[i], vec_index);
		float f_stdev = indexed_stdev(mat_data[i], vec_index);
		if ( f_stdev > FLT_EPS ) { // non-zero stdev
			for (int j=0; j<i_num_cols; j++) {
				mat_data_zscore[i][j] = (mat_data[i][j]-f_mean)/f_stdev;
				if (mat_data_zscore[i][j] < -0.5) {
					mat_data_zscore[i][j] = -1;
				} else if (mat_data_zscore[i][j] > 0.5) {
					mat_data_zscore[i][j] = 1;
				} else {
					mat_data_zscore[i][j] = 0;
				}					
			}
		} else {
			// skip dividing by zero, since all the values are the same anyway
			// this essentially sets everything to zero
			for (int j=0; j<i_num_cols; j++)
				mat_data_zscore[i][j] = 0;
		}
	}

	// calculate mutual info of all features with labels
	vector<float> vec_mut_info(i_num_rows,0);
	for (int i=0; i<i_num_rows; i++)
		vec_mut_info[i] = compute_mutual_info(vec_labels, mat_data_zscore[i], vec_index);

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
					mat_mut_info[k][j] = compute_mutual_info(mat_data_zscore[i_idx_j], mat_data_zscore[i_idx_k], vec_index);
					//cout << "new mut: " << mat_mut_info[k][j] << endl;
				}
				f_red_val += mat_mut_info[k][j];
			}
			f_red_val /= i;

			float f_score = 0;
			if (b_miq) 
				//f_score = f_rel_val/(f_red_val+FLT_EPS);
				f_score = f_rel_val/(f_red_val+0.01);
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

int discretize_vector(
	vector<float>& vec_float,
	vector<int>& vec_int,
	vector<int>& vec_index
) {
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
			int i_tmp = (f_tmp > 0) ? int(f_tmp+0.5) : int(f_tmp-0.5);
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

void compute_joint_prob(
	Matrix<float>& mat_joint_prob,
	vector<float>& vec_f1,
	vector<float>& vec_f2,
	vector<int>& vec_index
) {
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

float compute_mutual_info(
	vector<float>& vec_f1,
	vector<float>& vec_f2,
	vector<int>& vec_index
) {
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

// rank product feature select: Breitling et al. FEBS Letters, 2004
void rankprod_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
) {
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
// pre-compute rank product data for quicker processing
void rankprod_precompute(
	Matrix<float>& mat_data,
	vector<vector<int> >& vec2_index_class,
	Matrix<vector<float> >& mat_scores
) {
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
// use the pre-computed scores to compute feature indexes
void rankprod_index_quick(
	Matrix<vector<float> >& mat_scores,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
) {
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

// --- functions for method by Choi et al, Bioinformatics, 2003
void choi_index(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	 vector<int>& vec_sort_index
) {
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

void choi_index(
	vector<Matrix<float> >& vec_mat_data,
	vector<vector<int> >& vec2_index,
	vector<vector<vector<int> > >& vec3_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
) {
	int i_num_studies = vec_mat_data.size();
	int i_num_features = vec_mat_data[0].getHeight();

	vector<vector<float> > vec2_means(i_num_studies, vector<float>());
	vector<vector<float> > vec2_vars(i_num_studies, vector<float>());

	for (int n=0; n<i_num_studies; n++) {
		int i_num_class1 = 0;
		for (int i=0; i<vec3_index_class[n][0].size(); i++)
			if (vec2_index[n][vec3_index_class[n][0][i]] == 0)
				i_num_class1++;
		int i_num_class2 = 0;
		for (int i=0; i<vec3_index_class[n][1].size(); i++)
			if (vec2_index[n][vec3_index_class[n][1][i]] == 0)
				i_num_class2++;
		
		choi_dstar(
			vec_mat_data[n],
			vec2_index[n],
			vec3_index_class[n],
			i_num_class1,
			i_num_class2,
			vec2_means[n]
		);
		choi_sigmad(
			vec2_means[n],
			i_num_class1,
			i_num_class2,
			 vec2_vars[n]
		);
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

float choi_t_stat(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_row,
	int i_num_class1,
	int i_num_class2
) {
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
void choi_dstar(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_num_class1,
	int i_num_class2,
	vector<float>& vec_dstar
) {
	int i_num_features = mat_data.getHeight();

	vec_dstar = vector<float>(i_num_features);
	for (int i=0; i<i_num_features; i++) {
		vec_dstar[i] = choi_t_stat(
			mat_data,
			vec_index,
			vec2_index_class,
			i,
			i_num_class1,
			i_num_class2
		);
		vec_dstar[i] = vec_dstar[i]-3*vec_dstar[i]/(4*(i_num_class1+i_num_class2-2)-1);
	}
}

// compute estimate of variance of unbiased d
void choi_sigmad(
	vector<float>& vec_dstar,
	int i_num_class1,
	int i_num_class2,
	vector<float>& vec_sigmad
) {
	int i_num_features = vec_dstar.size();
	vec_sigmad = vector<float>(i_num_features);

	for (int i=0; i<i_num_features; i++)
		vec_sigmad[i] = 1/(float)i_num_class1
			+1/(float)i_num_class2
			+vec_dstar[i]*vec_dstar[i]
				/((float)2*(i_num_class1+i_num_class2));
}

void choi_Q(
	vector<vector<float> >& vec2_means,
	vector<vector<float> >& vec2_vars,
	vector<float>& vec_q
) {
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

void choi_tau2DL(
	vector<float>& vec_q,
	vector<vector<float> >& vec2_vars,
	vector<float>& vec_tau2DL
) {
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

void choi_mutau2(
	vector<vector<float> >& vec2_means,
	vector<vector<float> >& vec2_vars,
	vector<float>& vec_tau2,
	vector<float>& vec_mu
) {
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

// --- functions for method by Wang et al., Bioinformatics, 2004
void wang_index(
	Matrix<float >& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
) {
	int i_num_features = mat_data.getHeight();

	vector<float> vec_means;
	wang_means(mat_data, vec_index, vec2_index_class, vec_means);

	vec_scores = vector<float>(i_num_features);
	for (int i=0; i<i_num_features; i++)
		vec_scores[i] = fabs(vec_means[i]);

	quicksort_i(vec_scores, vec_sort_index, SORT_DESCENDING);
}

void wang_index(
	vector<Matrix<float> >& vec_mat_data,
	vector<vector<int> >& vec2_index,
	vector<vector<vector<int> > >& vec3_index_class,
	vector<float>& vec_scores,
	vector<int>& vec_sort_index
) {
	int i_num_studies = vec_mat_data.size();
	int i_num_features = vec_mat_data[0].getHeight();

	vector<vector<float> > vec2_means(i_num_studies, vector<float>());
	vector<vector<float> > vec2_vars(i_num_studies, vector<float>());

	for (int n=0; n<i_num_studies; n++) {
		int i_num_class1 = 0;
		for (int i=0; i<vec3_index_class[n][0].size(); i++)
			if (vec2_index[n][vec3_index_class[n][0][i]] == 0)
				i_num_class1++;
		int i_num_class2 = 0;
		for (int i=0; i<vec3_index_class[n][1].size(); i++)
			if (vec2_index[n][vec3_index_class[n][1][i]] == 0)
				i_num_class2++;

		wang_means(
			vec_mat_data[n],
			vec2_index[n],
			vec3_index_class[n],
			vec2_means[n]
		);
		wang_vars(
			vec_mat_data[n],
			vec2_index[n],
			vec3_index_class[n],
			i_num_class1,
			i_num_class2,
			vec2_vars[n]
		);
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

void wang_means(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	vector<float>& vec_means
) {
	int i_num_features = mat_data.getHeight();
	vec_means = vector<float>(i_num_features);

	for (int i=0; i<i_num_features; i++) {
		float f_mean_class1 = indexed_mean(mat_data[i], vec_index, vec2_index_class[0]);
		float f_mean_class2 = indexed_mean(mat_data[i], vec_index, vec2_index_class[1]);
		vec_means[i] = f_mean_class1-f_mean_class2;
	}
}

void wang_vars(
	Matrix<float>& mat_data,
	vector<int>& vec_index,
	vector<vector<int> >& vec2_index_class,
	int i_num_class1,
	int i_num_class2,
	vector<float>& vec_vars
) {
	int i_num_features = mat_data.getHeight();
	vec_vars = vector<float>(i_num_features);

	for (int i=0; i<i_num_features; i++) {

		float f_stdev_class1 = indexed_stdev(mat_data[i], vec_index, vec2_index_class[0]);
		float f_stdev_class2 = indexed_stdev(mat_data[i], vec_index, vec2_index_class[1]);

		vec_vars[i] = f_stdev_class1*f_stdev_class1
			/(float)i_num_class1
			+f_stdev_class2*f_stdev_class2
				/(float)i_num_class2;
	}
}

// use the libc random function to fill vec_features with 
// random indexes, vec_features is pre-allocated
void random_feature_indexes(
	int i_seed,
	int i_num_features,
	vector<int>& vec_features
) {
	srand(i_seed);
	for (int i=0; i<vec_features.size(); i++) {
		vec_features[i] = rand()%i_num_features;
	}
}




//rmp: threshold by Jarque-Bera Gaussian-ness (more likely to be Gaussian) and rank by p-value from Kolmogorov-Smirnov test (smaller chance of being same distribution)
void jbks_test_index(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, vector<float>& vec_scores, vector<int>& vec_sort_index) {
	int i_num_rows = mat_data.getHeight();

	int i_num_class1 = 0;
	for (int i=0; i<vec2_index_class[0].size(); i++) if (vec_index[vec2_index_class[0][i]] == 0) i_num_class1++;
	int i_num_class2 = 0;
	for (int i=0; i<vec2_index_class[1].size(); i++) if (vec_index[vec2_index_class[1][i]] == 0) i_num_class2++;

	vec_scores = vector<float>(i_num_rows);
	for (int i=0; i<i_num_rows; i++) {
		float f_jb_pval1 = indexed_compute_jb_test(mat_data[i], vec_index, vec2_index_class[0]);
		float f_jb_pval2 = indexed_compute_jb_test(mat_data[i], vec_index, vec2_index_class[1]);
		float f_ks_pval = compute_ks_stat(mat_data, vec_index, vec2_index_class, i, i_num_class1, i_num_class2);
		if (f_jb_pval1 >= 0.025 && f_jb_pval2 >= 0.025)
			vec_scores[i] = f_ks_pval;
		else
			vec_scores[i] = 1 + f_ks_pval; // put non-Gaussians at end of the list.
	}

	quicksort_i(vec_scores, vec_sort_index, SORT_ASCENDING);
}

// Kolmogorov-Smirnov test of equal distributions
float compute_ks_stat(Matrix<float>& mat_data, vector<int>& vec_index, vector<vector<int> >& vec2_index_class, int i_row, int i_num_class1, int i_num_class2) {
	float f_p,f_ks_stat;
	vector<float> vec_1;
	vector<float> vec_2;

        for (int i=0; i<vec2_index_class[0].size(); i++)
                if (vec_index[vec2_index_class[0][i]] == 0)
                        vec_1.push_back(mat_data[i_row][vec2_index_class[0][i]]);

        for (int i=0; i<vec2_index_class[1].size(); i++)
                if (vec_index[vec2_index_class[1][i]] == 0)
                        vec_2.push_back(mat_data[i_row][vec2_index_class[1][i]]);
 
	kstwo(vec_1,vec_2,&f_ks_stat,&f_p);
	return f_p;
}

// numerical recipes in c
//#define EPS1 0.001
//#define EPS2 1.0e-8
float probks(float alam){
	// Kolmogorov-Smirnov probability function
	const float EPS1 = 0.001;
	const float EPS2 = 1.0e-8;
	int j;
	float a2,fac=2.0,sum=0.0,term,termbf=0.0;
	a2 = -2.0*alam*alam;
	for (j=1;j<=100;j++){
		term=fac*exp(a2*j*j);
		sum += term;
		if (fabs(term) <= EPS1*termbf || fabs(term) <= EPS2*sum) return sum;
		fac = -fac; // alternating signs in sum
		termbf=fabs(term);

	}
	return 1.0; // Get here only by failing to converge
}

// numerical recipes in c
void kstwo(vector<float> &v1, vector<float> &v2, float *d, float *prob){
	// Given an array data1[1..n1], and an array data2[1..n2], this routine returns the K-S statistic 'd',
	// and the significance level 'prob' for the null hypothesis that the data sets are drawn from the same
	// distribution.  Small values of 'prob' show that the cumulative distribution function of 'data1' is
	// significantly different from that of 'data2'.  The arrays 'data1' and 'data2' are modified by being 
	// sorted into ascending order.	
	unsigned long j1=1,j2=1;
	float d1,d2,dt,en1,en2,en,fn1=0.0,fn2=0.0;
	int n1=v1.size();
	int n2=v2.size();
	vector<float> data1(v1);
	vector<float> data2(v2);
	quicksort(data1);
	quicksort(data2);
	en1=n1;
	en2=n2;
	*d=0.0;
	while (j1 <= n1 && j2 <= n2) {
		if ((d1=data1[j1-1]) <= (d2=data2[j2-1])) fn1=j1++/en1;
		if (d2 <= d1) fn2=j2++/en2;
		if ((dt=fabs(fn2-fn1)) > *d) *d=dt;
	}
	en=sqrt(en1*en2/(en1+en2));
	*prob=probks((en+0.12+0.11/en)*(*d));
}



// rmp: compute skewness of elements in vec_data that are indexed
float indexed_skewness(vector<float>& vec_data, vector<int>& vec_index, vector<int>& vec_index_class) {
	float f_mean = indexed_mean(vec_data,vec_index,vec_index_class);
	float f_std = indexed_stdev(vec_data,vec_index,vec_index_class,1);

        int i_max = vec_index_class.size();
        int i_num = 0;
        float f_skew = 0;
        for (int i=0; i<i_max; i++) {
                if (vec_index[vec_index_class[i]] == 0) {
                        i_num++;
                        float f_data = vec_data[vec_index_class[i]];
			float f_zscore = (f_data - f_mean) / f_std;
                        f_skew += f_zscore * f_zscore * f_zscore;
                }
        }
        f_skew = f_skew / (float)i_num;
        return f_skew;	
}
// rmp: compute kurtosis excess of elements in vec_data that are indexed
float indexed_kurtosis(vector<float>& vec_data, vector<int>& vec_index, vector<int>& vec_index_class) {
        float f_mean = indexed_mean(vec_data,vec_index,vec_index_class);
        float f_std = indexed_stdev(vec_data,vec_index,vec_index_class,1);

        int i_max = vec_index_class.size();
        int i_num = 0;
        float f_kurt = 0;
        for (int i=0; i<i_max; i++) {
                if (vec_index[vec_index_class[i]] == 0) {
                        i_num++;
                        float f_data = vec_data[vec_index_class[i]];
                        float f_zscore = (f_data - f_mean) / f_std;
			f_zscore *= f_zscore; // z^2
                        f_kurt += f_zscore * f_zscore; // z^4
                }
        }
        f_kurt = f_kurt / (float)i_num;
	f_kurt -= 3;
        return f_kurt;
}
// from numerical recipes
void spline(const double x[], const double y[], int n, double yp1, double ypn, double y2[])
// Given arrays x[1..n] and y[1..n] containing a tabulated function, i.e., yi = f(xi), with
// x1 < x2 < ... < xN, and given values yp1 and ypn for the first derivative of the interpolating
// function at points 1 and n, respectively, this routine returns an array y2[1..n] that contains
// the second derivatives of the interpolating function at the tabulated points xi. If yp1 and/or
// ypn are equal to 1e30 or larger, the routine is signaled to set the corresponding boundary
// condition for a natural spline, with zero second derivative on that boundary.*/
{
        int i,k;
        double p,qn,sig,un,*u;
        u=new double[n];         // u=vector(1,n-1);
        if (yp1 > 0.99e30)      // The lower boundary condition is set either to be natural
                y2[1]=u[1]=0.0;
        else {                  // or else to have a specified first derivative.
                y2[1] = -0.5;
                u[1]=(3.0/(x[2]-x[1]))*((y[2]-y[1])/(x[2]-x[1])-yp1);
        }
        for (i=2;i<=n-1;i++) {  // This is the decomposition loop of the tridiagonal algorithm.
                                // y2 and u are used for temporary
                                // storage of the decomposed
                                // factors.
                sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
                p=sig*y2[i-1]+2.0;
                y2[i]=(sig-1.0)/p;
                u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
                u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
        }
        if (ypn > 0.99e30)      // The upper boundary condition is set either to be "natural"
                qn=un=0.0;
        else {                  // or else to have a specified first derivative.
                qn=0.5;
                un=(3.0/(x[n]-x[n-1]))*(ypn-(y[n]-y[n-1])/(x[n]-x[n-1]));
        }
        y2[n]=(un-qn*u[n-1])/(qn*y2[n-1]+1.0);
        for (k=n-1;k>=1;k--)    // This is the backsubstitution loop of the tridiagonal algorithm.
                y2[k]=y2[k]*y2[k+1]+u[k];
        delete u;               // free_vector(u,1,n-1);
}

void splint(const double xa[], const double ya[], double y2a[], int n, double x, double *y)                                                                                                          // Given the arrays xa[1..n] and ya[1..n], which tabulate a function (with the xai's in order),
// and given the array y2a[1..n], which is the output from spline above, and given a value of                                                                                                        // x, this routine returns a cubic-spline interpolated value y.
{                                                                                                                                                                                                            // void nrerror(char error_text[]);
        int klo,khi,k;                                                                                                                                                                                       double h,b,a;
        klo=1;  /* We will find the right place in the table by means of                                                                                                                                             bisection. This is optimal if sequential calls to this
                routine are at random values of x. If sequential calls                                                                                                                                               are in order, and closely spaced, one would do better
                to store previous values of klo and khi and test if                                                                                                                                                  they remain appropriate on the next call. */
        khi=n;                                                                                                                                                                                               while (khi-klo > 1) {
                k=(khi+klo) >> 1;                                                                                                                                                                                    if (xa[k] > x) khi=k;
                else klo=k;                                                                                                                                                                                  } // klo and khi now bracket the input value of x.
        h=xa[khi]-xa[klo];                                                                                                                                                                                   if (h == 0.0) cerr << "Bad xa input to routine splint." << endl; // The xa's must be distinct.
        a=(xa[khi]-x)/h;
        b=(x-xa[klo])/h;        // Cubic spline polynomial is now evaluated.
        *y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
}

float indexed_compute_jb_test(vector<float>& vec_data, vector<int>& vec_index, vector<int>& vec_index_class) {
        double rev_CVs[_RMP_N_ALPHAS];
        double CVs[_RMP_N_ALPHAS];
        double y2[_RMP_N_SAMPLESIZES];
        double y3[_RMP_N_ALPHAS];
	float f_p;

	int i_num = 0;
	int i_max = vec_index_class.size();
	for (int i=0;i<i_max;i++) if (vec_index[vec_index_class[i]] == 0) i_num++;

	// based on matlab's jbtest
	float f_skew = indexed_skewness(vec_data,vec_index,vec_index_class);
	float f_kurt = indexed_kurtosis(vec_data,vec_index,vec_index_class);

	// jarque-bera test statistic
	float f_jb_stat = (float)i_num * (f_skew*f_skew/6.0 + f_kurt*f_kurt/24.0);

	//compute p-value using critical value table 

        // get row of critical value table for the current sample size.
        for (int i=0; i<_RMP_N_ALPHAS; i++){
                spline(_RMP_SAMPLESIZES-1,&_RMP_CRITICAL_VALUES[i][0]-1,_RMP_N_SAMPLESIZES,1e30,1e30,y2-1);
                splint(_RMP_SAMPLESIZES-1,&_RMP_CRITICAL_VALUES[i][0]-1,y2-1,_RMP_N_SAMPLESIZES,(double)i_num,&CVs[i]);
        }
        // interpolate p-value
        if (f_jb_stat < CVs[_RMP_N_ALPHAS-1]){
                f_p = _RMP_ALPHAS[_RMP_N_ALPHAS-1];
        } else if (CVs[0] < f_jb_stat) {
                f_p = _RMP_ALPHAS[0];
        } else {
                double p=-1;
                // must sort CVs and corresponding alphas into non-decreasing order.
                for (int i=0;i<_RMP_N_ALPHAS;i++) rev_CVs[i]=CVs[_RMP_N_ALPHAS-i-1];
                spline(rev_CVs-1,_RMP_REV_ALPHAS-1,_RMP_N_ALPHAS,1e30,1e30,y3-1);
                splint(rev_CVs-1,_RMP_REV_ALPHAS-1,y3-1,_RMP_N_ALPHAS,f_jb_stat,&p);
                f_p=p;
        }
        return f_p;
}



