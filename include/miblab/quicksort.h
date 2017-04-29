#ifndef __QUICKSORT_H__
#define __QUICKSORT_H__

#include <vector>

#include <miblab/error.h>
#include <miblab/matrix.h>

#define SORT_ASCENDING	0
#define SORT_DESCENDING	1

// sorting
template <class T> void quicksort(vector<T>& ar, int direction = SORT_ASCENDING);
template <class T> void quicksort_recursive(vector<T>& ar, int left, int right, int direction);

template <class T> void quicksort(Matrix<T>& mat, int col, int direction = SORT_ASCENDING);
template <class T> void quicksort_recursive(Matrix<T>& mat, int left, int right, int direction);

// indexed quicksort
template <class T> void quicksort_i(Matrix<T>& mat, vector<int>& index, int col, int direction = SORT_ASCENDING);
template <class T> void quicksort_i(vector<T>& vec, vector<int>& index, int direction = SORT_ASCENDING);
template <class T> void quicksort_i_recursive(Matrix<T>& mat, vector<int>& index, int col, int left, int right, int direction);
template <class T> void quicksort_i_recursive(vector<T>& vec, vector<int>& index, int left, int right, int direction);

// sort by several columns
template <class T> void quicksort_bycol(Matrix<T>& mat, vector<int>& cols, bool allcols = false, int direction = SORT_ASCENDING);
template <class T> void quicksort_bycol_recursive(Matrix<T>& mat, vector<int>& cols, bool allcols, int left, int right, int direction = SORT_ASCENDING);

template <class T> bool isLess(Matrix<T>& mat, vector<int>& cols, int index, int pivot);
template <class T> bool isGreater(Matrix<T>& mat, vector<int>& cols, int index, int pivot);

template <class T>
void quicksort(vector<T>& ar, int direction) {
	quicksort_recursive(ar, 0, ar.size()-1, direction);
}

template <class T>
void quicksort(Matrix<T>& mat, int col, int direction) {
	quicksort_recursive(mat, col, 0, mat.getHeight()-1, direction);
}

template <class T>
void quicksort(Matrix<T>& mat, vector<int>& cols, int col, int direction) {
	quicksort_recursive(mat, cols, col, 0, mat.getHeight()-1, direction);
}

template <class T>
void quicksort_i(Matrix<T>& mat, vector<int>& index, int col, int direction) {
	index = vector<int>(mat.getHeight(),0);
	for (int i=0; i<mat.getHeight(); i++)
		index[i] = i;
	quicksort_i_recursive(mat, index, col, 0, mat.getHeight()-1, direction);
}

template <class T>
void quicksort_i(vector<T>& vec, vector<int>& index, int direction) {
	index = vector<int>(vec.size(),0);
	for (int i=0; i<vec.size(); i++)
		index[i] = i;
	quicksort_i_recursive(vec, index, 0, vec.size()-1, direction);
}

// sort by several columns
template <class T> 
void quicksort_bycol(Matrix<T>& mat, vector<int>& cols, bool allcols, int direction) {
	quicksort_bycol_recursive(mat, cols, allcols, 0, mat.getHeight()-1, direction);
}

template <class T>
bool isLess(Matrix<T>& mat, vector<int>& cols, int index, int pivot) {
	int numcols = cols.size();
	for (int n=0; n<numcols; n++) {
		if (n == numcols-1) {
			if (mat[index][cols[n]] < mat[pivot][cols[n]]) return true;
			return false;	// returns even if equal
		} else {
			if (mat[index][cols[n]] < mat[pivot][cols[n]]) return true;
			if (mat[index][cols[n]] > mat[pivot][cols[n]]) return false;
		}
	}
	return false;
}

template <class T>
bool isGreater(Matrix<T>& mat, vector<int>& cols, int index, int pivot) {
	int numcols = cols.size();
	for (int n=0; n<numcols; n++) {
		if (n == numcols-1) {
			if (mat[index][cols[n]] > mat[pivot][cols[n]]) return true;
			return false;	// returns even if equal
		} else {
			if (mat[index][cols[n]] > mat[pivot][cols[n]]) return true;
			if (mat[index][cols[n]] < mat[pivot][cols[n]]) return false;
		}
	}
	return true;
}

template <class T> void quicksort_bycol_recursive(Matrix<T>& mat, vector<int>& cols, bool allcols, int left, int right, int direction) {
	if (left > right) return;

	int pivot = left;
	int i=left;
	int j=right+1;

	do {
		do {
			i++;
		} while (i < mat.getHeight() && ((direction==SORT_ASCENDING && isLess(mat, cols, i, pivot)) || (direction==SORT_DESCENDING && isGreater(mat, cols, i, pivot))));
		do {
			j--;
		} while ((direction==SORT_ASCENDING && isGreater(mat, cols, j, pivot)) || (direction==SORT_DESCENDING && isLess(mat, cols, j, pivot)));
		if (i < j) {
			if (allcols) {
				vector<T> mat_row;
				mat_row = mat[i];
				mat[i] = mat[j];
				mat[j] = mat_row;
			} else {
				// swap other elements in cols
				int colsize = cols.size();
				T temp;
				for (int n=0; n<colsize; n++) {
					temp = mat[i][cols[n]];
					mat[i][cols[n]] = mat[j][cols[n]];
					mat[j][cols[n]] = temp;
				}
			}
		}
	} while (i < j);

	if (allcols) {
		vector<T> mat_row;
		mat_row = mat[left];
		mat[left] = mat[j];
		mat[j] = mat_row;
	} else {
		// swap other elements in cols
		int colsize = cols.size();
		T temp;
		for (int n=0; n<colsize; n++) {
			temp = mat[left][cols[n]];
			mat[left][cols[n]] = mat[j][cols[n]];
			mat[j][cols[n]] = temp;
		}
	}

	quicksort_bycol_recursive(mat, cols, allcols, left, j-1, direction);
	quicksort_bycol_recursive(mat, cols, allcols, j+1, right, direction);
}


template <class T>
void quicksort_recursive(vector<T>& ar, int left, int right, int direction) {
	if (left > right) return;

	T pivot = ar[left];
	int i=left;
	int j=right+1;

	do {
		do {
			i++;
		} while (i < ar.size() && ((direction==SORT_ASCENDING && ar[i] < pivot) || (direction==SORT_DESCENDING && ar[i] > pivot)));
		do {
			j--;
		} while ( (direction==SORT_ASCENDING && ar[j] > pivot) || (direction==SORT_DESCENDING && ar[j] < pivot) );
		if (i < j) {
			T temp = ar[i];
			ar[i] = ar[j];
			ar[j] = temp;
		}
	} while (i < j);
	T temp = ar[left];
	ar[left] = ar[j];
	ar[j] = temp;
	
	quicksort_recursive(ar, left, j-1, direction);
	quicksort_recursive(ar, j+1, right, direction);
}

template <class T>
void quicksort_recursive(Matrix<T>& mat, int col, int left, int right, int direction) {
	if (left > right) return;

	T pivot = mat[left][col];
	int i=left;
	int j=right+1;

	do {
		do {
			i++;
		} while (i < mat.getHeight() && ((direction==SORT_ASCENDING && mat[i][col] < pivot) || (direction==SORT_DESCENDING && mat[i][col] > pivot)));
		do {
			j--;
		} while ((direction==SORT_ASCENDING && mat[j][col] > pivot) || (direction==SORT_DESCENDING && mat[j][col] < pivot));
		if (i < j) {
			T temp = mat[i][col];
			mat[i][col] = mat[j][col];
			mat[j][col] = temp;
		}
	} while (i < j);
	T temp = mat[left][col];
	mat[left][col] = mat[j][col];
	mat[j][col] = temp;

	quicksort_recursive(mat, col, left, j-1, direction);
	quicksort_recursive(mat, col, j+1, right, direction);
}

template <class T>
void quicksort_i_recursive(Matrix<T>& mat, vector<int>& index, int col, int left, int right, int direction) {
	if (left > right) return;

	T pivot = mat[left][col];
	int i=left;
	int j=right+1;

	do {
		do {
			i++;
		} while (i < mat.getHeight() && ((direction==SORT_ASCENDING && mat[i][col] < pivot) || (direction==SORT_DESCENDING && mat[i][col] > pivot)));
		do {
			j--;
		} while ((direction==SORT_ASCENDING && mat[j][col] > pivot) || (direction==SORT_DESCENDING && mat[j][col] < pivot));
		if (i < j) {
			T temp = mat[i][col];
			mat[i][col] = mat[j][col];
			mat[j][col] = temp;

			int tempi = index[i];
			index[i] = index[j];
			index[j] = tempi;
		}
	} while (i < j);
	T temp = mat[left][col];
	mat[left][col] = mat[j][col];
	mat[j][col] = temp;

	int tempi = index[left];
	index[left] = index[j];
	index[j] = tempi;

	quicksort_i_recursive(mat, index, col, left, j-1, direction);
	quicksort_i_recursive(mat, index, col, j+1, right, direction);
}

template <class T>
void quicksort_i_recursive(vector<T>& vec, vector<int>& index, int left, int right, int direction) {
	if (left > right) return;

	T pivot = vec[left];
	int i=left;
	int j=right+1;

	do {
		do {
			i++;
		} while (i < vec.size() && ((direction==SORT_ASCENDING && vec[i] < pivot) || (direction==SORT_DESCENDING && vec[i] > pivot)));
		do {
			j--;
		} while ((direction==SORT_ASCENDING && vec[j] > pivot) || (direction==SORT_DESCENDING && vec[j] < pivot));
		if (i < j) {
			T temp = vec[i];
			vec[i] = vec[j];
			vec[j] = temp;

			int tempi = index[i];
			index[i] = index[j];
			index[j] = tempi;
		}
	} while (i < j);
	T temp = vec[left];
	vec[left] = vec[j];
	vec[j] = temp;

	int tempi = index[left];
	index[left] = index[j];
	index[j] = tempi;

	quicksort_i_recursive(vec, index, left, j-1, direction);
	quicksort_i_recursive(vec, index, j+1, right, direction);
}

#endif
