#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <iostream>
#include <vector>
#include <math.h>

#include <miblab/error.h>

using namespace std;

template <class T>
class Matrix;

template <class T>
ostream& operator<< (ostream& ostr, const Matrix<T> &m);

template <class T>
class Matrix : public vector< vector<T> > {

protected:
	int m_width;
	
	void copy(const Matrix<T>& m);
	
	T dotProduct(Matrix<T>* m1, int row, const Matrix<T>* m2, int col);
	
public:
	Matrix(int height=0, int width=0, T val=T());
	Matrix(const Matrix<T>& m);
	~Matrix();

	Matrix<T>& operator=(const Matrix<T>& m);

	int getHeight() const { return vector< vector<T> >::size(); };
	int getWidth() const { return m_width; };

	Matrix<T> operator~();
	Matrix<T> operator*(const Matrix<T>& mat);
	Matrix<T> operator*(const T val);
	Matrix<T> operator+(const Matrix<T>& mat);
	Matrix<T> operator+(const T val);
	Matrix<T> operator-(const Matrix<T>& mat);
	Matrix<T> operator-(const T val);
	Matrix<T> operator/(const T val);

	// range function
	Matrix<T> operator()(int rstart, int rend, int cstart, int cend);

	// append functions
	Matrix<T> operator()(const Matrix<T>& mat, int dummy); // add rows
	Matrix<T> operator()(int dummy, const Matrix<T>& mat); // add cols

	// dynamic append functions
	bool appendRow(const Matrix<T>& mat);
	bool appendCol(const Matrix<T>& mat);

	void push_back(const vector<T>& row);
	void setData(int h, int w, const Matrix<T>& mat);

	int find(int col, T val, int start = 0);

	vector<T> getColumn(int col);

	friend ostream& operator<< <> (ostream& ostr, const Matrix<T>& mat);
};


template <class T>
T Matrix<T>::dotProduct(Matrix<T>* m1, int row, const Matrix<T>* m2, int col) {
	if ((row < 0) || (row >= m1->getHeight()))
		throw Error("!exception: Matrix::dotProduct(), row out of range");
	if ((col < 0) || (col >= m2->getWidth()))
		throw Error("!exception: Matrix::dotProduct(), col out of range");

	T val = 0;
	for (int i=0; i<m1->getWidth(); i++)
		val += (*m1)[row][i]*(*m2)[i][col];
	return val;
}

template <class T>
Matrix<T>::Matrix(int height, int width, T val) : vector< vector<T> >(height) {
	m_width = width;
	for (int i=0; i<height; i++) {
		(*this)[i] = vector<T>(width, val);
	}
}

template <class T>
Matrix<T>::Matrix(const Matrix<T>& m) {
	copy(m);
}

template <class T>
Matrix<T>::~Matrix() {
}

template <class T>
void Matrix<T>::copy(const Matrix<T>& m) {
	vector< vector<T> >::clear();	// clear all contents from the matrix
	
	int height = m.getHeight();
	m_width = m.getWidth();
	
	for(int i=0;i<height;i++)
		push_back(m[i]);
	
}

template <class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& m) {
	copy(m);
	return *this;
}

// transpose matrix
template <class T>
Matrix<T> Matrix<T>::operator~() {
	int height = vector< vector<T> >::size();
	if (m_width == 0 || height == 0)
		return Matrix<T>();	// empty object
	Matrix<T> newm(m_width, height);
	for (int i=0; i<height; i++)
		for (int j=0; j<m_width; j++)
			newm[j][i] = (*this)[i][j];
	return newm;
}

// multiply two matrices
template <class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& mat) {
	if (m_width != mat.getHeight())
		throw Error("!exception: Matrix::operator*(), matrix dimensions don't agree");
	int newW = mat.getWidth();
	int newH = vector< vector<T> >::size();

	Matrix<T> m(newH, newW);
	for (int i=0; i<newH; i++)
		for (int j=0; j<newW; j++) {
			T val = dotProduct(this, i, &mat, j);
			m[i][j] = val;
		}
	return m;
}

// multiply a matrix by a constant
template <class T>
Matrix<T> Matrix<T>::operator*(const T val) {
	int height = vector< vector<T> >::size();
	Matrix<T> newm(height, m_width);
	for (int i=0; i<height; i++)
		for (int j=0; j<m_width; j++)
			newm[i][j] = (*this)[i][j]*val;
	return newm;
}

// add two matrices
template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& mat) {
	int height = vector< vector<T> >::size();
	if (m_width != mat.getWidth() || height != mat.getHeight()) 
		throw Error("!exception: Matrix::operator+(), matrix dimensions don't agree");
	Matrix<T> newm(height, m_width);
	for (int i=0; i<height; i++)
		for (int j=0; j<m_width; j++)
			newm[i][j] = (*this)[i][j]+mat[i][j];
	return newm;
}

// add a constant to a matrix
template <class T>
Matrix<T> Matrix<T>::operator+(const T val) {
	int height = vector< vector<T> >::size();
	Matrix<T> newm(height, m_width);
	for (int i=0; i<height; i++)
		for (int j=0; j<m_width; j++)
			newm[i][j] = (*this)[i][j]+val;
	return newm;
}

// subtract two matrices
template <class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& mat) {
	int height = vector< vector<T> >::size();
	if (m_width != mat.getWidth() || height != mat.getHeight()) 
		throw Error("!exception: Matrix::operator-(), matrix dimensions don't agree");
	Matrix<T> newm(height, m_width);
	for (int i=0; i<height; i++)
		for (int j=0; j<m_width; j++)
			newm[i][j] = (*this)[i][j] - mat[i][j];
	return newm;
}

// subtract a constant
template <class T>
Matrix<T> Matrix<T>::operator-(const T val) {
	return (*this)+(-1)*val;
}

// divide by a constant
template <class T>
Matrix<T> Matrix<T>::operator/(const T val) {
	int height = vector< vector<T> >::size();
	Matrix<T> newm(height, m_width);
	for (int i=0; i<height; i++)
		for (int j=0; j<m_width; j++)
			newm[i][j] = (*this)[i][j]/val;
	return newm;
}

// range function
template <class T>
Matrix<T> Matrix<T>::operator()(int rstart, int rend, int cstart, int cend) {
	int height = vector< vector<T> >::size();
	if (rstart >= height || rend >= height || (rstart > rend && rend > -1))
		throw Error("!exception: Matrix<T>::operator()()[range], row out of range");
	if (cstart >= m_width || cend >= m_width || (cstart > cend && cend > -1))
		throw Error("!exception: Matrix<T>::operator()()[range], column out of range");

	int newH, newW;
	if (rstart < 0)
		rstart = 0;
	if (rend < 0)
		rend = height-1;
	newH = rend-rstart+1;

	if (cstart < 0)
		cstart = 0;
	if (cend < 0)
		cend = m_width-1;
	newW = cend-cstart+1;

	Matrix<T> newm(newH, newW);
	for (int i=0; i<newH; i++)
		for (int j=0; j<newW; j++)
			newm[i][j] = (*this)[i+rstart][j+cstart];
	return newm;
}

// appending functions ==============

// append rows
template <class T>
Matrix<T> Matrix<T>::operator()(const Matrix<T>& mat, int dummy) {
	if (m_width != mat.getWidth() && m_width != 0)
		throw Error("!exception: Matrix<T>::operator()()[append row], widths don't agree");
	
	Matrix<T> newm(*this);
	int height = mat.getHeight();
	for (int i=0; i<height; i++)
		newm.push_back(mat[i]);

	return newm;
}

template <class T>
bool Matrix<T>::appendRow(const Matrix<T>& mat) {
	if (m_width != mat.getWidth() && m_width != 0)
		throw Error("!exception: Matrix<T>::appendRow(), widths don't agree");

	int height = mat.getHeight();
	for (int i=0; i<height; i++)
		push_back(mat[i]);

	if (m_width == 0) m_width = mat.getWidth();
	return true;
}

template <class T>
void Matrix<T>::push_back(const vector<T>& vec) {
	if (m_width != vec.size() && m_width != 0) {
		throw Error("!exception: Matrix<T>::push_back(), widths don't agree");
	}
	vector< vector<T> >::push_back(vec);
	if (m_width == 0) m_width = vec.size();
}

// append cols
template <class T>
Matrix<T> Matrix<T>::operator()(int dummy, const Matrix<T>& mat) {
	int height = vector< vector<T> >::size();
	if (height != mat.getHeight() && height != 0)
		throw Error("!exception: Matrix<T>::operator()()[append col], heights don't agree");

	if (height == 0) {
		return mat;
	}
	
	Matrix<T> newm = *this;
	int width = mat.getWidth();
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++)
			newm[i].push_back(mat[i][j]);
	}
	return newm;	
}

template <class T>
bool Matrix<T>::appendCol(const Matrix<T>& mat) {
	int height = vector< vector<T> >::size();
	if (height != mat.getHeight() && height != 0)
		throw Error("!exception: Matrix<T>::operator()()[append col], heights don't agree");

	if (height == 0) {
		copy(mat);
		return true;
	}

	int width = mat.getWidth();
        m_width += width;
        for (int i=0; i<height; i++)
                for (int j=0; j<width; j++)
                        (*this)[i].push_back(mat[i][j]);
			
	return true;
}

template <class T>
void Matrix<T>::setData(int h, int w, const Matrix<T>& mat) {
	int height = vector< vector<T> >::size();
	if (h >= height || w >= m_width || h < 0 || w < 0)
		throw Error("!exception: Matrix<T>::setData(), index out of range");
	if ((h+mat.getHeight()-1)>height || (w+mat.getWidth()-1)>m_width)
		throw Error("!exception: Matrix<T>::setData(), data block out of range");

	for (int i=0; i<mat.getHeight(); i++)
		for (int j=0; j<mat.getWidth(); j++)
			(*this)[h+i][w+j] = mat[i][j];
}

template <class T>
int Matrix<T>::find(int col, T val, int start) {
	if (m_width == 0)
		return -1;
	if (col < 0 || col >= m_width)
		throw Error("!exception: Matrix::find(), column out of range");
	int height = vector< vector<T> >::size();
	for (int i=start; i<height; i++)
		if ((*this)[i][col] == val) return i;
	return -1;
}

template <class T>
vector<T> Matrix<T>::getColumn(int col) {
	if (col < 0 || col >= m_width)
		throw Error("!exception: Matrix::getColumn(), column out of range");
	int height = vector< vector<T> >::size();
	vector<T> ar(height);
	for (int i=0; i<height; i++)
		ar[i] = (*this)[i][col];
	return ar;
}

template <class T>
ostream& operator<< (ostream& ostr, const Matrix<T>& m) {
	int height = m.getHeight();
	int width = m.getWidth();
	for (int i=0; i<height; i++)
	{
		for(int j=0;j<width;j++) 
			ostr << m[i][j] << " ";
		if (i<height-1) ostr << endl;
	}
	return ostr;
}

#endif
