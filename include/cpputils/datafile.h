#ifndef __DATAFILE__H__
#define __DATAFILE__H__

#define MAX_BUF_LEN	1048576

#include <iostream>
#include <fstream>
#include <cpputils/matrix.h>
#include <cpputils/stringtokenizer.h>
#include <cpputils/error.h>

using namespace std;

template <class T>
class DataFile {

protected:
	string m_filename;
	int m_rows;
	char m_delim;


	vector<T>* readLineArray(FILE* file);


public:
	int readFileLine(FILE* f, char* line);
	DataFile(string filename, char delim = 0, int rows = 0);
	~DataFile();

	void setFileName(string filename) { m_filename = filename; };
	void setDelim(char delim) { m_delim = delim; };
	void setRows(int rows) { m_rows = rows; };

	string& getFileName() { return m_filename; };
	char getDelim() { return m_delim; };
	int getRows() { return m_rows; };
	bool fileExists();

	void readDataFile(Matrix<T>& data);
	void readDataFileArray(vector< vector<T> >& data);
	void writeDataFile(const Matrix<T>& mat, ofstream::openmode mode = ofstream::out|ofstream::trunc); // write matrix (of same type, T)

	void readDataFileLines(vector<string>& data);	// read file into an array of strings
};

template <class T>
int DataFile<T>::readFileLine(FILE* f, char* line) {
	int size;
	char ch;
	size = 0;

	ch = fgetc(f);
	while ((ch != '\n') && (ch != EOF) && (size<MAX_BUF_LEN-1)) {
		line[size] = ch;
		ch = fgetc(f);
		size++;
	}
	line[size] = '\0';
	if (ch == EOF) return 0;
	return size+1;
}

template <class T>
vector<T>* DataFile<T>::readLineArray(FILE* file) {
	vector<T>* arline = new vector<T>();
	char* line = new char[MAX_BUF_LEN];

	readFileLine(file, line);
	// skip all lines beginning with a #
	while (line[0] == '#')
		readFileLine(file, line);

	StringTokenizer stoken(line, m_delim,true);
	delete[] line;
	T val;
	while (stoken.getNextToken(val)) {
		arline->push_back(val);
	}
	return arline;
}

template <class T>
void DataFile<T>::readDataFile(Matrix<T>& data) {
	FILE* file;
	if ((file = fopen(m_filename.c_str(), "r")) == NULL)
		throw Error("!exception: DataFile<T>::readDataFile(), can't open file");

	// empty the data matrix
	data = Matrix<T>();

	// read the first line	
	vector<T>* arline = readLineArray(file);
	int count = 0;
	while (arline->size() > 0 && ((count < m_rows && m_rows > 0) || (m_rows == 0))) {
		if (m_rows > 0) count++;
		data.push_back(*arline);
		delete arline;
		arline = readLineArray(file);
	}
	delete arline;
	
	fclose(file);
}

template <class T>
void DataFile<T>::readDataFileArray(vector< vector<T> >& data) {
	FILE* file;
	if ((file = fopen(m_filename.c_str(), "r")) == NULL)
		throw Error("!exception: DataFile<T>::readDataFileArray(), can't open file");

	data = vector< vector<T> >();

	// read the first line	
	vector<T>* arline = readLineArray(file);
	int count = 0;
	while (arline->size() > 0 && ((count < m_rows && m_rows > 0) || (m_rows == 0))) {
		if (m_rows > 0) count++;
		data.push_back(*arline);
		delete arline;
		arline = readLineArray(file);
	}
	delete arline;
	
	fclose(file);
}


template <class T>
DataFile<T>::DataFile(string filename, char delim, int rows) {
	m_filename = filename;
	m_rows = rows;
	m_delim = delim;
}

template <class T>
DataFile<T>::~DataFile() {

}

template <class T>
bool DataFile<T>::fileExists() {
	FILE* file = fopen(m_filename.c_str(), "r");
	if (file == NULL)
		return false;
	fclose(file);
	return true;
}

template <class T>
void DataFile<T>::writeDataFile(const Matrix<T>& mat, ofstream::openmode mode) {
	fstream mFile(m_filename.c_str(), mode);
	for (int i=0; i<mat.getHeight(); i++) {
		for (int j=0; j<mat.getWidth()-1; j++)
			mFile << mat[i][j] << m_delim;
		if (mat.getWidth() > 0)
			mFile << mat[i][mat.getWidth()-1];
		mFile << endl;
	}
	mFile.close();
}

template <class T>
void DataFile<T>::readDataFileLines(vector<string>& data) {
	FILE* file;
	if ((file = fopen(m_filename.c_str(), "r")) == NULL)
		throw Error("!exception: DataFile<T>::readDataFileLines(), can't open file");
	data.clear();

	char buf[MAX_BUF_LEN];

	while (readFileLine(file, buf)) {
		data.push_back(string(buf));
	}

	fclose(file);
}

#endif
