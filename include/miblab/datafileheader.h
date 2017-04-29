#ifndef __DATAFILEHEADER__H__
#define __DATAFILEHEADER__H__

#include <miblab/datafile.h>

using namespace std;

template <class T>
class DataFileHeader : public DataFile<T> {

private:
	vector<T>* readLineArray(FILE* file, string& headerval, bool header);
	vector<string> m_colheaders;
	vector<string> m_rowheaders;
	void readHeader(FILE* file);	// populate column headers

public:
	DataFileHeader(string filename, char delim = 0, int rows = 0);
	~DataFileHeader();

	vector<string>& getColHeaders() { return m_colheaders; };
	vector<string>& getRowHeaders() { return m_rowheaders; };
	void setColHeaders(vector<string>& colheaders) { m_colheaders = colheaders; };
	void setRowHeaders(vector<string>& rowheaders) { m_rowheaders = rowheaders; };

	void readDataFile(Matrix<T>& data, bool colHeader, bool rowHeader);
	void readDataFileArray(vector< vector<T> >& data, bool colHeader, bool rowHeader);
	void writeDataFile(const Matrix<T>& mat, bool colHeader, bool rowHeader, ofstream::openmode mode = ofstream::out|ofstream::trunc); // write matrix (of same type, T)
};

template <class T>
void DataFileHeader<T>::readHeader(FILE* file) {
	char line[MAX_BUF_LEN];
	DataFile<T>::readFileLine(file, line);
	StringTokenizer st(string(line), DataFile<T>::m_delim);
	m_colheaders = st.split();
}

template <class T>
vector<T>* DataFileHeader<T>::readLineArray(FILE* file, string& headerval, bool header) {
	vector<T>* arline = new vector<T>();
	char line[MAX_BUF_LEN];
	DataFile<T>::readFileLine(file, line);
	// skip all lines beginning with a #
	while (line[0] == '#')
		DataFile<T>::readFileLine(file, line);

	StringTokenizer stoken(line, DataFile<T>::m_delim);
	if (header) stoken.getNextToken(headerval);
	T val;
	while (stoken.getNextToken(val)) {
		arline->push_back(val);
	}
	return arline;
}

template <class T>
void DataFileHeader<T>::readDataFile(Matrix<T>& data, bool colHeader, bool rowHeader) {
	FILE* file;
	if ((file = fopen(DataFile<T>::m_filename.c_str(), "r")) == NULL)
		throw Error("!exception: DataFile<T>::readDataFile(), can't open file");

	// empty the data matrix
	data.clear();

	if (colHeader) readHeader(file);
	m_rowheaders = vector<string>();

	// read the first line	
	string rowheaderval;
	vector<T>* arline = readLineArray(file, rowheaderval, rowHeader);
	int count = 0;
	while (arline->size() > 0 && ((count < DataFile<T>::m_rows && DataFile<T>::m_rows > 0) || (DataFile<T>::m_rows == 0))) {
		if (DataFile<T>::m_rows > 0) count++;
		data.push_back(*arline);
		if (rowHeader) m_rowheaders.push_back(rowheaderval);
		delete arline;
		arline = readLineArray(file, rowheaderval, rowHeader);
	}
	delete arline;
	
	fclose(file);
}

template <class T>
void DataFileHeader<T>::readDataFileArray(vector< vector<T> >& data, bool colHeader, bool rowHeader) {
	FILE* file;
	if ((file = fopen(DataFile<T>::m_filename.c_str(), "r")) == NULL)
		throw Error("!exception: DataFile<T>::readDataFileArray(), can't open file");

	data = vector< vector<T> >();
	
	if (colHeader) readHeader(file);
	m_rowheaders = vector<string>();

	// read the first line
	string rowheaderval;	
	vector<T>* arline = readLineArray(file, rowheaderval, rowHeader);
	int count = 0;
	while (arline->size() > 0 && ((count < DataFile<T>::m_rows && DataFile<T>::m_rows > 0) || (DataFile<T>::m_rows == 0))) {
		if (DataFile<T>::m_rows > 0) count++;
		data.push_back(*arline);
		if (rowHeader) m_rowheaders.push_back(rowheaderval);
		delete arline;
		arline = readLineArray(file);
	}
	delete arline;
	
	fclose(file);
}


template <class T>
DataFileHeader<T>::DataFileHeader(string filename, char delim, int rows) : DataFile<T>(filename, delim, rows) {
}

template <class T>
DataFileHeader<T>::~DataFileHeader() {
}

template <class T>
void DataFileHeader<T>::writeDataFile(const Matrix<T>& mat, bool colHeader, bool rowHeader, ofstream::openmode mode) {
	fstream mFile(DataFile<T>::m_filename.c_str(), mode);
	if (colHeader) {
		if (m_colheaders.size() != mat.getWidth())
			throw Error("!exception: DataFileHeader<T>::writeDataFile(): column headers and data size do not match");
	}
	if (rowHeader) {
		if (m_rowheaders.size() != mat.getHeight())
			throw Error("!exception: DataFileHeader<T>::writeDataFile(): row headers and data size do not match");
	}
	if (colHeader) {
		if (rowHeader) mFile << DataFile<T>::m_delim;
		for (int i=0; i<m_colheaders.size()-1; i++)
			mFile << m_colheaders[i] << DataFile<T>::m_delim;
		if (m_colheaders.size() > 0)
			mFile << m_colheaders[m_colheaders.size()-1];
		mFile << endl;
	}

	for (int i=0; i<mat.getHeight(); i++) {
		if (rowHeader) {
			mFile << m_rowheaders[i];
			if (mat.getWidth() > 0) mFile << DataFile<T>::m_delim;
		}
		for (int j=0; j<mat.getWidth()-1; j++)
			mFile << mat[i][j] << DataFile<T>::m_delim;
		if (mat.getWidth() > 0)
			mFile << mat[i][mat.getWidth()-1];
		mFile << endl;
	}
	mFile.close();
}

#endif
