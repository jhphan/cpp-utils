#include <cpputils/stringtokenizer.h>

StringTokenizer::StringTokenizer(char *str, char delim, bool greedy) {
	m_delim = delim;
	m_index = 0;
	m_greedy = greedy;
	main_str = str;
	str_len = main_str.length();
}

StringTokenizer::StringTokenizer(const string& str, char delim, bool greedy) {
	m_delim = delim;
	m_index = 0;
	m_greedy = greedy;
	main_str = str;
	str_len = main_str.length();
}

StringTokenizer::~StringTokenizer() {

}

bool StringTokenizer::inDelims(char chr) {
	if (m_delim == 0)
		// all whitespace characters
		return (chr == ' ' || chr == '\n' || chr == '\t' || chr == '\r');
	else
		return (chr == m_delim);
}

bool StringTokenizer::getNextTokenUngreedy(string& str) {
	// traverse the string until a delim is found or the end of is reached
	
	char temp[str_len+1];
	strncpy(temp, main_str.c_str(), str_len);
	temp[str_len] = 0; //null terminate
	
	int start_index = m_index;
	
	while (!inDelims(temp[m_index]) && m_index < str_len)
		m_index++;
	str = main_str.substr(start_index, m_index - start_index);
	m_index++;
	return true;
}

bool StringTokenizer::getNextToken(string& str) {

	const char* temp = main_str.c_str();

	if (!m_greedy) return getNextTokenUngreedy(str);
	
	// start from m_index and traverse the string until the end is reached
	// 	or a non-delim character is reached
	while (inDelims(temp[m_index]) && m_index < str_len )
		m_index++;
	if (m_index >= str_len) return false;

	// non-delim character found at m_index
	int start_index = m_index;
	m_index++;
	// continue traversing the string until a delim character is found
	//	or until the end is reached
	while (!inDelims(temp[m_index]) && m_index < str_len)
		m_index++;
	// return substr
	str = main_str.substr(start_index, m_index - start_index);

	return true;
}

bool StringTokenizer::getNextToken(int& val) {
	string str = "";
	if (!getNextToken(str))
		return false;
	val = atoi(str.c_str());
	return true;
}

bool StringTokenizer::getNextToken(float& val) {
	string str = "";
	if (!getNextToken(str))
		return false;
	val = atof(str.c_str());
	return true;
}

bool StringTokenizer::getNextToken(double& val) {
	string str = "";
	if (!getNextToken(str))
		return false;
	val = atof(str.c_str());
	return true;
}

void StringTokenizer::resetIndex() {
	m_index = 0;
}

// split the string at delimiter
vector<string> StringTokenizer::split() {
	vector<string> ar_str;
	string str;
	while (getNextToken(str)) {
		ar_str.push_back(str);
	}
	return ar_str;
}

vector<int> StringTokenizer::split_int() {
	vector<int> ar_int;
	int val;
	while (getNextToken(val)) {
		ar_int.push_back(val);
	}
	return ar_int;
}

vector<float> StringTokenizer::split_float() {
	vector<float> ar_float;
	float val;
	while (getNextToken(val)) {
		ar_float.push_back(val);
	}
	return ar_float;
}

vector<double> StringTokenizer::split_double() {
	vector<double> ar_double;
	double val;
	while (getNextToken(val)) {
		ar_double.push_back(val);
	}
	return ar_double;
}

