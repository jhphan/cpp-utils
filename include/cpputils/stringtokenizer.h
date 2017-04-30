#ifndef __STRINGTOKENIZER_H__
#define __STRINGTOKENIZER_H__

#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include <stdlib.h>

using namespace std;

class StringTokenizer {

private:
	int m_index;
	bool m_greedy;
	string main_str;
	int str_len;
protected:
	char m_delim;

	bool getNextTokenUngreedy(string &str);
	bool inDelims(char chr);
	
public:

	StringTokenizer(char* str, char delim = ' ', bool greedy = true);
	StringTokenizer(const string& str, char delim = ' ', bool greedy = true);

	~StringTokenizer();
	
	bool getNextToken(string& str);
	bool getNextToken(int& val);
	bool getNextToken(float& val);
	bool getNextToken(double& val);
	void resetIndex();

	vector<string> split();
	vector<int> split_int();
	vector<float> split_float();
	vector<double> split_double();
};

#endif
