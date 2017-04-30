#ifndef __COMMANDLINE_H__
#define __COMMANDLINE_H__

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <cpputils/stringtokenizer.h>

using namespace std;

class CommandLine {

private:
	int m_argc;
	char** m_argv;

	vector<string> str_argv;
	void initializeStrings();

public:
	
	CommandLine();
	CommandLine(int argc, char* argv[]);

	~CommandLine();

	void setArgs(int argc, char* argv[]);

	bool getArg(int& val, string str);
	bool getArg(float& val, string str);
	bool getArg(double& val, string str);
	bool getArg(string& val, string str);
	bool getArg(string str);
};

#endif
