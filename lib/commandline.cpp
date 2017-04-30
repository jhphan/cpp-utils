#include <cpputils/commandline.h>

void CommandLine::initializeStrings() {
	str_argv = vector<string> (m_argc);
	for (int i=0; i<m_argc; i++)
		str_argv[i] = string(m_argv[i]);
}

void CommandLine::setArgs(int argc, char** argv) {
	m_argc = argc;
	m_argv = argv;
	initializeStrings();
}

CommandLine::CommandLine() {
}

CommandLine::CommandLine(int argc, char** argv) {
	m_argc = argc;
	m_argv = argv;
	initializeStrings();
}

CommandLine::~CommandLine() {
}

bool CommandLine::getArg(int& val, string str) {
	string str_val;
	bool result = getArg(str_val, str);
	val = atoi(str_val.c_str());
	return result;
}

bool CommandLine::getArg(float& val, string str) {
	double dval;
	bool result = getArg(dval, str);
	val = (float)dval;
	return result;
}

bool CommandLine::getArg(double& val, string str) {
	string str_val;
	bool result = getArg(str_val, str);
	val = atof(str_val.c_str());
	return result;
}

bool CommandLine::getArg(string& val, string str) {
	// find the str
	vector<string>::iterator itr = find(str_argv.begin(),str_argv.end(),string("-")+str);

	if (itr < str_argv.end()) {
		int index = itr - str_argv.begin();
		if (index<m_argc-1)
			val = str_argv[index+1];
		return true;
	}
	return false;
}

bool CommandLine::getArg(string str) {
	string str_val;
	bool result = getArg(str_val, str);
	return result;
}
