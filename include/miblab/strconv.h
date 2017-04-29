#ifndef __STRCONV_H__
#define __STRCONV_H__

using namespace std;

#include <string>
#include <sstream>
#include <string.h>
#include <stdio.h>

string conv2str(int val);
string conv2str(long val);
string conv2str(float val);
string conv2str(double val);
string conv2str(unsigned long val);

#endif
