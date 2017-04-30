#include <cpputils/uniqid.h>

string uniqid(int num) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	int sec = (int) tv.tv_sec;
	int usec = (int) (tv.tv_usec % 0x100000);
	
	char* tempstr = new char[33];
	char* str = new char[33];

	sprintf(tempstr, "%05d%08x%05x\0", num, sec, usec);

	// compute md5 sum of uniqid
	md5str(str, tempstr);
	string str_id = string(str);
	delete[] str;
	delete[] tempstr;

	return str_id;
}
