#include <miblab/strconv.h>

string conv2str(int val) {
	char ch[64];
	sprintf(ch, "%d\0", val);
	return string(ch);
}

string conv2str(long val) {
	char ch[64];
	sprintf(ch, "%ld\0", val);
	return string(ch);
}

string conv2str(float val) {
	ostringstream os;
	os << val;
	return os.str();
	//char ch[64];
	//sprintf(ch, "%f\0", val);
	//return string(ch);
}

string conv2str(double val) {
	char ch[64];
	sprintf(ch, "%f\0", val);
	return string(ch);
}

string conv2str(unsigned long val) {
	char ch[64];
	sprintf(ch,"%u\0",val);
	return string(ch);
}

