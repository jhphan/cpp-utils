#ifndef __ERROR_H__
#define __ERROR_H__

#include <string>
#include <exception>

using namespace std;

class Error : public exception
{
private:
	string m_what;

public:
	explicit Error(const string& what) : m_what(what) {
		 m_what = what; 
	};

	virtual ~Error() throw() {};

	virtual const char * what() const throw() {
      		return m_what.c_str();
   	};

};
#endif
