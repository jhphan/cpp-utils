#include <cpputils/rand.h>
#include <iostream>
using namespace std;
double nrand() {
	double sum = 0;
	for (int i=0; i<12; i++)
		sum+=drand48();
	return sum-6;
}

int wrand(vector<double>& weights, double total) {
	double val = total*drand48();
	int i=-1;
	while (val > 0)
		val -= weights[++i];
	return i;
}

