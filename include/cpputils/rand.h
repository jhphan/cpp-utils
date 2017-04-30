#ifndef __RAND_H__
#define __RAND_H__

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>

using namespace std;

double nrand();
int wrand(vector<double>& weights, double total);

#endif
