#ifndef __RAND_H__
#define __RAND_H__

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>

using namespace std;

double nrand();
int wrand(vector<double>& weights, double total);
unsigned long long int rand_pair(vector<int> &pair, unsigned long long int ngenes, unsigned long long int pair_index);
             

#endif
