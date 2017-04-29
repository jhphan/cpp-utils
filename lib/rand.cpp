#include <miblab/rand.h>
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

// pick two genes in a pseudo-random sequence such that no pair is repeated.
unsigned long long int rand_pair(vector<int> &pair, unsigned long long int ngenes, unsigned long long int pair_index){
        static unsigned long long int m=0,n=0,ng=0,a=0,c=0,seed=0,npairs=0;
        unsigned long long int x0 = 0,i=0,j=0,f=0,first=0;
        unsigned int primes[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97};

        //printf("size of unsigned long long int = %u\n", sizeof(m));
//cerr << "ng=" << ng << ", seed=" << seed << ", m=" << m << ", a=" << a << ", c=" << c << endl;
        if (ngenes != ng) { // initial setup
                // initialize linear congruential generator http://en.wikipedia.org/wiki/Linear_congruential_generator
                // will have full period for all seed values if and only if:
                // (1) 'c' and 'm' are relatively prime,
                // (2) 'a-1' is divisible by all prime factors of 'm',
                // (3) 'a-1' is a multiple of 4 if 'm' is a multiple of 4.
                npairs = ngenes*(ngenes-1)/2;        // total number of pairs
//cerr << npairs << " pairs" << endl;
                m = 8 * npairs; // * 4 to accomodate 'a' < 'm' and still divisible by all prime factors of 'm'
//cerr << "m=" << m << endl;
                // multiply factors of m to find a;
                f=m;
                a=1;    // 'a-1' divisible by all prime factors of 'm' and (possibly) by 4.
                c=1;    // relatively prime
                if ((f%4)==0){ // if 'm' is a multiple of 4, 'a-1' is multiple of 4
                        a*=2;  // pick up second *2 in next while loop.
                }
//cerr << "a=" << a << endl;
                if ((f%2)!=0) c*=2; // if m is not divisible by two, make 2 a factor of c.
//cerr << "f=" << f << endl;
                first=1;
                while ((f%2)==0){
                        if (first) { a*=2; first=0; printf("%lu,",2); } // only multiply 'a' by each factor once.
                        f = f/2;
                }
//cerr << "f=" << f << endl;
                i=3; j=1; first=1;
		unsigned int prime=1;
                while (i <= (sqrt(f)+1)) {
                        if (first && (f%i)==0){
                                a*=i; // multiply 'a' by a factor of 'm'
                                f=f/i;
                                first = 0;
                        } else if (first && (i==primes[j]) && prime==1) {
				prime = i;
                        } else if ((f%i)==0){
                                f=f/i;
                        } else {
                                i = i + 2;
                                first = 1;
				if (primes[j] < i) j=(j+1)%25; // make sure we don't go past end of primes array
                        }
//cerr << "i=" << i << ", a=" << a << ", f=" << f << ", c=" << c << ", prime = " << prime << endl;
                }
                if (f>1) a*=f;
                a=a+1;

		// make 'c' largest multiple of the smallest prime non-factor of 'm' that is less than 'm'
		c=1; while(c < (m/prime)) c*=prime;
//cerr << "c = " << c << endl;

                ng = ngenes;
                printf("a=%llu,c=%llu,m=%llu\n",a,c,m);
                n=0;
		seed=x0;
        }
//cerr << "pair_index=" << pair_index << endl;
        // generate the 'pair_index'th pair in the sequence
        if (pair_index<n){ // start over from initial seed.
                n=0;
                seed=x0;
        }
//cerr << "n=" << n << ", seed = " << seed << endl;
        while (n < pair_index){
                while((seed = (a*seed + c) %m) >= npairs); // because we're actually cycling through 8*npairs possible seeds, just use the ones < npairs
                n++;
        }
//cerr << "n=" << n << ", seed = " << seed << endl;
        f=seed;
        i = 1;
        j = 0;
        while(f>ngenes-2-j) {
                f-=ngenes-1-j;
                j++;
                i++;
        }
        i+=f;
        pair[0]=j;
        pair[1]=i;
        return seed;

}

