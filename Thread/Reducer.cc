
#include "Reducer.h"
#include "ThreadGroup.h"

/**
 * Perform reduction operations over a set of threads.  Reduction
 * operations include things like global sums, global min/max, etc.
 * In these operations, a local sum (operation) is performed on each
 * thread, and these sums are added together.
 */

void Reducer::collective_resize(int proc) {
    // Extra barrier here to change the array size...

    // We must wait until everybody has seen the array size change,
    // or they will skip down too soon...
    wait();
    if(proc==0){
        int n=threadGroup?threadGroup->nactive(true):nthreads;
        delete[] join[0];
        delete[] join[1];
        join[0]=new join_array[n];
        join[0]=new join_array[n];
        delete[] p;
        p=new pdata[n];
        for(int i=0;i<n;i++)
    	p[i].buf=0;
        arraysize=n;
    }
    wait();
}

Reducer::Reducer(const char* name, int nthreads) : Barrier(name, nthreads) {
    arraysize=nthreads;
    join[0]=new join_array[nthreads];
    join[1]=new join_array[nthreads];
    p=new pdata[nthreads];
    for(int i=0;i<nthreads;i++)
        p[i].buf=0;
}

Reducer::Reducer(const char* name, ThreadGroup* group) : Barrier(name, nthreads) {
    arraysize=group->nactive(true);
    join[0]=new join_array[arraysize];
    join[1]=new join_array[arraysize];
    p=new pdata[arraysize];
    for(int i=0;i<arraysize;i++)
        p[i].buf=0;
}

Reducer::~Reducer() {
    delete[] join[0];
    delete[] join[1];
    delete[] p;
}

double Reducer::sum(int proc, double mysum) {
    int n=threadGroup?threadGroup->nactive(true):nthreads;
    if(n != arraysize){
        collective_resize(proc);
    }

    int buf=p[proc].buf;
    p[proc].buf=1-buf;

    join_array* j=join[buf];
    j[proc].d.d=mysum;
    wait();
    double sum=0;
    for(int i=0;i<n;i++)
        sum+=j[i].d.d;
    return sum;
}

double Reducer::max(int proc, double mymax) {
    int n=threadGroup?threadGroup->nactive(true):nthreads;
    if(n != arraysize){
        collective_resize(proc);
    }

    int buf=p[proc].buf;
    p[proc].buf=1-buf;

    join_array* j=join[buf];
    j[proc].d.d=mymax;
    Barrier::wait();
    double gmax=j[0].d.d;
    for(int i=1;i<n;i++)
        if(j[i].d.d > gmax)
    	gmax=j[i].d.d;
    return gmax;
}

