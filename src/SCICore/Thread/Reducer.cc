
#include "Reducer.h"
#include "ThreadGroup.h"

/**
 * Perform reduction operations over a set of threads.  Reduction
 * operations include things like global sums, global min/max, etc.
 * In these operations, a local sum (operation) is performed on each
 * thread, and these sums are added together.
 */

void Reducer::collectiveResize(int proc)
{
    // Extra barrier here to change the array size...

    // We must wait until everybody has seen the array size change,
    // or they will skip down too soon...
    wait();
    if(proc==0){
        int n=d_threadGroup?d_threadGroup->numActive(true):d_numThreads;
        delete[] d_join[0];
        delete[] d_join[1];
        d_join[0]=new joinArray[n];
        d_join[0]=new joinArray[n];
        delete[] d_p;
        d_p=new pdata[n];
        for(int i=0;i<n;i++)
	    d_p[i].d_buf=0;
        d_arraySize=n;
    }
    wait();
}

Reducer::Reducer(const std::string& name, int numThreads)
    : Barrier(name, numThreads)
{
    d_arraySize=numThreads;
    d_join[0]=new joinArray[numThreads];
    d_join[1]=new joinArray[numThreads];
    d_p=new pdata[d_numThreads];
    for(int i=0;i<d_numThreads;i++)
        d_p[i].d_buf=0;
}

Reducer::Reducer(const std::string& name, ThreadGroup* group)
    : Barrier(name, group)
{
    d_arraySize=group->numActive(true);
    d_join[0]=new joinArray[d_arraySize];
    d_join[1]=new joinArray[d_arraySize];
    d_p=new pdata[d_arraySize];
    for(int i=0;i<d_arraySize;i++)
        d_p[i].d_buf=0;
}

Reducer::~Reducer()
{
    delete[] d_join[0];
    delete[] d_join[1];
    delete[] d_p;
}

double Reducer::sum(int proc, double mysum)
{
    int n=d_threadGroup?d_threadGroup->numActive(true):d_numThreads;
    if(n != d_arraySize){
        collectiveResize(proc);
    }

    int buf=d_p[proc].d_buf;
    d_p[proc].d_buf=1-buf;

    joinArray* j=d_join[buf];
    j[proc].d_d.d_d=mysum;
    wait();
    double sum=0;
    for(int i=0;i<n;i++)
        sum+=j[i].d_d.d_d;
    return sum;
}

double Reducer::max(int proc, double mymax)
{
    int n=d_threadGroup?d_threadGroup->numActive(true):d_numThreads;
    if(n != d_arraySize){
        collectiveResize(proc);
    }

    int buf=d_p[proc].d_buf;
    d_p[proc].d_buf=1-buf;

    joinArray* j=d_join[buf];
    j[proc].d_d.d_d=mymax;
    Barrier::wait();
    double gmax=j[0].d_d.d_d;
    for(int i=1;i<n;i++)
        if(j[i].d_d.d_d > gmax)
	    gmax=j[i].d_d.d_d;
    return gmax;
}

