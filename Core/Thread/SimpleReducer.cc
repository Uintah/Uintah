
/*
 *  SimpleReducer: A barrier with reduction operations
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Thread/SimpleReducer.h>
#include <Core/Thread/ThreadGroup.h>
namespace SCIRun {


SimpleReducer::SimpleReducer(const char* name)
    : Barrier(name)
{
    d_array_size=-1;
    d_p=0;
    d_join[0]=0;
    d_join[1]=0;
}

SimpleReducer::~SimpleReducer()
{
    if(d_p){
	delete[] d_join[0];
	delete[] d_join[1];
	delete[] d_p;
    }
}

void
SimpleReducer::collectiveResize(int proc, int n)
{
    // Extra barrier here to change the array size...

    // We must wait until everybody has seen the array size change,
    // or they will skip down too soon...
    wait(n);
    if(proc==0){
	if(d_p){
	    delete[] d_join[0];
	    delete[] d_join[1];
	    delete[] d_p;
	}
        d_join[0]=new joinArray[n];
        d_join[1]=new joinArray[n];
        d_p=new pdata[n];
        for(int i=0;i<n;i++)
	    d_p[i].d_buf=0;
        d_array_size=n;
    }
    wait(n);
}

double
SimpleReducer::sum(int proc, int n, double mysum)
{
    if(n != d_array_size){
        collectiveResize(proc, n);
    }

    int buf=d_p[proc].d_buf;
    d_p[proc].d_buf=1-buf;

    joinArray* j=d_join[buf];
    j[proc].d_d.d_d=mysum;
    wait(n);
    double sum=0;
    for(int i=0;i<n;i++)
        sum+=j[i].d_d.d_d;
    return sum;
}

double
SimpleReducer::max(int proc, int n, double mymax)
{
    if(n != d_array_size){
        collectiveResize(proc, n);
    }

    int buf=d_p[proc].d_buf;
    d_p[proc].d_buf=1-buf;

    joinArray* j=d_join[buf];
    j[proc].d_d.d_d=mymax;
    Barrier::wait(n);
    double gmax=j[0].d_d.d_d;
    for(int i=1;i<n;i++)
        if(j[i].d_d.d_d > gmax)
	    gmax=j[i].d_d.d_d;
    return gmax;
}

double
SimpleReducer::min(int proc, int n, double mymin)
{
    if(n != d_array_size){
        collectiveResize(proc, n);
    }

    int buf=d_p[proc].d_buf;
    d_p[proc].d_buf=1-buf;

    joinArray* j=d_join[buf];
    j[proc].d_d.d_d=mymin;
    Barrier::wait(n);
    double gmin=j[0].d_d.d_d;
    for(int i=1;i<n;i++)
        if(j[i].d_d.d_d < gmin)
	    gmin=j[i].d_d.d_d;
    return gmin;
}


} // End namespace SCIRun
