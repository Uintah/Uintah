
/* REFERENCED */
static char *id="$Id$";

/*
 *  Reducer: A barrier with reduction operations
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <SCICore/Thread/Reducer.h>
#include <SCICore/Thread/ThreadGroup.h>

SCICore::Thread::Reducer::Reducer(const char* name, int numThreads)
    : Barrier(name, numThreads)
{
    d_array_size=numThreads;
    d_join[0]=new joinArray[numThreads];
    d_join[1]=new joinArray[numThreads];
    d_p=new pdata[d_num_threads];
    for(int i=0;i<d_num_threads;i++)
        d_p[i].d_buf=0;
}

SCICore::Thread::Reducer::Reducer(const char* name, ThreadGroup* group)
    : Barrier(name, group)
{
    d_array_size=group->numActive(true);
    d_join[0]=new joinArray[d_array_size];
    d_join[1]=new joinArray[d_array_size];
    d_p=new pdata[d_array_size];
    for(int i=0;i<d_array_size;i++)
        d_p[i].d_buf=0;
}

SCICore::Thread::Reducer::~Reducer()
{
    delete[] d_join[0];
    delete[] d_join[1];
    delete[] d_p;
}

void
SCICore::Thread::Reducer::collectiveResize(int proc)
{
    // Extra barrier here to change the array size...

    // We must wait until everybody has seen the array size change,
    // or they will skip down too soon...
    wait();
    if(proc==0){
        int n=d_thread_group?d_thread_group->numActive(true):d_num_threads;
        delete[] d_join[0];
        delete[] d_join[1];
        d_join[0]=new joinArray[n];
        d_join[0]=new joinArray[n];
        delete[] d_p;
        d_p=new pdata[n];
        for(int i=0;i<n;i++)
	    d_p[i].d_buf=0;
        d_array_size=n;
    }
    wait();
}

double
SCICore::Thread::Reducer::sum(int proc, double mysum)
{
    int n=d_thread_group?d_thread_group->numActive(true):d_num_threads;
    if(n != d_array_size){
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

double
SCICore::Thread::Reducer::max(int proc, double mymax)
{
    int n=d_thread_group?d_thread_group->numActive(true):d_num_threads;
    if(n != d_array_size){
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

//
// $Log$
// Revision 1.4  1999/08/25 19:00:50  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:37:59  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
