
/*
 *  SimpleReducer: A barrier with reduction operations
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <SCICore/Thread/SimpleReducer.h>
#include <SCICore/Thread/ThreadGroup.h>

using SCICore::Thread::SimpleReducer;

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

//
// $Log$
// Revision 1.5  2000/02/15 00:23:49  sparker
// Added:
//  - new Thread::parallel method using member template syntax
//  - Parallel2 and Parallel3 helper classes for above
//  - min() reduction to SimpleReducer
//  - ThreadPool class to help manage a set of threads
//  - unmap page0 so that programs will crash when they deref 0x0.  This
//    breaks OpenGL programs, so any OpenGL program linked with this
//    library must call Thread::allow_sgi_OpenGL_page0_sillyness()
//    before calling any glX functions.
//  - Do not trap signals if running within CVD (if DEBUGGER_SHELL env var set)
//  - Added "volatile" to fetchop barrier implementation to workaround
//    SGI optimizer bug
//
// Revision 1.4  1999/09/21 23:19:06  dmw
// fixed a bug Steve missed... ;)
//
// Revision 1.3  1999/09/21 18:37:20  sparker
// Fixed memory allocation bug
//
// Revision 1.2  1999/08/29 00:47:01  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.1  1999/08/28 03:46:50  sparker
// Final updates before integration with PSE
//
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
