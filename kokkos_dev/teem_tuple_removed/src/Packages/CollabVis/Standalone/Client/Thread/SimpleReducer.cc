/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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
    array_size_=-1;
    p_=0;
    join_[0]=0;
    join_[1]=0;
}

SimpleReducer::~SimpleReducer()
{
    if(p_){
	delete[] join_[0];
	delete[] join_[1];
	delete[] p_;
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
	if(p_){
	    delete[] join_[0];
	    delete[] join_[1];
	    delete[] p_;
	}
        join_[0]=new joinArray[n];
        join_[1]=new joinArray[n];
        p_=new pdata[n];
        for(int i=0;i<n;i++)
	    p_[i].buf_=0;
        array_size_=n;
    }
    wait(n);
}

double
SimpleReducer::sum(int proc, int n, double mysum)
{
    if(n != array_size_){
        collectiveResize(proc, n);
    }

    int buf=p_[proc].buf_;
    p_[proc].buf_=1-buf;

    joinArray* j=join_[buf];
    j[proc].d_.d_=mysum;
    wait(n);
    double sum=0;
    for(int i=0;i<n;i++)
        sum+=j[i].d_.d_;
    return sum;
}

double
SimpleReducer::max(int proc, int n, double mymax)
{
    if(n != array_size_){
        collectiveResize(proc, n);
    }

    int buf=p_[proc].buf_;
    p_[proc].buf_=1-buf;

    joinArray* j=join_[buf];
    j[proc].d_.d_=mymax;
    Barrier::wait(n);
    double gmax=j[0].d_.d_;
    for(int i=1;i<n;i++)
        if(j[i].d_.d_ > gmax)
	    gmax=j[i].d_.d_;
    return gmax;
}

double
SimpleReducer::min(int proc, int n, double mymin)
{
    if(n != array_size_){
        collectiveResize(proc, n);
    }

    int buf=p_[proc].buf_;
    p_[proc].buf_=1-buf;

    joinArray* j=join_[buf];
    j[proc].d_.d_=mymin;
    Barrier::wait(n);
    double gmin=j[0].d_.d_;
    for(int i=1;i<n;i++)
        if(j[i].d_.d_ < gmin)
	    gmin=j[i].d_.d_;
    return gmin;
}


} // End namespace SCIRun
