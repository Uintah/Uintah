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


#include <Core/Thread/Thread.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <iostream>
using std::cerr;
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>



using namespace SCIRun;

Mutex io("io lock");

class BPS : public Runnable {
    Barrier* barrier;
    int count;
    int proc;
    int np;
public:
    BPS(Barrier* barrier, int count, int proc, int np);
    virtual void run();
};

void usage(char* progname)
{
    cerr << "usage: " << progname << " nprocessors count\n";
    exit(1);
}

int main(int argc, char* argv[])
{
    int np=0;
    int count=0;
    if(argc != 3){
	usage(argv[0]);
    }
    np=atoi(argv[1]);
    count=atoi(argv[2]);
    Barrier* barrier=new Barrier("test barrier");
    ThreadGroup* group=new ThreadGroup("test group");
    for(int i=0;i<np;i++){
	char buf[100];
	sprintf(buf, "worker %d", i);
	new Thread(new BPS(barrier, count, i, np), strdup(buf), group);
    }
    group->join();
}

BPS::BPS(Barrier* barrier, int count, int proc, int np)
    : barrier(barrier), count(count), proc(proc), np(np)
{
}

void BPS::run()
{
    barrier->wait(np);
    double time=Time::currentSeconds();
    for(int i=0;i<count;i++){
	barrier->wait(np);
	static int g=0;
	if(g != i)
	    cerr << "OOPS!: " << g << " vs. " << i << ", proc=" << proc << "\n";
	barrier->wait(np);
	if(proc==0)
	    g++;
    }
    double etime=Time::currentSeconds();
    if(proc==0){
	cerr << "done in " << etime-time << " seconds \n";
	cerr << count/(etime-time) << " barriers/second\n";
	cerr << (etime-time)/count*1000 << " ms/barrier\n";
    }
}

