
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

