
#include <Core/Thread/Thread.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Mutex.h>
#include <iostream>
#include <stdlib.h>
#include <sys/sysmp.h>
#include <unistd.h>

using SCIRun::Mutex;
using SCIRun::Runnable;
using SCIRun::Barrier;
using SCIRun::ThreadGroup;
using SCIRun::Thread;

Mutex io_lock_("io lock");

class BPS : public Runnable {
    Barrier* barrier;
    int count;
    int proc;
public:
    BPS(Barrier* barrier, int count, int proc);
    virtual void run();
};

void usage(char* progname)
{
    cerr << "usage: " << progname << " nprocessors count\n";
    exit(1);
}

int main(int argc, char* argv[])
{
    cout << "Program starting\n";
    int np=0;
    int count=0;
    if(argc != 3){
	usage(argv[1]);
    }
    np=atoi(argv[1]);
    count=atoi(argv[2]);
#if 0
    Barrier* barrier=new Barrier("test barrier");
    ThreadGroup* group=new ThreadGroup("test group");
    for(int i=0;i<np;i++){
	char buf[100];
	sprintf(buf, "worker %d", i);
	//	new Thread(new BPS(barrier, count, i), buf, group);
    }
    group->join();
#endif
    cout << "Program ending\n";
}

BPS::BPS(Barrier* barrier, int count, int proc)
    : barrier(barrier), count(count), proc(proc)
{
}

void BPS::run()
{
    int np=Thread::numProcessors();
    int p=(24+proc)%np;
#if 0
    io.lock();
    cerr << "Mustrun: " << p << "(pid=" << getpid() << ")\n";
    io.unlock();
    if(sysmp(MP_MUSTRUN, p) == -1){
	perror("sysmp - MP_MUSTRUN");
    }
#endif
    barrier->wait(np);
    //    double time=Thread::currentSeconds();
    for(int i=0;i<count;i++){
	barrier->wait(np);
	static int g=0;
	if(g != i)
	    cerr << "OOPS!\n";
	barrier->wait(np);
	if(proc==0)
	    g++;
    }
#if 0
    double etime=Thread::currentSeconds();
    if(proc==0){
	cerr << "done in " << etime-time << " seconds \n";
	cerr << count/(etime-time) << " barriers/second\n";
	cerr << (etime-time)/count*1000 << " ms/barrier\n";
    }
#endif
}
