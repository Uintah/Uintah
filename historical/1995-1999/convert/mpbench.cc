
#include <Classlib/Timer.h>
#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <iostream.h>
#include <stdlib.h>
#include <string.h>

static int np;
static int r;
static Barrier* barrier;
static Mutex* mutex;
static Semaphore* semaphore(0);

Semaphore* done;

void do_barrier(void*, int)
{
    int rep=r;
    for(int i=0;i<rep;i++)
	barrier->wait(np);
}

void do_mutex(void*, int)
{
    int rep=r;
    for(int i=0;i<rep;i++){
	mutex->lock();
	mutex->unlock();
    }
}

void do_semaphore(void*, int proc)
{
    int rep=r;
    if(proc&1){
	for(int i=0;i<rep;i++)
	    semaphore->down();
    } else {
	for(int i=0;i<rep;i++)
	    semaphore->up();
    }
}

void usage(char* progname)
{
    cerr << progname << " barrier|mutex|semaphore r [nprocessors]\n";
    exit(-1);
}

int main(int argc, char** argv)
{
    Task::initialize(argv[0]);
    if(argc<2 || argc >4){
	usage(argv[0]);
    }
    r=atoi(argv[2]);
    if(argc==4)
	np=atoi(argv[3]);
    WallClockTimer timer;
    void (*fn)(void*, int);
    if(!strcmp(argv[1], "barrier")){
	barrier=new Barrier;
	fn=do_barrier;
    } else if(!strcmp(argv[1], "mutex")){
	mutex=new Mutex;
	fn=do_mutex;
    } else if(!strcmp(argv[1], "semaphire")){
	semaphore=new Semaphore(0);
	fn=do_semaphore;
    }
    timer.start();
    Task::multiprocess(np, fn, 0);
    timer.stop();
    cerr << np*r << " operations in " << timer.time() << " seconds" << endl;
    cerr << timer.time()/r/np*1.e6 << " usec/operation" << endl;
    //Task::exit_all(0);
    Task::multiprocess(np, (void (*)(void *, int))exit, 0);
    return 0;
}

