
/* REFERENCED */
static char *id="$Id$";

/*
 *  Thread_irix.cc: Irix implementation of the Thread library
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

/*
 * User signals:
 *    SIGQUIT: Tells all of the processes to quit
 *    SIGUSR1: Tells another thread to start profiling.
 *    SIGUSR2: Tells the main thread when another thread exits.
 *             NOTE - this might not be one of our threads!
 */

#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/AtomicCounter.h>
#include <SCICore/Thread/Barrier.h>
#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/ThreadError.h>
#include <SCICore/Thread/ThreadGroup.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Thread/WorkQueue.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <ulocks.h>
#include <unistd.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <sys/signal.h>
#include <sys/sysmp.h>
#include <sys/syssgi.h>
#include <sys/time.h>
#include <sys/types.h>
extern "C" {
#include <sys/pmo.h>
#include <fetchop.h>
}

/*
 * The irix implementation uses the default version of ConditionVariable,
 * CrowdMonitor, and RecursiveMutex.  It provides native implementations
 * of AtomicCounter (using the Origin fetch&op counters if available),
 * Barrier, Mutex, and Semaphore.
 *
 */

#include "ConditionVariable_default.cc"
#include "CrowdMonitor_default.cc"
#include "RecursiveMutex_default.cc"

using SCICore::Thread::Thread;
using SCICore::Thread::ThreadGroup;
using SCICore::Thread::Time;

#define TOPBIT ((unsigned int)0x80000000)

extern "C" int __ateachexit(void(*)());
#define THREAD_STACKSIZE 256*1024*4

#define MAXBSTACK 10
#define MAXTHREADS 4000

#define N_POOLMUTEX 301

namespace SCICore {
    namespace Thread {
	struct Thread_private {
	    Thread* thread;
	    int pid;
	    caddr_t sp;
	    caddr_t stackbot;
	    size_t stacklen;
	    usema_t* startup;
	    usema_t* done;
	    usema_t* delete_ready;
	    int state;
	    int bstacksize;
	    const char* blockstack[MAXBSTACK];
	};
    }
}
using SCICore::Thread::Thread_private;

static Thread_private* idle_main;
static Thread_private* idle[MAXTHREADS];
static int nidle;
static Thread_private* active[MAXTHREADS];
static int nactive;
static bool initialized;
static usema_t* schedlock;
static usptr_t* arena;
static fetchop_reservoir_t reservoir;
static bool use_fetchop=true;

static usptr_t* main_sema;
static int devzero_fd;
static int main_pid;
static bool exiting;
static int exit_code;
static bool aborting;
static int nprocessors;
static usema_t* control_c_sema;



struct ThreadLocalMemory {
    Thread* current_thread;
};
ThreadLocalMemory* thread_local;

static
void
lock_scheduler()
{
    if(uspsema(schedlock) == -1){
	perror("uspsema");
	Thread::niceAbort();
    }
}

static
void
unlock_scheduler()
{
    if(usvsema(schedlock) == -1){
	perror("usvsema");
	Thread::niceAbort();
    }
}

static
Thread_private*
find_thread_from_tid(int tid)
{
    for(int i=0;i<nactive;i++){
	Thread_private* p=active[i];
	if(p->pid == tid)
	    return p;
    }
    return 0;
}

/*
 * This is the ateachexit handler
 */
static
void
Thread_exit()
{
    if(exiting)
	return;
    Thread::exit();
}

static
void
wait_shutdown()
{
    long n;
    if((n=prctl(PR_GETNSHARE)) > 1){
	fprintf(stderr, "Waiting for %d threads to shut down: ", (int)(n-1));
	sginap(10);
	int delay=10;
	while((n=prctl(PR_GETNSHARE)) > 1){
	    fprintf(stderr, "%d...", (int)(n-1));
	    sginap(delay);
	    delay+=10;
	}
	fprintf(stderr, "done\n");
    }
}

/*
 * Intialize threads for irix
 */
void
SCICore::Thread::Thread::initialize()
{
    usconfig(CONF_ARENATYPE, US_SHAREDONLY);
    usconfig(CONF_INITSIZE, 3*1024*1024);
    usconfig(CONF_INITUSERS, (unsigned int)140);
    arena=usinit("/dev/zero");
    if(!arena)
	throw ThreadError(std::string("Error calling usinit: ")
			  +strerror(errno));

    //
    // Fetchop init prints out a really annoying message.  This
    // ugly code is to make it not do that
    // failed to create fetchopable area error
    // 
    int tmpfd=dup(2);
    close(2);
    reservoir=fetchop_init(USE_DEFAULT_PM, 10);
    int newfd=dup(tmpfd);
    if(newfd != 2)
	throw ThreadError("Wrong FD returned from dup!");
    close(tmpfd);

    if(!reservoir){
	use_fetchop=false;
    } else {
	use_fetchop=true;
    }

    devzero_fd=open("/dev/zero", O_RDWR);
    if(devzero_fd == -1)
	throw ThreadError(std::string("Error opening /dev/zero: ")
			  +strerror(errno));

    schedlock=usnewsema(arena, 1);
    main_sema=usnewsema(arena, 0);
    nprocessors=Thread::numProcessors();

    main_pid=getpid();
    /*
     * The functionality relies on SGI's __ateachexit function.  If
     * that ever goes away, then we will probably have to write our
     * own exit.
     */
    __ateachexit(Thread_exit);

    thread_local=(ThreadLocalMemory*)mmap(0, sizeof(ThreadLocalMemory),
					  PROT_READ|PROT_WRITE,
					  MAP_PRIVATE|MAP_LOCAL,
					  devzero_fd, 0);
    /*
     * We have to say that we are initialized here, because the
     * following calls can create synchronization primitives that
     * will rely only on the above initialization.
     */
    initialized=true;

    ThreadGroup::s_default_group=new ThreadGroup("default group", 0);
    Thread* mainthread=new Thread(ThreadGroup::s_default_group, "main");
    mainthread->d_priv=new Thread_private;
    mainthread->d_priv->pid=main_pid;
    mainthread->d_priv->thread=mainthread;
    mainthread->d_priv->state=RUNNING;
    mainthread->d_priv->bstacksize=0;
    mainthread->d_priv->done=usnewsema(arena, 0);
    mainthread->d_priv->delete_ready=0;
    lock_scheduler();
    active[nactive]=mainthread->d_priv;
    nactive++;
    unlock_scheduler();

    thread_local->current_thread=mainthread;
    if(!getenv("THREAD_NO_CATCH_SIGNALS"))
	install_signal_handlers();

}

void
Thread_run(Thread* t)
{
    t->run_body();
}

static
void
run_threads(void* priv_v, size_t)
{
    Thread_private* priv=(Thread_private*)priv_v;
    if(!getenv("THREAD_NO_CATCH_SIGNALS"))
	install_signal_handlers();
    for(;;){
	/* Wait to be started... */
	priv->state=Thread::IDLE;
	if(uspsema(priv->startup) == -1){
	    perror("uspsema");
	    Thread::niceAbort();
	}
	thread_local->current_thread=priv->thread;
	priv->state=Thread::RUNNING;
	Thread_run(priv->thread);
	priv->state=Thread::SHUTDOWN;
	Thread::exit();
	priv->state=Thread::IDLE;
    }
}

void
Thread_shutdown(Thread* thread)
{
    Thread_private* priv=thread->d_priv;
    int pid=getpid();

    if(usvsema(priv->done) == -1){
	perror("usvsema");
	Thread::niceAbort();
    }

    // Wait to be deleted...
    if(priv->delete_ready){
	if(uspsema(priv->delete_ready) == -1) {
	    perror("uspsema");
	    Thread::niceAbort();
	}
    }
    // Allow this thread to run anywhere...
    if(thread->d_cpu != -1)
	thread->migrate(-1);

    delete thread;

    priv->thread=0;
    thread_local->current_thread=0;
    lock_scheduler();
    /* Remove it from the active queue */
    int i;
    for(i=0;i<nactive;i++){
	if(active[i]==priv)
	    break;
    }
    for(i++;i<nactive;i++){
	active[i-1]=active[i];
    }
    nactive--;
    if(priv->pid != main_pid){
	idle[nidle]=priv;
	nidle++;
    } else {
	idle_main=priv;
    }
    unlock_scheduler();
    Thread::checkExit();
    if(pid == main_pid){
	priv->state=Thread::PROGRAM_EXIT;
	if(uspsema(main_sema) == -1){
	    perror("uspsema");
	    Thread::niceAbort();
	}
    }
}

void
SCICore::Thread::Thread::checkExit()
{
    lock_scheduler();
    int done=true;
    for(int i=0;i<nactive;i++){
	Thread_private* p=active[i];
	if(!p->thread->isDaemon()){
	    done=false;
	    break;
	}
    }
    unlock_scheduler();

    if(done)
	Thread::exitAll(0);
}

void
SCICore::Thread::Thread::exitAll(int code)
{
    exiting=true;
    exit_code=code;

    // We want to do this:
    //   kill(0, SIGQUIT);
    // but that has the unfortunate side-effect of sending a signal
    // to things like par, perfex, and other things that somehow end
    // up in the same share group.  So we have to kill each process
    // independently...
    pid_t me=getpid();
    for(int i=0;i<nidle;i++){
	if(idle[i]->pid != me)
	    kill(idle[i]->pid, SIGQUIT);
    }
    for(int i=0;i<nactive;i++){
	if(active[i]->pid != me)
	    kill(active[i]->pid, SIGQUIT);
    }
    if(idle_main && idle_main->pid != me)
	kill(idle_main->pid, SIGQUIT);
    kill(me, SIGQUIT);
    fprintf(stderr, "Should not reach this point!\n");
}

/*
 * Startup the given thread...
 */
void
SCICore::Thread::Thread::os_start(bool stopped)
{
    /* See if there is a thread waiting around ... */
    if(!initialized){
	Thread::initialize();
    }
    lock_scheduler();
    if(nidle){
	nidle--;
	d_priv=idle[nidle];
    } else {
	d_priv=new Thread_private;
	d_priv->stacklen=THREAD_STACKSIZE;
	d_priv->stackbot=(caddr_t)mmap(0, d_priv->stacklen, PROT_READ|PROT_WRITE,
				     MAP_SHARED, devzero_fd, 0);
	d_priv->sp=d_priv->stackbot+d_priv->stacklen-1;
	if((long)d_priv->sp == -1)
	    throw ThreadError(std::string("Not enough memory for thread stack")
			      +strerror(errno));
	d_priv->startup=usnewsema(arena, 0);
	d_priv->done=usnewsema(arena, 0);
	d_priv->delete_ready=usnewsema(arena, 0);
	d_priv->state=STARTUP;
	d_priv->bstacksize=0;
	d_priv->pid=sprocsp(run_threads, PR_SALL, d_priv, d_priv->sp, d_priv->stacklen);
	if(d_priv->pid == -1)
	    throw ThreadError(std::string("Cannot start new thread")
			      +strerror(sprocsp));
    }
    d_priv->thread=this;
    active[nactive]=d_priv;
    nactive++;
    unlock_scheduler();
    if(stopped){
	if(blockproc(d_priv->pid) != 0)
	    throw ThreadError(std::string("blockproc returned error: ")
			      +strerror(errno));
    }
    /* The thread is waiting to be started, release it... */
    if(usvsema(d_priv->startup) == -1){
	perror("usvsema");
	Thread::niceAbort();
    }
    //Thread::event(this, THREAD_START);
}

void
SCICore::Thread::Thread::stop()
{
    if(blockproc(d_priv->pid) != 0)
	throw ThreadError(std::string("blockproc returned error: ")
				      +strerror(errno));
}

void
SCICore::Thread::Thread::resume()
{
    if(unblockproc(d_priv->pid) != 0)
	throw ThreadError(std::string("unblockproc returned error: ")
				      +strerror(errno));
}

void
SCICore::Thread::Thread::detach()
{
    if(d_detached)
	throw ThreadError(std::string("Thread detached when already detached"));
    if(usvsema(d_priv->delete_ready) == -1)
	throw ThreadError(std::string("usvsema returned error: ")
				      +strerror(errno));
    d_detached=true;
}

void
SCICore::Thread::Thread::join()
{
    Thread* us=Thread::self();
    int os=push_bstack(us->d_priv, JOINING, d_threadname);
    if(uspsema(d_priv->done) == -1)
	throw ThreadError(std::string("uspsema returned error: ")
				      +strerror(errno));

    pop_bstack(us->d_priv, os);
    detach();
}

/*
 * Thread stack manipulation
 */
int
SCICore::Thread::Thread::push_bstack(Thread_private* p,
				     SCICore::Thread::Thread::ThreadState state,
				     const char* name)
{
    int oldstate=p->state;
    p->state=state;
    p->blockstack[p->bstacksize]=name;
    p->bstacksize++;
    if(p->bstacksize>=MAXBSTACK)
	throw ThreadError("Blockstack Overflow!\n");
    return oldstate;
}

void
SCICore::Thread::Thread::pop_bstack(Thread_private* p, int oldstate)
{
    p->bstacksize--;
    if(p->bstacksize < 0)
	throw ThreadError("Blockstack Underflow!\n");
    p->state=oldstate;
}

/*
 * Return the current thread...
 */
Thread*
SCICore::Thread::Thread::self()
{
    return thread_local->current_thread;
}


int
SCICore::Thread::Thread::numProcessors()
{
    static int nproc=-1;
    if(nproc==-1){
	char* np=getenv("SCI_NPROCESSORS");
	if(np){
	    nproc = atoi(np);
	    if(nproc<=0)
		nproc=1;
	} else {
	    nproc = (int)sysmp(MP_NAPROCS);
	}
    }
    return nproc;
}

/*
 * Migrate the thread to a CPU.
 */
void
SCICore::Thread::Thread::migrate(int proc)
{
    if(proc==-1){
	if(sysmp(MP_RUNANYWHERE_PID, d_priv->pid) == -1){
	    perror("sysmp - MP_RUNANYWHERE_PID");
	}
    } else {
	if(sysmp(MP_MUSTRUN_PID, proc, d_priv->pid) == -1){
	    perror("sysmp - MP_MUSTRUN_PID");
	}
    }
    d_cpu=proc;
}

/*
 * Mutex implementation
 */

SCICore::Thread::Mutex::Mutex(const char* name)
    : d_name(name)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=(Mutex_private*)usnewlock(arena);
    if(!d_priv){
	perror("usnewlock");
	Thread::niceAbort();
    }
}

SCICore::Thread::Mutex::~Mutex()
{
    usfreelock((ulock_t)d_priv, arena);
}

void
SCICore::Thread::Mutex::lock()
{
    Thread* t=Thread::self();
    int os;
    Thread_private* p=0;
    if(t){
	p=t->d_priv;
	os=Thread::push_bstack(p, Thread::BLOCK_MUTEX, d_name);
    }
    if(ussetlock((ulock_t)d_priv) == -1){
	perror("ussetlock");
	Thread::niceAbort();
    }
    if(t)
	Thread::pop_bstack(p, os);
}

bool
SCICore::Thread::Mutex::tryLock()
{
    int st=uscsetlock((ulock_t)d_priv, 100);
    if(st==-1){
	perror("uscsetlock");
	Thread::niceAbort();
    }
    return st!=0;
}

void
SCICore::Thread::Mutex::unlock()
{
    if(usunsetlock((ulock_t)d_priv) == -1){
	perror("usunsetlock");
	Thread::niceAbort();
    }
}

/*
 * Semaphore implementation
 */

SCICore::Thread::Semaphore::Semaphore(const char* name, int count)
    : d_name(name)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=(Semaphore_private*)usnewsema(arena, count);
    if(!d_priv){
	perror("usnewsema");
	Thread::niceAbort();
    }
}

SCICore::Thread::Semaphore::~Semaphore()
{
    if(d_priv)
	usfreesema((usema_t*)d_priv, arena);
}

void
SCICore::Thread::Semaphore::down(int count)
{
    Thread_private* p=Thread::self()->d_priv;
    int oldstate=push_bstack(p, BLOCK_SEMAPHORE, name);
    for(int i=0;i<count;i++){
	if(uspsema((usema_t*)d_priv) == -1){
	    perror("upsema");
	    Thread::niceAbort();
	}
    }
    Thread::pop_bstack(p, oldstate);
}

bool
SCICore::Thread::Semaphore::tryDown()
{
    int stry=uscpsema((usema_t*)d_priv);
    if(stry == -1){
	perror("upsema");
	Thread::niceAbort();
    }
    return stry;
}

void
SCICore::Thread::Semaphore::up(int count)
{
    for(int i=0;i<count;i++){
	if(usvsema((usema_t*)d_priv) == -1){
	    perror("usvsema");
	    Thread::niceAbort();
	}
    }
}

namespace SCICore {
    namespace Thread {
	struct Barrier_private {
	    Barrier_private();

	    // These variables used only for single processor implementation
	    Mutex mutex;
	    ConditionVariable cond0;
	    ConditionVariable cond1;
	    int cc;
	    int nwait;

	    // Only for MP, non fetchop implementation
	    barrier_t* barrier;

	    // Only for fetchop implementation
	    fetchop_var_t* pvar;
	    char pad[128];
	    int flag;  // We want this on it's own cache line
	    char pad2[128];
	};
    }
}

SCICore::Thread::Barrier_private::Barrier_private()
    : cond0("Barrier condition 0"), cond1("Barrier condition 1"),
      mutex("Barrier lock"), nwait(0), cc(0)
{
    if(nprocessors > 1){
	if(use_fetchop){
	    flag=0;
	    pvar=fetchop_alloc(reservoir);
	    fprintf(stderr, "***Alloc: %p\n", pvar);
	    if(!pvar){
		perror("fetchop_alloc");
		Thread::niceAbort();
	    }
	    storeop_store(pvar, 0);
	} else {
	    // Use normal SGI barrier
	    barrier=new_barrier(arena);
	}
    }
}   

SCICore::Thread::Barrier::Barrier(const char* name, int nthreads)
    : d_name(name), d_num_threads(nthreads), d_thread_group(0)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=new Barrier_private;
}

SCICore::Thread::Barrier::Barrier(const char* name, ThreadGroup* threadGroup)
    : d_name(name), d_num_threads(0), d_thread_group(threadGroup)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=new Barrier_private;
}

SCICore::Thread::Barrier::~Barrier()
{
    if(use_fetchop){
	fetchop_free(reservoir, d_priv->pvar);
    } else {
	free_barrier(d_priv->barrier);
    }
    delete priv;
}

void
SCICore::Thread::Barrier::wait()
{
    int n=d_thread_group?d_thread_group->nactive(true):d_num_threads;
    Thread_private* p=Thread::self()->d_priv;
    int oldstate=push_bstack(p, BLOCK_BARRIER, name);
    if(nprocessors > 1){
	if(use_fetchop){
	    int gen=d_priv->flag;
	    fetchop_var_t val=fetchop_increment(d_priv->pvar);
	    if(val == n-1){
		storeop_store(priv->pvar, 0);
		priv->flag++;
	    }
	    while(priv->flag==gen)
		/* spin */ ;
	} else {
	    barrier(d_priv->barrier, n);
	}
    } else {
	d_priv->mutex.lock();
	ConditionVariable& cond=priv->cc?priv->cond0:priv->cond1;
	priv->nwait++;
	if(priv->nwait == n){
	    // Wake everybody up...
	    priv->nwait=0;
	    priv->cc=1-priv->cc;
	    cond.conditionBroadcast();
	} else {
	    cond.wait(priv->mutex);
	}
	priv->mutex.unlock();
    }
    pop_bstack(p, oldstate);
}

//
// $Log$
// Revision 1.4  1999/08/25 19:00:52  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:38:02  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
