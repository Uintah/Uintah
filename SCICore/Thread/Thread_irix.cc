
/*
 *  Thread_irix.cc: Irix implementation of the Thread library
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

/*
 * User signals:
 *    SIGQUIT: Tells all of the processes to quit
 */

//
// This is brutal, but effective.  We want this file to have access
// to the private members of Thread, without having to explicitly
// declare friendships in the class.
#define private public
#define protected public
#include <SCICore/Thread/Thread.h>
#undef private
#undef protected
#include <SCICore/Thread/AtomicCounter.h>
#include <SCICore/Thread/Barrier.h>
#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/ThreadError.h>
#include <SCICore/Thread/ThreadGroup.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Thread/WorkQueue.h>
#include "Thread_unix.h"
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
 * The irix implementation uses the default version of CrowdMonitor
 * and RecursiveMutex.  It provides native implementations of
 * AtomicCounter (using the Origin fetch&op counters if available),
 * Barrier, Mutex, and Semaphore.
 *
 */

#include "CrowdMonitor_default.cc"
#include "RecursiveMutex_default.cc"

using SCICore::Thread::Barrier;
using SCICore::Thread::Mutex;
using SCICore::Thread::Semaphore;
using SCICore::Thread::Thread;
using SCICore::Thread::ThreadError;
using SCICore::Thread::ThreadGroup;
using SCICore::Thread::Time;

extern "C" int __ateachexit(void(*)());

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
	    Thread::ThreadState state;
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
static atomic_reservoir_t reservoir;
static bool use_fetchop=true;

static usptr_t* main_sema;
static int devzero_fd;
static int main_pid;
static bool exiting;
static int exit_code;
static bool aborting;
static int nprocessors;
static usema_t* control_c_sema;

static void install_signal_handlers();

struct ThreadLocalMemory {
    Thread* current_thread;
};
ThreadLocalMemory* thread_local;

static
void
lock_scheduler()
{
    if(!initialized)
	Thread::initialize();
    if(uspsema(schedlock) == -1)
	throw ThreadError(std::string("lock_scheduler failed")
			  +strerror(errno));
}

static
void
unlock_scheduler()
{
    if(usvsema(schedlock) == -1)
	throw ThreadError(std::string("unlock_scheduler failed")
			  +strerror(errno));
}

void
Thread_shutdown(Thread* thread, bool actually_exit)
{
    Thread_private* priv=thread->d_priv;
    int pid=getpid();

    if(usvsema(priv->done) == -1)
	throw ThreadError(std::string("usvema failed on priv->done")
			  +strerror(errno));

    // Wait to be deleted...
    if(priv->delete_ready){
	if(uspsema(priv->delete_ready) == -1)
	    throw ThreadError(std::string("uspsema failed on priv->delete_ready")
			      +strerror(errno));
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
	if(!actually_exit){
	    idle[nidle]=priv;
	    nidle++;
	}
    } else {
	idle_main=priv;
    }
    unlock_scheduler();
    Thread::checkExit();
    if(pid == main_pid){
	priv->state=Thread::PROGRAM_EXIT;
	if(uspsema(main_sema) == -1)
	    throw ThreadError(std::string("uspsema failed on main_sema")
			      +strerror(errno));
    }
    if(actually_exit)
	_exit(0);
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
    Thread* self=Thread::self();
    Thread_shutdown(self, false);
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

void
Thread::allow_sgi_OpenGL_page0_sillyness()
{
#if 0
    if(mprotect(0, getpagesize(), PROT_READ|PROT_WRITE) != 0){
	fprintf(stderr, "\007\007!!! Strange error re-mapping page 0 - tell Steve this number: %d\n", errno);
    }
#endif
}

/*
 * Intialize threads for irix
 */
void
Thread::initialize()
{
#if 0
    if(mprotect(0, getpagesize(), PROT_NONE) != -1){
	//fprintf(stderr, "\007\007!!! WARNING: page 0 protected - talk to Steve if GL programs fail!\n");
    } else if(errno != EINVAL){
	fprintf(stderr, "\007\007!!! Strange error protecting page 0 - tell Steve this number: %d\n", errno);
    }
#endif
    usconfig(CONF_ARENATYPE, US_SHAREDONLY);
    usconfig(CONF_INITSIZE, 30*1024*1024);
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
    int nullfd=open("/dev/null", O_WRONLY);
    if(nullfd != 2)
	throw ThreadError("Wrong FD returned from open");
    reservoir=atomic_alloc_reservoir(USE_DEFAULT_PM, 100, NULL);
    close(2);
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

    control_c_sema=usnewsema(arena, 1);
    if(!control_c_sema)
	throw ThreadError(std::string("Error creating semaphore")
			  +strerror(errno));

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
    if(!getenv("THREAD_NO_CATCH_SIGNALS") && !getenv("DEBUGGER_SHELL"))
	install_signal_handlers();
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
	if(uspsema(priv->startup) == -1)
	    throw ThreadError(std::string("uspsema failed on priv->startup")
			      +strerror(errno));
	thread_local->current_thread=priv->thread;
	priv->state=Thread::RUNNING;
	priv->thread->run_body();
	priv->state=Thread::SHUTDOWN;
	Thread_shutdown(priv->thread, false);
	priv->state=Thread::IDLE;
    }
}

void
Thread::exit()
{
    Thread* self=Thread::self();
    Thread_shutdown(self, true);
}

void
Thread::checkExit()
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
Thread::exitAll(int code)
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
Thread::os_start(bool stopped)
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
	d_priv->stacklen=d_stacksize;
	d_priv->stackbot=(caddr_t)mmap(0, d_priv->stacklen, PROT_READ|PROT_WRITE,
				     MAP_PRIVATE, devzero_fd, 0);
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
			      +strerror(errno));
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
    if(usvsema(d_priv->startup) == -1)
	throw ThreadError(std::string("usvsema failed on d_priv->startup")
			  +strerror(errno));
}

void
Thread::stop()
{
    if(blockproc(d_priv->pid) != 0)
	throw ThreadError(std::string("blockproc returned error: ")
				      +strerror(errno));
}

void
Thread::resume()
{
    if(unblockproc(d_priv->pid) != 0)
	throw ThreadError(std::string("unblockproc returned error: ")
			  +strerror(errno));
}

void
Thread::detach()
{
    if(d_detached)
	throw ThreadError(std::string("Thread detached when already detached"));
    if(usvsema(d_priv->delete_ready) == -1)
	throw ThreadError(std::string("usvsema returned error: ")
				      +strerror(errno));
    d_detached=true;
}

void
Thread::join()
{
    Thread* us=Thread::self();
    int os=push_bstack(us->d_priv, JOINING, d_threadname);
    if(uspsema(d_priv->done) == -1)
	throw ThreadError(std::string("uspsema returned error: ")
				      +strerror(errno));

    pop_bstack(us->d_priv, os);
    detach();
}

void
ThreadGroup::gangSchedule()
{
    // This doesn't actually do anything on IRIX because it causes more
    // problems than it solves
}

/*
 * Thread block stack manipulation
 */
int
Thread::push_bstack(Thread_private* p, Thread::ThreadState state,
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
Thread::pop_bstack(Thread_private* p, int oldstate)
{
    p->bstacksize--;
    if(p->bstacksize < 0)
	throw ThreadError("Blockstack Underflow!\n");
    p->state=(Thread::ThreadState)oldstate;
}

int
Thread::get_thread_id()
{
    return d_priv->pid;
}

/*
 * Signal handling cruft
 */
void
Thread::print_threads()
{
    FILE* fp=stderr;
    for(int i=0;i<nactive;i++){
	Thread_private* p=active[i];
	const char* tname=p->thread?p->thread->getThreadName():"???";
	fprintf(fp, " %d: %s (", p->pid, tname);
	if(p->thread){
	    if(p->thread->isDaemon())
		fprintf(fp, "daemon, ");
	    if(p->thread->isDetached())
		fprintf(fp, "detached, ");
	}
	fprintf(fp, "state=%s", Thread::getStateString(p->state));
	for(int i=0;i<p->bstacksize;i++){
	    fprintf(fp, ", %s", p->blockstack[i]);
	}
	fprintf(fp, ")\n");
    }
    for(int i=0;i<nidle;i++){
	Thread_private* p=idle[i];
	fprintf(fp, " %d: Idle worker\n", p->pid);
    }
    if(idle_main){
	fprintf(fp, " %d: Completed main thread\n", idle_main->pid);
    }
}

/*
 * Handle sigquit - usually sent by control-C
 */
static
void
handle_quit(int sig, int /* code */, sigcontext_t*)
{
    if(exiting){
	if(getpid() == main_pid){
	    wait_shutdown();
	}
	exit(exit_code);
    }
    // Try to acquire a lock.  If we can't, then assume that somebody
    // else already caught the signal...
    Thread* self=Thread::self();
    if(self==0)
	return; // This is an idle thread...
    if(sig == SIGINT){
	int st=uscpsema(control_c_sema);
	if(st==-1)
	    throw ThreadError(std::string("uscpsema failed")
			      +strerror(errno));
    
	if(st == 0){
	    // This will wait until the other thread is done
	    // handling the interrupt
	    uspsema(control_c_sema);
	    usvsema(control_c_sema);
	    return;
	}
	// Otherwise, we handle the interrupt
    }

    const char* tname=self?self->getThreadName():"main?";

    // Kill all of the threads...
    char* signam=SCICore_Thread_signal_name(sig, 0);
    int pid=getpid();
    fprintf(stderr, "Thread \"%s\"(pid %d) caught signal %s\n", tname, pid, signam);
    if(sig==SIGINT){
	// Print out the thread states...
	Thread::niceAbort();
	usvsema(control_c_sema);
    } else {
	exiting=true;
	exit_code=1;
	exit(1);
    }
}

/*
 * Handle an abort signal - like segv, bus error, etc.
 */
static
void
handle_abort_signals(int sig, int /* code */, sigcontext_t* context)
{
    if(aborting)
	exit(0);
    struct sigaction action;
    sigemptyset(&action.sa_mask);
    action.sa_handler=SIG_DFL;
    action.sa_flags=0;
    if(sigaction(sig, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction (SIG_DFL) failed")
			  +strerror(errno));

    Thread* self=Thread::self();
    const char* tname=self?self->getThreadName():"idle or main";
#if defined(_LONGLONG)
    caddr_t addr=(caddr_t)context->sc_badvaddr;
#else
    caddr_t addr=(caddr_t)context->sc_badvaddr.lo32;
#endif
    char* signam=SCICore_Thread_signal_name(sig, addr);
    fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) caught signal %s\n", 7,7,7,tname, getpid(), signam);
    Thread::niceAbort();

    action.sa_handler=(SIG_PF)handle_abort_signals;
    action.sa_flags=0;
    if(sigaction(sig, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction (restore) failed")
			  +strerror(errno));
}

/*
 * Setup signals for the current thread
 */
static
void
install_signal_handlers()
{
    struct sigaction action;
    sigemptyset(&action.sa_mask);
    action.sa_flags=0;

    action.sa_handler=(SIG_PF)handle_abort_signals;
    if(sigaction(SIGILL, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction(SIGILL) failed")
			  +strerror(errno));
    if(sigaction(SIGABRT, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction(SIGABRT) failed")
			  +strerror(errno));
    if(sigaction(SIGTRAP, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction(SIGTRAP) failed")
			  +strerror(errno));
    if(sigaction(SIGBUS, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction(SIGBUS) failed")
			  +strerror(errno));
    if(sigaction(SIGSEGV, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction(SIGSEGV) failed")
			  +strerror(errno));
    if(sigaction(SIGFPE, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction(SIGFPE) failed")
			  +strerror(errno));

    action.sa_handler=(SIG_PF)handle_quit;
    if(sigaction(SIGQUIT, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction(SIGQUIT) failed")
			  +strerror(errno));
    if(sigaction(SIGINT, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction(SIGINT) failed")
			  +strerror(errno));
}

/*
 * Return the current thread...
 */
Thread*
Thread::self()
{
    return thread_local->current_thread;
}


int
Thread::numProcessors()
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
 * Yield the CPU
 */
void
Thread::yield()
{
    sginap(0);
}

/*
 * Migrate the thread to a CPU.
 */
void
Thread::migrate(int proc)
{
    if(proc==-1){
	if(sysmp(MP_RUNANYWHERE_PID, d_priv->pid) == -1)
	    throw ThreadError(std::string("sysmp(MP_RUNANYWHERE_PID) failed")
			      +strerror(errno));
    } else {
	if(sysmp(MP_MUSTRUN_PID, proc, d_priv->pid) == -1)
	    throw ThreadError(std::string("sysmp(_MP_MUSTRUN_PID) failed")
			      +strerror(errno));
    }
    d_cpu=proc;
}

/*
 * Mutex implementation
 */
Mutex::Mutex(const char* name)
    : d_name(name)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=(Mutex_private*)usnewlock(arena);
    if(!d_priv)
	throw ThreadError(std::string("usnewlock failed")
			  +strerror(errno));
}

Mutex::~Mutex()
{
    usfreelock((ulock_t)d_priv, arena);
}

void
Mutex::lock()
{
    Thread* t=Thread::self();
    int os;
    Thread_private* p=0;
    if(t){
	p=t->d_priv;
	os=Thread::push_bstack(p, Thread::BLOCK_MUTEX, d_name);
    }
    if(ussetlock((ulock_t)d_priv) == -1)
	throw ThreadError(std::string("ussetlock failed")
			  +strerror(errno));
    if(t)
	Thread::pop_bstack(p, os);
}

bool
Mutex::tryLock()
{
    int st=uscsetlock((ulock_t)d_priv, 100);
    if(st==-1)
	throw ThreadError(std::string("uscsetlock failed")
			  +strerror(errno));
    return st!=0;
}

void
Mutex::unlock()
{
    if(usunsetlock((ulock_t)d_priv) == -1)
	throw ThreadError(std::string("usunsetlock failed")
			  +strerror(errno));
}

/*
 * Semaphore implementation
 */

Semaphore::Semaphore(const char* name, int count)
    : d_name(name)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=(Semaphore_private*)usnewsema(arena, count);
    if(!d_priv)
	throw ThreadError(std::string("usnewsema failed")
			  +strerror(errno));
}

Semaphore::~Semaphore()
{
    if(d_priv)
	usfreesema((usema_t*)d_priv, arena);
}

void
Semaphore::down(int count)
{
    Thread_private* p=Thread::self()->d_priv;
    int oldstate=Thread::push_bstack(p, Thread::BLOCK_SEMAPHORE, d_name);
    for(int i=0;i<count;i++){
	if(uspsema((usema_t*)d_priv) == -1)
	    throw ThreadError(std::string("uspsema failed")
			      +strerror(errno));
    }
    Thread::pop_bstack(p, oldstate);
}

bool
Semaphore::tryDown()
{
    int stry=uscpsema((usema_t*)d_priv);
    if(stry == -1)
	throw ThreadError(std::string("uscpsema failed")
			  +strerror(errno));

    return stry;
}

void
Semaphore::up(int count)
{
    for(int i=0;i<count;i++){
	if(usvsema((usema_t*)d_priv) == -1)
	    throw ThreadError(std::string("usvsema failed")
			      +strerror(errno));
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
	    atomic_var_t* pvar;
	    char pad[128];
	    volatile int flag;  // We want this on it's own cache line
	    char pad2[128];
	};
    }
}

using SCICore::Thread::Barrier_private;

Barrier_private::Barrier_private()
    : cond0("Barrier condition 0"), cond1("Barrier condition 1"),
      mutex("Barrier lock"), nwait(0), cc(0)
{
    if(nprocessors > 1){
	if(use_fetchop){
	    flag=0;
	    pvar=atomic_alloc_variable(reservoir, 0);
	    fprintf(stderr, "***Alloc: %p\n", pvar);
	    if(!pvar)
		throw ThreadError(std::string("fetchop_alloc failed")
				  +strerror(errno));

	    storeop_store(pvar, 0);
	} else {
	    // Use normal SGI barrier
	    barrier=new_barrier(arena);
	}
    }
}   

Barrier::Barrier(const char* name)
    : d_name(name)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=new Barrier_private;
}

Barrier::~Barrier()
{
    if (nprocessors > 1) {
    	if(use_fetchop){
	    fprintf(stderr, "***Alloc free: %p\n", d_priv->pvar);
//	    atomic_free_variable(reservoir, d_priv->pvar);
        } else {
	    free_barrier(d_priv->barrier);
        }
    }
    delete d_priv;
}

void
Barrier::wait(int n)
{
    Thread_private* p=Thread::self()->d_priv;
    int oldstate=Thread::push_bstack(p, Thread::BLOCK_BARRIER, d_name);
    if(nprocessors > 1){
	if(use_fetchop){
	    int gen=d_priv->flag;
	    atomic_var_t val=atomic_fetch_and_increment(d_priv->pvar);
	    if(val == n-1){
		storeop_store(d_priv->pvar, 0);
		d_priv->flag++;
	    }
	    while(d_priv->flag==gen)
		/* spin */ ;
	} else {
	    barrier(d_priv->barrier, n);
	}
    } else {
	d_priv->mutex.lock();
	ConditionVariable& cond=d_priv->cc?d_priv->cond0:d_priv->cond1;
	d_priv->nwait++;
	if(d_priv->nwait == n){
	    // Wake everybody up...
	    d_priv->nwait=0;
	    d_priv->cc=1-d_priv->cc;
	    cond.conditionBroadcast();
	} else {
	    cond.wait(d_priv->mutex);
	}
	d_priv->mutex.unlock();
    }
    Thread::pop_bstack(p, oldstate);
}

namespace SCICore {
    namespace Thread {
	struct AtomicCounter_private {
	    AtomicCounter_private();

	    // These variables used only for non fectchop implementation
	    Mutex lock;
	    int value;
	};
    }
}

using SCICore::Thread::AtomicCounter_private;
using SCICore::Thread::AtomicCounter;

AtomicCounter_private::AtomicCounter_private()
    : lock("AtomicCounter lock")
{
}

AtomicCounter::AtomicCounter(const char* name)
    : d_name(name)
{
    if(!initialized){
	Thread::initialize();
    }
    if(use_fetchop){
	d_priv=(AtomicCounter_private*)atomic_alloc_variable(reservoir, 0);
 	fprintf(stderr, "***Alloc atomcounter: %p\n", d_priv);
	if(!d_priv)
	    throw ThreadError(std::string("fetchop_alloc failed")
					  +strerror(errno));
    } else {
	d_priv=new AtomicCounter_private;
    }
}

AtomicCounter::AtomicCounter(const char* name, int value)
    : d_name(name)
{
    if(!initialized){
	Thread::initialize();
    }
    if(use_fetchop){
	d_priv=(AtomicCounter_private*)atomic_alloc_variable(reservoir, 0);
 	fprintf(stderr, "***Alloc atomcounter: %p\n", d_priv);
	if(!d_priv)
	    throw ThreadError(std::string("fetchop_alloc failed")
					  +strerror(errno));
	atomic_store((atomic_var_t*)d_priv, value);
    } else {
	d_priv=new AtomicCounter_private;
	d_priv->value=value;
    }
}

AtomicCounter::~AtomicCounter()
{
    if(use_fetchop){
 	fprintf(stderr, "***Alloc free: %p\n", d_priv);
//	atomic_free_variable(reservoir, (atomic_var_t*)d_priv);
    } else {
	delete d_priv;
    }
}

AtomicCounter::operator int() const
{
    if(use_fetchop){
	return (int)atomic_load((atomic_var_t*)d_priv);
    } else {
	return d_priv->value;
    }
}

int
AtomicCounter::operator++()
{
    if(use_fetchop){
	// We do not use the couldBlock/couldBlockDone pairs here because
	// they are so fast (microsecond), and never block...
	return (int)atomic_fetch_and_increment((atomic_var_t*)d_priv)+1;
    } else {
	int oldstate=Thread::couldBlock(d_name);
	d_priv->lock.lock();
	int ret=++d_priv->value;
	d_priv->lock.unlock();
	Thread::couldBlockDone(oldstate);
	return ret;
    }
}

int
AtomicCounter::operator++(int)
{
    if(use_fetchop){
	return (int)atomic_fetch_and_increment((atomic_var_t*)d_priv);
    } else {
	int oldstate=Thread::couldBlock(d_name);
	d_priv->lock.lock();
	int ret=d_priv->value++;
	d_priv->lock.unlock();
	Thread::couldBlockDone(oldstate);
	return ret;
    }
}

int
AtomicCounter::operator--()
{
    if(use_fetchop){
	return (int)atomic_fetch_and_decrement((atomic_var_t*)d_priv)-1;
    } else {
	int oldstate=Thread::couldBlock(d_name);
	d_priv->lock.lock();
	int ret=--d_priv->value;	
	d_priv->lock.unlock();
	Thread::couldBlockDone(oldstate);
	return ret;
    }
}

int
AtomicCounter::operator--(int)
{
    if(use_fetchop){
	return (int)atomic_fetch_and_increment((atomic_var_t*)d_priv);
    } else {
	int oldstate=Thread::couldBlock(d_name);
	d_priv->lock.lock();
	int ret=d_priv->value--;
	d_priv->lock.unlock();
	Thread::couldBlockDone(oldstate);
	return ret;
    }
}

void
AtomicCounter::set(int v)
{
    if(use_fetchop){
	atomic_store((atomic_var_t*)d_priv, v);
    } else {
	int oldstate=Thread::couldBlock(d_name);
	d_priv->lock.lock();
	d_priv->value=v;
	d_priv->lock.unlock();
	Thread::couldBlockDone(oldstate);
    }
}

namespace SCICore {
    namespace Thread {
	struct ConditionVariable_private {
	    int num_waiters;
	    usema_t* semaphore;
	    bool pollsema;
	};
    }
}

SCICore::Thread::ConditionVariable::ConditionVariable(const char* name)
    : d_name(name)
{
    d_priv=new ConditionVariable_private();
    d_priv->num_waiters=0;
    d_priv->pollsema=false;
    d_priv->semaphore=usnewsema(arena, 0);
    if(!d_priv->semaphore)
	throw ThreadError(std::string("usnewsema failed")
			  +strerror(errno));
}

SCICore::Thread::ConditionVariable::~ConditionVariable()
{
    if(d_priv->semaphore){
	if(d_priv->pollsema)
	    usfreepollsema(d_priv->semaphore, arena);
	else
	    usfreesema(d_priv->semaphore, arena);
    }
    delete d_priv;
}

void
SCICore::Thread::ConditionVariable::wait(Mutex& m)
{
    if(d_priv->pollsema){
	if(!timedWait(m, 0)){
	    throw ThreadError("timedWait with infinite timeout didn't wait");
	}
    } else {
	d_priv->num_waiters++;
	m.unlock();
	Thread_private* p=Thread::self()->d_priv;
	int oldstate=Thread::push_bstack(p, Thread::BLOCK_CONDITIONVARIABLE, d_name);
	// Block until woken up by signal or broadcast
	if(uspsema(d_priv->semaphore) == -1)
	    throw ThreadError(std::string("uspsema failed")
			      +strerror(errno));
	Thread::couldBlockDone(oldstate);
	m.lock();
    }
}

/*
 * pollable semaphores are strange beasts.  This code isn't meant
 * to be understood.
 */
bool
SCICore::Thread::ConditionVariable::timedWait(Mutex& m, const struct timespec* abstime)
{
    if(!d_priv->pollsema){
	// Convert to a pollable semaphore...
	if(d_priv->num_waiters){
	    // There would be much code for this case, so I hope it
	    // never happens.
	    throw ThreadError("Cannot convert to timedWait/pollable semaphore if regular wait is in effect");
	} else {
	    usfreesema(d_priv->semaphore, arena);
	    d_priv->pollsema=true;
	    d_priv->semaphore=usnewpollsema(arena, 0);
	    if(!d_priv->semaphore)
		throw ThreadError(std::string("usnewpollsema failed")
				  +strerror(errno));
	}
    }

    d_priv->num_waiters++;

    /*
     * Open the file descriptor before we unlock the mutex because
     * it is required for the conditionSignal to call usvsema.
     */
    int pollfd = usopenpollsema(d_priv->semaphore, S_IRWXU);
    if(pollfd == -1)
	throw ThreadError(std::string("usopenpollsema failed")
			  +strerror(errno));

    m.unlock();

    Thread_private* p=Thread::self()->d_priv;
    int oldstate=Thread::push_bstack(p, Thread::BLOCK_CONDITIONVARIABLE, d_name);

    // Block until woken up by signal or broadcast or timed out
    bool success;

    int ss = uspsema(d_priv->semaphore);
    if(ss == -1){
	throw ThreadError(std::string("uspsema failed")
			  +strerror(errno));
    } else if(ss == 0){
	for(;;){
	    struct timeval timeout;
	    struct timeval* timeoutp;
	    if(abstime){
		struct timeval now;
		if(gettimeofday(&now, 0) != 0)
		    throw ThreadError(std::string("gettimeofday failed")
				      +strerror(errno));
		/* Compute the difference. */
		timeout.tv_sec = abstime->tv_sec - now.tv_sec;
		timeout.tv_usec = abstime->tv_nsec/1000 - now.tv_usec;
		if(timeout.tv_usec < 0){
		    int secs = (-timeout.tv_usec)/1000000+1;
		    timeout.tv_sec-=secs;
		    timeout.tv_usec+=secs*1000000;
		}
		if(timeout.tv_sec < 0){
		    timeout.tv_sec=0;
		    timeout.tv_usec=0;
		}
		timeoutp=&timeout;
	    } else {
		timeoutp=0;
	    }

	    fd_set fds;
	    FD_ZERO(&fds);
	    FD_SET(pollfd, &fds);
	    int s = select(pollfd+1, &fds, 0, 0, timeoutp);
	    if(s == -1){
		if(errno == EINTR)
		    continue;
		throw ThreadError(std::string("select failed")
				  +strerror(errno));
	    } else if(s == 0){
		// Timed out...
		success = false;
		break;
	    } else {
		// Got it
		success=true;
		break;
	    }
	}
    } else {
	// Semaphore available...
	success=true;
    }

    // Lock now so that the usvsema below will not interfere
    // with a poorly timed signal
    m.lock();
    if(!success){
	// v the semaphore so that the next p will work
	if(usvsema(d_priv->semaphore) == -1)
	    throw ThreadError(std::string("usvsema failed")
			      +strerror(errno));
	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(pollfd, &fds);
	timeval zerotimeout;
	zerotimeout.tv_sec = 0;
	zerotimeout.tv_usec = 0;
	int s = select(pollfd+1, &fds, 0, 0, &zerotimeout);
	if(s == -1){
	    throw ThreadError(std::string("select failed")
			      +strerror(errno));
	} else if(s == 0){
	    // Timed out...
	    throw ThreadError("ConditionVariable timeout recover failed");
	} else {
	    // Got it
	}
    }
    if(usclosepollsema(d_priv->semaphore) != 0)
	throw ThreadError(std::string("usclosepollsema failed")
			  +strerror(errno));
    Thread::couldBlockDone(oldstate);
    return success;
}

void
SCICore::Thread::ConditionVariable::conditionSignal()
{
    if(d_priv->num_waiters > 0){
        d_priv->num_waiters--;
	if(usvsema(d_priv->semaphore) == -1)
	    throw ThreadError(std::string("usvsema failed")
			      +strerror(errno));
    }
}

void
SCICore::Thread::ConditionVariable::conditionBroadcast()
{
    while(d_priv->num_waiters > 0){
        d_priv->num_waiters--;
	if(usvsema(d_priv->semaphore) == -1)
	    throw ThreadError(std::string("usvsema failed")
			      +strerror(errno));
    }
}

//
// $Log$
// Revision 1.17  2000/02/22 17:23:25  mmiller
// Added check for 1 processor to Barrier destructor to prevent destructor from
// freeing uninitialize barrier data member.  Was causing a SEGV.
//
// Revision 1.16  2000/02/16 00:29:45  sparker
// Comented out thread changes for now
//
// Revision 1.15  2000/02/15 00:23:51  sparker
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
// Revision 1.14  1999/11/09 08:15:51  dmw
// bumped up the arena size in order to fix Mesh problem
//
// Revision 1.13  1999/10/15 20:56:52  ikits
// Fixed conflict w/ get_tid in /usr/include/task.h. Replaced by get_thread_id.
//
// Revision 1.12  1999/09/29 06:05:46  dmw
// commented out atomic_free_variable lines - SGI bug
//
// Revision 1.11  1999/09/22 19:10:29  sparker
// Implemented timedWait method for ConditionVariable.  A default
// implementation of CV is no longer possible, so the code is moved
// to Thread_irix.cc.  The timedWait method for irix uses uspollsema
// and select.
//
// Revision 1.10  1999/09/05 05:58:33  sparker
// Fixed bug in handling of stderr, where stderr would sometimes
// not work.  In order to prevent extraneous print statements from
// the fetchop library, we play games with the stderr file descriptor.
// Now it is pointed to /dev/null temporarily instead of just temporarily
// closed.
//
// Revision 1.9  1999/09/01 22:31:11  sparker
// Changed mmap of thread stacks so that fork will work.
//
// Revision 1.8  1999/08/31 08:59:05  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.7  1999/08/29 00:47:02  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.6  1999/08/28 03:46:52  sparker
// Final updates before integration with PSE
//
// Revision 1.5  1999/08/25 22:36:02  sparker
// More thread library updates - now compiles
//
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
