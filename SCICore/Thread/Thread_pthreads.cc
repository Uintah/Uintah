
/*
 *  Thread_pthreads.cc: Posix threads implementation of the thread library
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

#define __USE_UNIX98
#include <pthread.h>
#define private public
#define protected public
#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/Mutex.h> // So ConditionVariable can get to Mutex::d_priv
#undef private
#undef protected
#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/AtomicCounter.h>
#include <SCICore/Thread/Barrier.h>
#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/RecursiveMutex.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/ThreadError.h>
#include <SCICore/Thread/ThreadGroup.h>
#include <SCICore/Thread/WorkQueue.h>
#include "Thread_unix.h"
#include <errno.h>
#include <iostream.h>
extern "C" {
#include <semaphore.h>
};
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

typedef void (*SIG_HANDLER_T)(int);

/*
 * The pthread implementation uses the default version of AtomicCounter,
 * Barrier, and CrowdMonitor.  It provides native implementations of
 * of ConditionVariable, Mutex, RecursiveMutex and Semaphore.
 *
 */

#include "AtomicCounter_default.cc"
#include "Barrier_default.cc"
#include "CrowdMonitor_default.cc"

using SCICore::Thread::ConditionVariable;
using SCICore::Thread::Mutex;
using SCICore::Thread::RecursiveMutex;
using SCICore::Thread::Semaphore;
using SCICore::Thread::Thread;
using SCICore::Thread::ThreadError;
using SCICore::Thread::ThreadGroup;
bool exiting=false;

#define MAXBSTACK 10
#define MAXTHREADS 4000

namespace SCICore {
    namespace Thread {
	struct Thread_private {
	    Thread* thread;
	    pthread_t threadid;
	    Thread::ThreadState state;
	    int bstacksize;
	    const char* blockstack[MAXBSTACK];
	    sem_t done;
	    sem_t delete_ready;
	    sem_t block_sema;
	};
    }
}

using SCICore::Thread::Thread_private;

static Thread_private* active[MAXTHREADS];
static int numActive;
static bool initialized;
static pthread_mutex_t sched_lock;
static pthread_key_t thread_key;
static sem_t main_sema;
static sem_t control_c_sema;

static
void
lock_scheduler()
{
    if(pthread_mutex_lock(&sched_lock))
	throw ThreadError(std::string("pthread_mutex_lock failed")
			  +strerror(errno));
}

static
void
unlock_scheduler()
{
    if(pthread_mutex_unlock(&sched_lock))
	throw ThreadError(std::string("pthread_mutex_unlock failed")
			  +strerror(errno));
}

int
Thread::push_bstack(Thread_private* p, Thread::ThreadState state,
		    const char* name)
{
    int oldstate=p->state;
    p->state=state;
    p->blockstack[p->bstacksize]=name;
    p->bstacksize++;
    if(p->bstacksize>MAXBSTACK){
	fprintf(stderr, "Blockstack Overflow!\n");
	Thread::niceAbort();
    }
    return oldstate;
}

void
Thread::pop_bstack(Thread_private* p, int oldstate)
{
    p->bstacksize--;
    p->state=(ThreadState)oldstate;
}

void
ThreadGroup::gangSchedule()
{
    // Cannot do this on pthreads unfortunately
}

static
void
Thread_shutdown(Thread* thread)
{
    Thread_private* priv=thread->d_priv;

    if(sem_post(&priv->done) != 0)
	throw ThreadError(std::string("sem_post failed")
			  +strerror(errno));

    delete thread;

    // Wait to be deleted...
    if(sem_wait(&priv->delete_ready) == -1)
	throw ThreadError(std::string("sem_wait failed")
			  +strerror(errno));

    // Allow this thread to run anywhere...
    if(thread->d_cpu != -1)
	thread->migrate(-1);

    priv->thread=0;
    lock_scheduler();
    /* Remove it from the active queue */
    int i;
    for(i=0;i<numActive;i++){
	if(active[i]==priv)
	    break;
    }
    for(i++;i<numActive;i++){
	active[i-1]=active[i];
    }
    numActive--;
    unlock_scheduler();
    Thread::checkExit();
    if(priv->threadid == 0){
	priv->state=Thread::PROGRAM_EXIT;
	if(sem_wait(&main_sema) == -1)
	    throw ThreadError(std::string("sem_wait failed")
			      +strerror(errno));
    }
    pthread_exit(0);
}

void
Thread::exit()
{
    Thread* self=Thread::self();
    Thread_shutdown(self);
}

void
Thread::checkExit()
{
    lock_scheduler();
    int done=true;
    for(int i=0;i<numActive;i++){
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

Thread*
Thread::self()
{
    void* p=pthread_getspecific(thread_key);
    return (Thread*)p;
}

void
Thread::join()
{
    Thread* us=Thread::self();
    int os=push_bstack(us->d_priv, JOINING, d_threadname);
    if(sem_wait(&d_priv->done) != 0)
	throw ThreadError(std::string("sem_wait failed")
			  +strerror(errno));
    pop_bstack(us->d_priv, os);
    detach();
}

int
Thread::numProcessors()
{
    return 1;
}

void
Thread_run(Thread* t)
{
    t->run_body();
}

static
void*
run_threads(void* priv_v)
{
    Thread_private* priv=(Thread_private*)priv_v;
    if(sem_wait(&priv->block_sema) != 0)
	throw ThreadError(std::string("sem_wait: ")
			  +strerror(errno));
    if(pthread_setspecific(thread_key, priv->thread) != 0)
	throw ThreadError(std::string("pthread_setspecific failed")
			  +strerror(errno));
    priv->state=Thread::RUNNING;
    Thread_run(priv->thread);
    priv->state=Thread::SHUTDOWN;
    Thread_shutdown(priv->thread);
    return 0; // Never reached
}

void
Thread::os_start(bool stopped)
{
    if(!initialized)
	Thread::initialize();

    d_priv=new Thread_private;

    if(sem_init(&d_priv->done, 0, 0) != 0)
	throw ThreadError(std::string("sem_init failed")
			  +strerror(errno));
    if(sem_init(&d_priv->delete_ready, 0, 0) != 0)
	throw ThreadError(std::string("sem_init failed")
			  +strerror(errno));
    d_priv->state=STARTUP;
    d_priv->bstacksize=0;
    d_priv->thread=this;
    d_priv->threadid=0;

    if(sem_init(&d_priv->block_sema, 0, stopped?0:1))
	throw ThreadError(std::string("sem_init failed")
			  +strerror(errno));

    lock_scheduler();
    if(pthread_create(&d_priv->threadid, NULL, run_threads, d_priv) != 0)
	throw ThreadError(std::string("pthread_create failed")
			  +strerror(errno));
    active[numActive]=d_priv;
    numActive++;
    unlock_scheduler();
}

void
Thread::stop()
{
    lock_scheduler();
    if(sem_trywait(&d_priv->block_sema) != 0){
	if(errno != EAGAIN)
	    throw ThreadError(std::string("sem_trywait: ")
			      +strerror(errno));
	if(this == self()) {
	    if(sem_wait(&d_priv->block_sema) != 0)
		throw ThreadError(std::string("sem_wait: ")
				  +strerror(errno));
	} else {
	    pthread_kill(d_priv->threadid, SIGUSR1);
	}
    }
    unlock_scheduler();
}

void
Thread::resume()
{
    lock_scheduler();
    if(sem_post(&d_priv->block_sema) != 0)
	throw ThreadError(std::string("sem_post: ")
			  +strerror(errno));
    unlock_scheduler();
}

void
Thread::detach()
{
    if(sem_post(&d_priv->delete_ready) != 0)
	throw ThreadError(std::string("sem_post failed")
			  +strerror(errno));
    d_detached=true;
    if(pthread_detach(d_priv->threadid) != 0)
	throw ThreadError(std::string("pthread_detach failed")
			  +strerror(errno));
}

void
Thread::exitAll(int code)
{
    exiting=true;
    ::exit(code);
}

/*
 * Handle an abort signal - like segv, bus error, etc.
 */
static
void
handle_abort_signals(int sig, struct sigcontext ctx)
{
    struct sigaction action;
    sigemptyset(&action.sa_mask);
    action.sa_handler=SIG_DFL;
    action.sa_flags=0;
    if(sigaction(sig, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction failed")
			  +strerror(errno));

    Thread* self=Thread::self();
    const char* tname=self?self->getThreadName():"idle or main";
    void* addr=(void*)ctx.cr2;
    char* signam=SCICore_Thread_signal_name(sig, addr);
    fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) caught signal %s\n", 7,7,7,tname, getpid(), signam);
    Thread::niceAbort();

    action.sa_handler=(SIG_HANDLER_T)handle_abort_signals;
    action.sa_flags=0;
    if(sigaction(sig, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction failed")
			  +strerror(errno));
}

int
Thread::get_tid()
{
    return d_priv->threadid;
}

void
Thread::print_threads()
{
    FILE* fp=stderr;
    for(int i=0;i<numActive;i++){
	Thread_private* p=active[i];
	const char* tname=p->thread?p->thread->getThreadName():"???";
	fprintf(fp, " %ld: %s (", p->threadid, tname);
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
}

/*
 * Handle sigquit - usually sent by control-C
 */
static
void
handle_quit(int sig, struct sigcontext ctx)
{
    // Try to acquire a lock.  If we can't, then assume that somebody
    // else already caught the signal...
    Thread* self=Thread::self();
    if(self==0)
	return; // This is an idle thread...
    if(sem_trywait(&control_c_sema) != 0){
	// This will wait until the other thread is done
	// handling the interrupt
	if(sem_wait(&control_c_sema) != 0)
	    throw ThreadError(std::string("sem_wait failed")
			      +strerror(errno));
	if(sem_post(&control_c_sema) != 0)
	    throw ThreadError(std::string("sem_post failed")
			      +strerror(errno));
	return;
    }
    // Otherwise, we got the semaphore and handle the interrupt
    const char* tname=self?self->getThreadName():"main?";

    // Kill all of the threads...
    char* signam=SCICore_Thread_signal_name(sig, 0);
    int pid=getpid();
    fprintf(stderr, "Thread \"%s\"(pid %d) caught signal %s\n", tname, pid, signam);
    Thread::niceAbort(); // Enter the monitor
    if(sem_post(&control_c_sema) != 0)
	throw ThreadError(std::string("sem_post failed")
			  +strerror(errno));
}

/*
 * Handle siguser1 - for stop/resume
 */
static
void
handle_siguser1(int)
{
    Thread* self=Thread::self();
    if(sem_wait(&self->d_priv->block_sema) != 0)
	throw ThreadError(std::string("sem_wait: ")
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

    action.sa_handler=(SIG_HANDLER_T)handle_abort_signals;
    if(sigaction(SIGILL, &action, NULL) == -1)
	throw ThreadError(std::string("SIGILL failed")
			  +strerror(errno));
    if(sigaction(SIGABRT, &action, NULL) == -1)
	throw ThreadError(std::string("SIGABRT failed")
			  +strerror(errno));
    if(sigaction(SIGTRAP, &action, NULL) == -1)
	throw ThreadError(std::string("SIGTRAP failed")
			  +strerror(errno));
    if(sigaction(SIGBUS, &action, NULL) == -1)
	throw ThreadError(std::string("SIGBUS failed")
			  +strerror(errno));
    if(sigaction(SIGSEGV, &action, NULL) == -1)
	throw ThreadError(std::string("SIGSEGV failed")
			  +strerror(errno));

    action.sa_handler=(SIG_HANDLER_T)handle_quit;
    if(sigaction(SIGQUIT, &action, NULL) == -1)
	throw ThreadError(std::string("SIGQUIT failed")
			  +strerror(errno));
    if(sigaction(SIGINT, &action, NULL) == -1)
	throw ThreadError(std::string("SIGINT failed")
			  +strerror(errno));

    action.sa_handler=(SIG_HANDLER_T)handle_siguser1;
    if(sigaction(SIGQUIT, &action, NULL) == -1)
	throw ThreadError(std::string("SIGUSR1 failed")
			  +strerror(errno));
}

static void exit_handler()
{
    if(exiting)
        return;
    // Wait forever...
    sem_t wait;
    if(sem_init(&wait, 0, 0) != 0)
	throw ThreadError(std::string("sem_init failed")
			  +strerror(errno));
    if(sem_wait(&wait) == -1)
	throw ThreadError(std::string("sem_wait failed")
			  +strerror(errno));
}

void
Thread::initialize()
{
  atexit(exit_handler);
    if(pthread_mutex_init(&sched_lock, NULL) != 0)
	throw ThreadError(std::string("pthread_mutex_init failed")
			  +strerror(errno));

    if(pthread_key_create(&thread_key, NULL) != 0)
	throw ThreadError(std::string("pthread_key_create failed")
			  +strerror(errno));

    ThreadGroup::s_default_group=new ThreadGroup("default group", 0);
    Thread* mainthread=new Thread(ThreadGroup::s_default_group, "main");
    mainthread->d_priv=new Thread_private;
    mainthread->d_priv->thread=mainthread;
    mainthread->d_priv->state=RUNNING;
    mainthread->d_priv->bstacksize=0;
    if(pthread_setspecific(thread_key, mainthread) != 0)
	throw ThreadError(std::string("pthread_setspecific failed")
			  +strerror(errno));
    if(sem_init(&mainthread->d_priv->done, 0, 0) != 0)
	throw ThreadError(std::string("sem_init failed")
			  +strerror(errno));
    if(sem_init(&mainthread->d_priv->delete_ready, 0, 0) != 0)
	throw ThreadError(std::string("sem_init failed")
			  +strerror(errno));
    if(sem_init(&main_sema, 0, 0) != 0)
	throw ThreadError(std::string("sem_init failed")
			  +strerror(errno));
    if(sem_init(&control_c_sema, 0, 1) != 0)
	throw ThreadError(std::string("sem_init failed")
			  +strerror(errno));
    lock_scheduler();
    active[numActive]=mainthread->d_priv;
    numActive++;
    unlock_scheduler();
    if(!getenv("THREAD_NO_CATCH_SIGNALS"))
	install_signal_handlers();

    initialized=true;
}

void
Thread::yield()
{
    sched_yield();
}

void
Thread::migrate(int proc)
{
    // Nothing for now...
}

namespace SCICore {
    namespace Thread {
	struct Mutex_private {
	    pthread_mutex_t mutex;
	};
    }
}

Mutex::Mutex(const char* name)
    : d_name(name)
{
    d_priv=new Mutex_private;
    if(pthread_mutex_init(&d_priv->mutex, NULL) != 0)
	throw ThreadError(std::string("pthread_mutex_init: ")
			  +strerror(errno));
}

Mutex::~Mutex()
{
    if(pthread_mutex_destroy(&d_priv->mutex) != 0)
	throw ThreadError(std::string("pthread_mutex_destroy: ")
			  +strerror(errno));
    delete d_priv;
}

void
Mutex::unlock()
{
    if(pthread_mutex_unlock(&d_priv->mutex) != 0)
	throw ThreadError(std::string("pthread_mutex_unlock: ")
			  +strerror(errno));
}

void
Mutex::lock()
{
    Thread* t=Thread::self();
    int oldstate=-1;
    Thread_private* p=0;
    if(t){
	p=t->d_priv;
	oldstate=Thread::push_bstack(p, Thread::BLOCK_MUTEX, d_name);
    }
    if(pthread_mutex_lock(&d_priv->mutex) != 0)
	throw ThreadError(std::string("pthread_mutex_lock: ")
			  +strerror(errno));
    if(t)
	Thread::pop_bstack(p, oldstate);
}

bool
Mutex::tryLock()
{
    if(pthread_mutex_trylock(&d_priv->mutex) != 0){
	if(errno == EAGAIN || errno == EINTR)
	    return false;
	throw ThreadError(std::string("pthread_mutex_trylock: ")
			  +strerror(errno));
    }
    return true;
}

namespace SCICore {
    namespace Thread {
	struct RecursiveMutex_private {
	    pthread_mutex_t mutex;
	};
    }
}

RecursiveMutex::RecursiveMutex(const char* name)
    : d_name(name)
{
    d_priv=new RecursiveMutex_private;
    pthread_mutexattr_t attr;
    if(pthread_mutexattr_init(&attr) != 0)
	throw ThreadError(std::string("pthread_mutexattr_init: ")
			  +strerror(errno));
    if(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE_NP) != 0)
	throw ThreadError(std::string("pthread_mutexattr_setkind_np: ")
			  +strerror(errno));
    if(pthread_mutex_init(&d_priv->mutex, &attr) != 0)
	throw ThreadError(std::string("pthread_mutex_init: ")
			  +strerror(errno));
    if(pthread_mutexattr_destroy(&attr) != 0)
	throw ThreadError(std::string("pthread_mutexattr_destroy: ")
			  +strerror(errno));
}

RecursiveMutex::~RecursiveMutex()
{
    if(pthread_mutex_destroy(&d_priv->mutex) != 0)
	throw ThreadError(std::string("pthread_mutex_destroy: ")
			  +strerror(errno));
    delete d_priv;
}

void
RecursiveMutex::unlock()
{
    if(pthread_mutex_unlock(&d_priv->mutex) != 0)
	throw ThreadError(std::string("pthread_mutex_unlock: ")
			  +strerror(errno));
}

void
RecursiveMutex::lock()
{
    Thread_private* p=Thread::self()->d_priv;
    int oldstate=Thread::push_bstack(p, Thread::BLOCK_ANY, d_name);
    if(pthread_mutex_lock(&d_priv->mutex) != 0)
	throw ThreadError(std::string("pthread_mutex_lock: ")
			  +strerror(errno));
    Thread::pop_bstack(p, oldstate);
}

namespace SCICore {
    namespace Thread {
	struct Semaphore_private {
	    sem_t sem;
	};
    }
}

Semaphore::Semaphore(const char* name, int value)
    : d_name(name)
{
    d_priv=new Semaphore_private;
    if(sem_init(&d_priv->sem, 0, value) != 0)
	throw ThreadError(std::string("sem_init: ")
			  +strerror(errno));
}
    
Semaphore::~Semaphore()
{
    if(sem_destroy(&d_priv->sem) != 0)
	throw ThreadError(std::string("sem_destroy: ")
			  +strerror(errno));

    delete d_priv;
}

void
Semaphore::up(int count)
{
    for(int i=0;i<count;i++){
	if(sem_post(&d_priv->sem) != 0)
	    throw ThreadError(std::string("sem_post: ")
			      +strerror(errno));
    }
}

void
Semaphore::down(int count)
{
    Thread_private* p=Thread::self()->d_priv;
    int oldstate=Thread::push_bstack(p, Thread::BLOCK_SEMAPHORE, d_name);
    for(int i=0;i<count;i++){
	if(sem_wait(&d_priv->sem) != 0)
	    throw ThreadError(std::string("sem_wait: ")
			      +strerror(errno));
    }
    Thread::pop_bstack(p, oldstate);
}

bool
Semaphore::tryDown()
{
    if(sem_trywait(&d_priv->sem) != 0){
	if(errno == EAGAIN)
	    return false;
	throw ThreadError(std::string("sem_trywait: ")
			  +strerror(errno));
    }
    return true;
}

namespace SCICore {
    namespace Thread {
	struct ConditionVariable_private {
	    pthread_cond_t cond;
	};
    }
}

ConditionVariable::ConditionVariable(const char* name)
    : d_name(name)
{
    d_priv=new ConditionVariable_private;
    if(pthread_cond_init(&d_priv->cond, 0) != 0)
	throw ThreadError(std::string("pthread_cond_init: ")
			  +strerror(errno));
}

ConditionVariable::~ConditionVariable()
{
    if(pthread_cond_destroy(&d_priv->cond) != 0)
	throw ThreadError(std::string("pthread_cond_destroy: ")
			  +strerror(errno));
}

void
ConditionVariable::wait(Mutex& m)
{
    Thread_private* p=Thread::self()->d_priv;
    int oldstate=Thread::push_bstack(p, Thread::BLOCK_ANY, d_name);
    if(pthread_cond_wait(&d_priv->cond, &m.d_priv->mutex) != 0)
	throw ThreadError(std::string("pthread_cond_wait: ")
			  +strerror(errno));
    Thread::pop_bstack(p, oldstate);
}

void
ConditionVariable::conditionSignal()
{
    if(pthread_cond_signal(&d_priv->cond) != 0)
	throw ThreadError(std::string("pthread_cond_signal: ")
			  +strerror(errno));
}
void
ConditionVariable::conditionBroadcast()
{
    if(pthread_cond_broadcast(&d_priv->cond) != 0)
	throw ThreadError(std::string("pthread_cond_broadcast: ")
			  +strerror(errno));
}

//
// $Log$
// Revision 1.8  1999/08/29 08:07:39  sparker
// Fixed bug on linux where main exit before other threads are done.
//
// Revision 1.7  1999/08/29 07:50:59  sparker
// Mods to compile on linux
//
// Revision 1.6  1999/08/29 00:47:02  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/28 03:46:52  sparker
// Final updates before integration with PSE
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
