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
 *  Thread_pthreads.cc: Posix threads implementation of the thread library
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_UNIX98
#include <pthread.h>
#ifndef PTHREAD_MUTEX_RECURSIVE
#define PTHREAD_MUTEX_RECURSIVE PTHREAD_MUTEX_RECURSIVE_NP
#endif
#define private public
#define protected public
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h> // So ConditionVariable can get to Mutex::priv_
#undef private
#undef protected
#include <Core/Thread/Thread.h>
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/RecursiveMutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ThreadError.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/WorkQueue.h>
#include "Thread_unix.h"
#include <errno.h>
extern "C" {
#include <semaphore.h>
}
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


using SCIRun::ConditionVariable;
using SCIRun::Mutex;
using SCIRun::RecursiveMutex;
using SCIRun::Semaphore;
using SCIRun::Thread;
using SCIRun::ThreadError;
using SCIRun::ThreadGroup;

bool exiting=false;

#define MAXBSTACK 10
#define MAXTHREADS 4000


namespace SCIRun {
struct Thread_private {
  Thread* thread;
  pthread_t threadid;
  Thread::ThreadState state;
  int bstacksize;
  const char* blockstack[MAXBSTACK];
  sem_t done;
  sem_t delete_ready;
  sem_t block_sema;
  bool is_blocked;
  bool ismain;
};
}

static const char* bstack_init="Unused block stack entry";

using SCIRun::Thread_private;

static Thread_private* active[MAXTHREADS];
static int numActive;
static pthread_mutex_t sched_lock;
static pthread_key_t thread_key;
static sem_t main_sema;
static sem_t control_c_sema;
static Thread* mainthread;

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
   if(p->bstacksize>=MAXBSTACK){
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
   Thread_private* priv=thread->priv_;

   if(sem_post(&priv->done) != 0)
      throw ThreadError(std::string("sem_post failed")
			+strerror(errno));

   // Wait to be deleted...
   if (!priv->ismain)
      if(sem_wait(&priv->delete_ready) == -1)
	 throw ThreadError(std::string("sem_wait failed")
			   +strerror(errno));

   // Allow this thread to run anywhere...
   if(thread->cpu_ != -1)
      thread->migrate(-1);

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
   if(pthread_setspecific(thread_key, 0) != 0)
     fprintf(stderr, "Warning: pthread_setspecific failed");
   unlock_scheduler();
   bool wait_main = priv->ismain;
   delete thread;
   priv->thread=0;
   delete priv;
   Thread::checkExit();
   if(wait_main){
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
    int os=push_bstack(us->priv_, JOINING, threadname_);
    if(sem_wait(&priv_->done) != 0)
	throw ThreadError(std::string("sem_wait failed")
			  +strerror(errno));
    pop_bstack(us->priv_, os);
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
  int err;
  if((err=pthread_setspecific(thread_key, priv->thread)) != 0){
    fprintf(stderr, "errno=%d, key=%d\n", err, thread_key);
    throw ThreadError(std::string("pthread_setspecific failed")
		      +strerror(errno));
  }
  priv->is_blocked=true;
  if(sem_wait(&priv->block_sema) != 0)
    throw ThreadError(std::string("sem_wait: ")
		      +strerror(errno));
  priv->is_blocked=false;
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

  priv_=new Thread_private;

  if(sem_init(&priv_->done, 0, 0) != 0)
    throw ThreadError(std::string("sem_init failed")
		      +strerror(errno));
  if(sem_init(&priv_->delete_ready, 0, 0) != 0)
    throw ThreadError(std::string("sem_init failed")
		      +strerror(errno));
  priv_->state=STARTUP;
  priv_->bstacksize=0;
  for(int i=0;i<MAXBSTACK;i++)
    priv_->blockstack[i]=bstack_init;
    
  priv_->thread=this;
  priv_->threadid=0;
  priv_->is_blocked=false;
  priv_->ismain=false;

  if(sem_init(&priv_->block_sema, 0, stopped?0:1))
    throw ThreadError(std::string("sem_init failed")
		      +strerror(errno));

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, stacksize_);
		
  lock_scheduler();
  active[numActive]=priv_;
  numActive++;
  if(pthread_create(&priv_->threadid, &attr, run_threads, priv_) != 0)
    throw ThreadError(std::string("pthread_create failed")
		      +strerror(errno));
  unlock_scheduler();
}

void
Thread::stop()
{
    lock_scheduler();
    if(sem_trywait(&priv_->block_sema) != 0){
	if(errno != EAGAIN)
	    throw ThreadError(std::string("sem_trywait: ")
			      +strerror(errno));
	if(this == self()) {
	    if(sem_wait(&priv_->block_sema) != 0)
		throw ThreadError(std::string("sem_wait: ")
				  +strerror(errno));
	} else {
	    pthread_kill(priv_->threadid, SIGUSR2);
	}
    }
    unlock_scheduler();
}

void
Thread::resume()
{
    lock_scheduler();
    if(sem_post(&priv_->block_sema) != 0)
	throw ThreadError(std::string("sem_post: ")
			  +strerror(errno));
    unlock_scheduler();
}

void
Thread::detach()
{
    detached_=true;
    pthread_t id = priv_->threadid;
    if(sem_post(&priv_->delete_ready) != 0)
	throw ThreadError(std::string("sem_post failed")
			  +strerror(errno));
    if(pthread_detach(id) != 0)
	throw ThreadError(std::string("pthread_detach failed")
			  +strerror(errno));
}

void
Thread::exitAll(int code)
{
  if(initialized && !exiting){
    exiting=true;
    lock_scheduler();
    if(initialized){
      // Stop all of the other threads before we die, because
      // global destructors may destroy primitives that other
      // threads are using...
      Thread* me = Thread::self();
      for(int i=0;i<numActive;i++){
	Thread_private* t = active[i];
	if(t->thread != me){
	  pthread_kill(t->threadid, SIGUSR2);
	}
      }
      // Wait for all threads to be in the signal handler
      int numtries=1000000;
      bool done=false;
      while(--numtries && !done){
	done=true;
	for(int i=0;i<numActive;i++){
	  Thread_private* t = active[i];
	  if(t->thread != me){
	    if(!t->is_blocked)
	      done=false;
	  }
	}
	sched_yield();
      }
      if(!numtries){
	for(int i=0;i<numActive;i++){
	  Thread_private* t = active[i];
	  if(t->thread != me && !t->is_blocked)
	    fprintf(stderr, "Thread: %s is slow to stop, giving up\n", t->thread->getThreadName());
	}
      }
      if(sem_destroy(&main_sema) != 0)
	throw ThreadError(std::string("sem_destroy failed")
			  +strerror(errno));
      unlock_scheduler();
    }
    ::exit(code);
  }
}

/*
 * Handle an abort signal - like segv, bus error, etc.
 */
static
void
handle_abort_signals(int sig,
#ifdef __sgi
		     int,
		     struct sigcontext* ctx)
#else
		     struct sigcontext ctx)
#endif
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
#ifdef __sgi
#if defined(_LONGLONG)
  caddr_t addr=(caddr_t)ctx->sc_badvaddr;
#else
  caddr_t addr=(caddr_t)ctx->sc_badvaddr.lo32;
#endif
#else
#ifdef PPC
    void* addr=(void*)ctx.regs->dsisr;
#else
    void* addr=(void*)ctx.cr2;
#endif
#endif
    char* signam=Core_Thread_signal_name(sig, addr);
    fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) caught signal %s\n", 7,7,7,tname, getpid(), signam);
    Thread::niceAbort();

    action.sa_handler=(SIG_HANDLER_T)handle_abort_signals;
    action.sa_flags=0;
    if(sigaction(sig, &action, NULL) == -1)
	throw ThreadError(std::string("sigaction failed")
			  +strerror(errno));
}

int
Thread::get_thread_id()
{
    return priv_->threadid;
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
handle_quit(int sig, struct sigcontext /*ctx*/)
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
    char* signam=Core_Thread_signal_name(sig, 0);
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
  if(!self){
    // This can happen if the thread is just started and hasn't had
    // the opportunity to call setspecific for the thread id yet
    for(int i=0;i<numActive;i++)
      if(pthread_self() == active[i]->threadid)
	self=active[i]->thread;
  }
  self->priv_->is_blocked=true;
  if(sem_wait(&self->priv_->block_sema) != 0)
    throw ThreadError(std::string("sem_wait: ")
		      +strerror(errno));
  self->priv_->is_blocked=false;
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
    if(sigaction(SIGUSR2, &action, NULL) == -1)
	throw ThreadError(std::string("SIGUSR2 failed")
			  +strerror(errno));
}

static void exit_handler()
{
    if(exiting)
        return;
    Thread_shutdown(Thread::self());
}

void
Thread::allow_sgi_OpenGL_page0_sillyness()
{
  // Nothing necessary here
}

void
Thread::disallow_sgi_OpenGL_page0_sillyness()
{
  // Nothing necessary here
}

void
Thread::initialize()
{
  if(initialized)
    return;
  if(exiting)
    abort(); // Something really weird happened!
  atexit(exit_handler);
  if(pthread_mutex_init(&sched_lock, NULL) != 0)
    throw ThreadError(std::string("pthread_mutex_init failed")
		      +strerror(errno));
  
  if(pthread_key_create(&thread_key, NULL) != 0)
    throw ThreadError(std::string("pthread_key_create failed")
		      +strerror(errno));

  initialized=true;
  ThreadGroup::s_default_group=new ThreadGroup("default group", 0);
  mainthread=new Thread(ThreadGroup::s_default_group, "main");
  mainthread->priv_=new Thread_private;
  mainthread->priv_->thread=mainthread;
  mainthread->priv_->state=RUNNING;
  mainthread->priv_->bstacksize=0;
  mainthread->priv_->is_blocked=false;
  mainthread->priv_->threadid=pthread_self();
  mainthread->priv_->ismain=true;
  for(int i=0;i<MAXBSTACK;i++)
      mainthread->priv_->blockstack[i]=bstack_init;
  if(pthread_setspecific(thread_key, mainthread) != 0)
    throw ThreadError(std::string("pthread_setspecific failed")
		      +strerror(errno));
  if(sem_init(&mainthread->priv_->done, 0, 0) != 0)
    throw ThreadError(std::string("sem_init failed")
		      +strerror(errno));
  if(sem_init(&mainthread->priv_->delete_ready, 0, 0) != 0)
    throw ThreadError(std::string("sem_init failed")
		      +strerror(errno));
  if(sem_init(&main_sema, 0, 0) != 0)
    throw ThreadError(std::string("sem_init failed")
		      +strerror(errno));
  if(sem_init(&control_c_sema, 0, 1) != 0)
    throw ThreadError(std::string("sem_init failed")
		      +strerror(errno));
  lock_scheduler();
  active[numActive]=mainthread->priv_;
  numActive++;
  unlock_scheduler();
  if(!getenv("THREAD_NO_CATCH_SIGNALS"))
    install_signal_handlers();
}

void
Thread::yield()
{
    sched_yield();
}

void
Thread::migrate(int /*proc*/)
{
    // Nothing for now...
}

namespace SCIRun {
struct Mutex_private {
  pthread_mutex_t mutex;
};
}

Mutex::Mutex(const char* name)
    : name_(name)
{
  // DO NOT CALL INITIALIZE in this CTOR!
  if(this == 0){
    fprintf(stderr, "WARNING: creation of null mutex\n");
  }
  
  priv_=new Mutex_private;
  if(pthread_mutex_init(&priv_->mutex, NULL) != 0)
    throw ThreadError(std::string("pthread_mutex_init: ")
		      +strerror(errno));		
}

Mutex::~Mutex()
{
  pthread_mutex_unlock(&priv_->mutex);
  if(pthread_mutex_destroy(&priv_->mutex) != 0) {
    fprintf(stderr, "pthread_mutex_destroy() failed!!\n");
    throw ThreadError(std::string("pthread_mutex_destroy: ")
		      +strerror(errno));
  }
  delete priv_;
  priv_=0;
}

void
Mutex::unlock()
{
   int status = pthread_mutex_unlock(&priv_->mutex);
   if(status != 0){
      fprintf(stderr, "unlock failed, status=%d (%s)\n", status, strerror(status));
      throw ThreadError(std::string("pthread_mutex_unlock: ")
			+strerror(status));
   }
}

void
Mutex::lock()
{
  Thread* t=Thread::isInitialized()?Thread::self():0;
  int oldstate=-1;
  Thread_private* p=0;
  if(t){
    p=t->priv_;
    oldstate=Thread::push_bstack(p, Thread::BLOCK_MUTEX, name_);
  }
  int status = pthread_mutex_lock(&priv_->mutex);
  if(status != 0){
    fprintf(stderr, "lock failed, status=%d (%s)\n", status, strerror(status));
    throw ThreadError(std::string("pthread_mutex_lock: ")
		      +strerror(status));
  }
		
  if(t)
    Thread::pop_bstack(p, oldstate);
}

bool
Mutex::tryLock()
{
    if(pthread_mutex_trylock(&priv_->mutex) != 0){
	if(errno == EAGAIN || errno == EINTR)
	    return false;
	throw ThreadError(std::string("pthread_mutex_trylock: ")
			  +strerror(errno));
    }
    return true;
}

namespace SCIRun {
struct RecursiveMutex_private {
  pthread_mutex_t mutex;
};
}

RecursiveMutex::RecursiveMutex(const char* name)
    : name_(name)
{
   if(!Thread::initialized)
	Thread::initialize();
    priv_=new RecursiveMutex_private;
    pthread_mutexattr_t attr;
    if(pthread_mutexattr_init(&attr) != 0)
	throw ThreadError(std::string("pthread_mutexattr_init: ")
			  +strerror(errno));
    if(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE) != 0)
	throw ThreadError(std::string("pthread_mutexattr_settype: ")
			  +strerror(errno));
    if(pthread_mutex_init(&priv_->mutex, &attr) != 0)
	throw ThreadError(std::string("pthread_mutex_init: ")
			  +strerror(errno));
    if(pthread_mutexattr_destroy(&attr) != 0)
	throw ThreadError(std::string("pthread_mutexattr_destroy: ")
			  +strerror(errno));
}

RecursiveMutex::~RecursiveMutex()
{
  pthread_mutex_unlock(&priv_->mutex);
  if(pthread_mutex_destroy(&priv_->mutex) != 0)
      throw ThreadError(std::string("pthread_mutex_destroy: ")
			+strerror(errno));
  delete priv_;
  priv_=0;
}

void
RecursiveMutex::unlock()
{
    if(pthread_mutex_unlock(&priv_->mutex) != 0)
	throw ThreadError(std::string("pthread_mutex_unlock: ")
			  +strerror(errno));
}

void
RecursiveMutex::lock()
{
    Thread_private* p=Thread::self()->priv_;
    int oldstate=Thread::push_bstack(p, Thread::BLOCK_ANY, name_);
    if(pthread_mutex_lock(&priv_->mutex) != 0)
	throw ThreadError(std::string("pthread_mutex_lock: ")
			  +strerror(errno));
    Thread::pop_bstack(p, oldstate);
}

namespace SCIRun {
struct Semaphore_private {
  sem_t sem;
};
}

Semaphore::Semaphore(const char* name, int value)
    : name_(name)
{
  if(!Thread::initialized)
    Thread::initialize();    
  priv_=new Semaphore_private;
  if(sem_init(&priv_->sem, 0, value) != 0)
    throw ThreadError(std::string("sem_init: ")
		      +strerror(errno));
}
    
Semaphore::~Semaphore()
{
  int val;
  sem_getvalue(&priv_->sem,&val);
  while(val<=0) {
    sem_post(&priv_->sem);
    sem_getvalue(&priv_->sem,&val);
  }
  if(sem_destroy(&priv_->sem) != 0)
    throw ThreadError(std::string("sem_destroy: ")
		      +strerror(errno));
  
  delete priv_;
  priv_=0;
}

void
Semaphore::up(int count)
{
    for(int i=0;i<count;i++){
	if(sem_post(&priv_->sem) != 0)
	    throw ThreadError(std::string("sem_post: ")
			      +strerror(errno));
    }
}

void
Semaphore::down(int count)
{
    Thread_private* p=Thread::self()->priv_;
    int oldstate=Thread::push_bstack(p, Thread::BLOCK_SEMAPHORE, name_);
    for(int i=0;i<count;i++){
	if(sem_wait(&priv_->sem) != 0)
	    throw ThreadError(std::string("sem_wait: ")
			      +strerror(errno));
    }
    Thread::pop_bstack(p, oldstate);
}

bool
Semaphore::tryDown()
{
    if(sem_trywait(&priv_->sem) != 0){
	if(errno == EAGAIN)
	    return false;
	throw ThreadError(std::string("sem_trywait: ")
			  +strerror(errno));
    }
    return true;
}

namespace SCIRun {
struct ConditionVariable_private {
  pthread_cond_t cond;
};
}

ConditionVariable::ConditionVariable(const char* name)
    : name_(name)
{
  if(!Thread::initialized)
    Thread::initialize();
  priv_=new ConditionVariable_private;
  if(pthread_cond_init(&priv_->cond, 0) != 0)
    throw ThreadError(std::string("pthread_cond_init: ")
		      +strerror(errno));
}

ConditionVariable::~ConditionVariable()
{
  if(pthread_cond_destroy(&priv_->cond) != 0)
    throw ThreadError(std::string("pthread_cond_destroy: ")
		      +strerror(errno));
  delete priv_;
  priv_=0;
}

void
ConditionVariable::wait(Mutex& m)
{
    Thread_private* p=Thread::self()->priv_;
    int oldstate=Thread::push_bstack(p, Thread::BLOCK_ANY, name_);
    if(pthread_cond_wait(&priv_->cond, &m.priv_->mutex) != 0)
	throw ThreadError(std::string("pthread_cond_wait: ")
			  +strerror(errno));
    Thread::pop_bstack(p, oldstate);
}

bool
ConditionVariable::timedWait(Mutex& m, const struct timespec* abstime)
{
  Thread_private* p=Thread::self()->priv_;
  int oldstate=Thread::push_bstack(p, Thread::BLOCK_ANY, name_);
  bool success;
  if(abstime){
    int err=pthread_cond_timedwait(&priv_->cond, &m.priv_->mutex,
				   abstime);
    if(err != 0){
      if(err == ETIMEDOUT)
	success=false;
      else
	throw ThreadError(std::string("pthread_cond_timedwait: ")
			  +strerror(errno));
    } else {
      success=true;
    }
  } else {
    if(pthread_cond_wait(&priv_->cond, &m.priv_->mutex) != 0)
      throw ThreadError(std::string("pthread_cond_wait: ")
			+strerror(errno));
    success=true;
  }
  Thread::pop_bstack(p, oldstate);
  return success;
}

void
ConditionVariable::conditionSignal()
{
  if(pthread_cond_signal(&priv_->cond) != 0)
    throw ThreadError(std::string("pthread_cond_signal: ")
		      +strerror(errno));
}

void
ConditionVariable::conditionBroadcast()
{
  if(pthread_cond_broadcast(&priv_->cond) != 0)
    throw ThreadError(std::string("pthread_cond_broadcast: ")
		      +strerror(errno));
}
