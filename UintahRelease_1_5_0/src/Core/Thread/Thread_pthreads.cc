/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
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

#include <sci_defs/bits_defs.h>

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#define __USE_UNIX98
#include <pthread.h>
#ifndef PTHREAD_MUTEX_RECURSIVE
#  define PTHREAD_MUTEX_RECURSIVE PTHREAD_MUTEX_RECURSIVE_NP
#endif

#include <Core/Exceptions/Exception.h>
#include <iostream>

//////////////////////////////////////////////////////
// begin: Danger Will Robinson! Danger Will Robinson!

#define private public
#define protected public

#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h> // So ConditionVariable can get to Mutex::priv_

#undef private
#undef protected

// end: Danger Will Robinson! Danger Will Robinson!
//////////////////////////////////////////////////////


#include <Core/Thread/Thread.h>
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/CleanupManager.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/RecursiveMutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ThreadError.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/WorkQueue.h>
#include <Core/Thread/Thread_unix.h>
#include <Core/Util/Assert.h>
#include <Core/Thread/CrashPad.h>

#include <cerrno>
extern "C" {
#  include <semaphore.h>
}
#include <signal.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <string>
#ifdef __APPLE__
#  include <sys/types.h>
#  include <sys/sysctl.h>
#endif

#ifdef __APPLE__
  #ifdef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
    #if __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 1050
      #define SigContext ucontext_t
    #else
      #define SigContext struct sigcontext
    #endif
  #else
    #define SigContext struct sigcontext
  #endif
#else
  #define SigContext struct sigcontext
#endif

#include <TauProfilerForSCIRun.h>

//#define __ia64__
#ifdef __ia64__
#  ifndef __int64
#    define __int64 long
#  endif
#  include <ia64intrin.h>
#endif

#if defined(_AIX)

#  include <sys/mman.h>
#  define sem_type msemaphore *
#  define SEM_UNLOCK(sem)            msem_unlock(*(sem),0)
#  define SEM_LOCK(sem)              msem_lock(*(sem),0)
#  define SEM_TRYLOCK(sem)           msem_lock(*(sem), MSEM_IF_NOWAIT)
#  define SEM_INIT(sem, shared, val) msem_init(*(sem), \
                                         ((val)==0)?MSEM_UNLOCKED:MSEM_LOCKED)
#  define SEM_INIT_SUCCESS(val)      (((val)!= 0)?true:false)
#  define SEM_DESTROY(sem)           msem_remove(*(sem))

#else

#  define sem_type sem_t
#  define SEM_UNLOCK(sem)            sem_post((sem))
#  define SEM_TRYLOCK(sem)           sem_trywait((sem))
#  define SEM_INIT(sem, shared, val) sem_init( (sem), (shared), (val) )
#  define SEM_INIT_SUCCESS(val)      (((val)== 0)?true:false)
#  define SEM_DESTROY(sem)           sem_destroy((sem))

// NOTE(boulos): This code is not currently used if __APPLE__ is
// defined, so defining this function produces a "defined but not used
// warning"
#  ifndef __APPLE__
static int SEM_LOCK(sem_type* sem)
{
  int returnValue = 0;
  while ( (returnValue = sem_wait(sem)) == -1 && (errno == EINTR) );
  return returnValue;
}
#  endif

#endif

typedef void (*SIG_HANDLER_T)(int);

/*
 * The pthread implementation uses the default version of AtomicCounter,
 * Barrier, and CrowdMonitor.  It provides native implementations of
 * of ConditionVariable, Mutex, RecursiveMutex and Semaphore.
 *
 */

#ifndef __ia64__
#  include <Core/Thread/Barrier_default.cc>
#endif

#include <Core/Thread/CrowdMonitor_pthreads.cc>

using SCIRun::ConditionVariable;
using SCIRun::Mutex;
using SCIRun::RecursiveMutex;
using SCIRun::Semaphore;
using SCIRun::Thread;
using SCIRun::ThreadError;
using SCIRun::ThreadGroup;

static bool exiting = false;

#define MAXBSTACK 10

#if defined(_AIX) && defined(MAXTHREADS)
#  undef MAXTHREADS
#endif

#define MAXTHREADS 4000

namespace SCIRun {
struct Thread_private {
  Thread_private(bool stopped);

  Thread* thread;
  pthread_t threadid;
  Thread::ThreadState state;
  int bstacksize;
  const char* blockstack[MAXBSTACK];
  Semaphore done;
  Semaphore delete_ready;
  Semaphore block_sema;
  bool is_blocked;
  bool ismain;
};
} // end namespace SCIRun

static const char* bstack_init = "Unused block stack entry";

using SCIRun::Thread_private;

static Thread_private* active[MAXTHREADS];
static int numActive = 0;

static pthread_mutex_t sched_lock;
static pthread_key_t   thread_key;

static Semaphore main_sema("main",0);
static Semaphore control_c_sema("control-c",1);
static Thread*  mainthread = 0;


Thread_private::Thread_private(bool stopped) :
  done("done",0),
  delete_ready("delete_ready",0),
  block_sema("block_sema",stopped?0:1)
{
}


static
void
lock_scheduler()
{
  const int status = pthread_mutex_lock(&sched_lock);
  if (status)
  {
    switch (status)
    {
    case EINVAL:
      throw ThreadError("pthread_mutex_lock:  Uninitialized lock.");
      break;

    case EDEADLK:
      throw ThreadError("pthread_mutex_lock:  Calling thread already holds this lock.");
      break;

    default:
      throw ThreadError("pthread_mutex_lock:  Unknown error.");
    }
  }
}


static
void
unlock_scheduler()
{
  const int status = pthread_mutex_unlock(&sched_lock);
  if (status)
  {
    switch (status)
    {
    case EINVAL:
      throw ThreadError("pthread_mutex_unlock:  Uninitialized lock.");
      break;

    case EPERM:
      throw ThreadError("pthread_mutex_unlock:  Unlocker did not lock.");
      break;

    default:
      throw ThreadError("pthread_mutex_unlock:  Unknown error.");
    }
  }
}

int
Thread::push_bstack(Thread_private* p, Thread::ThreadState state,
                    const char* name)
{
  int oldstate = p->state;
  p->state = state;
  p->blockstack[p->bstacksize]=name;
  p->bstacksize++;
  if (p->bstacksize>=MAXBSTACK){
    fprintf(stderr, "Blockstack Overflow!\n");
    Thread::niceAbort();
  }
  return oldstate;
}


void
Thread::pop_bstack(Thread_private* p, int oldstate)
{
  p->bstacksize--;
  p->state = (ThreadState)oldstate;
}

void
Thread::set_affinity(int cpu)
{
#ifndef __APPLE__
  //disable affinity on OSX since sched_setaffinity() is not avaible in OSX api
  cpu_set_t mask;
  unsigned int len = sizeof(mask);
  CPU_ZERO(&mask);
  CPU_SET(cpu,&mask);
  sched_setaffinity(0, len, &mask);
#endif
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
  Thread_private* priv = thread->priv_;

  priv->done.up();
  if (!priv->ismain) priv->delete_ready.down();

  // Allow this thread to run anywhere...
  if (thread->cpu_ != -1) {
    thread->migrate(-1);
  }

  lock_scheduler();
  /* Remove it from the active queue */
  int i;
  for (i = 0;i<numActive;i++){
    if (active[i]==priv)
      break;
  }
  for (i++;i<numActive;i++){
    active[i-1]=active[i];
  }
  numActive--;

  // This can't be done in checkExit, because of a potential race
  // condition.
  int done = true;
  for (int i = 0;i<numActive;i++){
    Thread_private* p = active[i];
    if (!p->thread->isDaemon()){
      done = false;
      break;
    }
  }

  thread->handleCleanup();

  unlock_scheduler();

  bool wait_main = priv->ismain;
  delete thread;
  if (pthread_setspecific(thread_key, 0) != 0) {
    fprintf(stderr, "Warning: pthread_setspecific failed");
  }
  priv->thread = 0;
  delete priv;
  if (done) {
    Thread::exitAll(0);
  }
  if (wait_main) {
    main_sema.down();
  }
  pthread_exit(0);
}


void
Thread::exit()
{
  Thread * self = Thread::self();
  Thread_shutdown(self);
}

void
Thread::checkExit()
{
  lock_scheduler();
  int done = true;
  for (int i = 0;i<numActive;i++)
  {
    Thread_private* p = active[i];
    if (!p->thread->isDaemon())
    {
      done = false;
      break;
    }
  }
  unlock_scheduler();

  if (done)
    Thread::exitAll(0);
}


Thread*
Thread::self()
{
  void* p = pthread_getspecific(thread_key);
  return (Thread*)p;
}


void
Thread::join()
{
  pthread_t id = priv_->threadid;

  priv_->delete_ready.up();

  const int status = pthread_join(id, 0);
  if (status)
  {
    switch (status)
    {
    case ESRCH:
      throw ThreadError("pthread_join:  No such thread.");
      break;

    case EINVAL:
      throw ThreadError("pthread_join:  Joining detached thread or joining same thread twice.");
      break;

    case EDEADLK:
      throw ThreadError("pthread_join:  Joining self, deadlock.");
      break;

    default:
      throw ThreadError("pthread_join:  Unknown error.");
    }
  }
}

int
Thread::numProcessors()
{
  static int np = 0;

  if (np == 0) {
#ifdef __APPLE__
    size_t len = sizeof(np);
    sysctl((int[2]) {CTL_HW, HW_NCPU}, 2, &np, &len, NULL, 0);
#else
    // Linux
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo) {
      int count = 0;
      while (!cpuinfo.eof()) {
        std::string str;
        cpuinfo >> str;
        if (str == "processor") {
          ++count;
        }
      }
      np = count;
    }
#endif
    if (np <= 0) np = 1;
  }
  return np;
}


static void
Thread_run(Thread* t)
{
  t->run_body();
}


static
void*
run_threads(void* priv_v)
{
  TAU_REGISTER_THREAD();

  Thread_private* priv = (Thread_private*)priv_v;
  if (pthread_setspecific(thread_key, priv->thread)) {
    throw ThreadError("pthread_setspecific: Unknown error.");
  }
  priv->is_blocked = true;

  priv->block_sema.down();
  priv->is_blocked = false;
  priv->state = Thread::RUNNING;
  Thread_run(priv->thread);
  priv->state = Thread::SHUTDOWN;
  Thread_shutdown(priv->thread);
  return 0; // Never reached
}


void
Thread::os_start(bool stopped)
{
  if (!initialized)
    Thread::initialize();

  priv_ = new Thread_private(stopped);

  priv_->state = STARTUP;
  priv_->bstacksize = 0;
  for (int i = 0;i<MAXBSTACK;i++)
    priv_->blockstack[i]=bstack_init;

  priv_->thread = this;
  priv_->threadid = 0;
  priv_->is_blocked = false;
  priv_->ismain = false;

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, stacksize_);

  lock_scheduler();
  active[numActive]=priv_;
  numActive++;
  if (pthread_create(&priv_->threadid, &attr, run_threads, priv_) != 0)
  {
    // Always EAGAIN
    throw ThreadError("pthread_create:  Out of thread resources.");
  }
  unlock_scheduler();
}


void
Thread::stop()
{
  lock_scheduler();
  if (priv_->block_sema.tryDown() == false)
  {
    if (this == self())
    {
      priv_->block_sema.down();
    }
    else
    {
      pthread_kill(priv_->threadid, SIGUSR2);
    }
  }
  unlock_scheduler();
}


void
Thread::resume()
{
  lock_scheduler();
  priv_->block_sema.up();
  unlock_scheduler();
}


void
Thread::detach()
{
  detached_=true;
  pthread_t id = priv_->threadid;

  priv_->delete_ready.up();

  const int status = pthread_detach(id);
  if (status)
  {
    switch (status)
    {
    case ESRCH:
      throw ThreadError("pthread_detach:  Thread does not exist.");
      break;

    case EINVAL:
      throw ThreadError("pthread_detach:  Thread is already detached.");
      break;

    default:
      throw ThreadError("pthread_detach:  Unknown error.");
    }
  }
}


void
Thread::exitAll(int code)
{
  if (getenv("SCIRUN_EXIT_CRASH_WORKAROUND")) {
    raise(SIGKILL);
  }
  CleanupManager::call_callbacks();
  if (initialized && !exiting) {
    exiting = true;
    lock_scheduler();
    if( initialized ){
      // Stop all of the other threads before we die, because
      // global destructors may destroy primitives that other
      // threads are using...
      Thread* me = Thread::self();
      for (int i = 0;i<numActive;i++){
        Thread_private * thread_priv = active[i];

        // It seems like this is the correct place to call handleCleanup (on all the threads)...
        // However, I haven't tested this feature in SCIRun itself... it does work for Uintah
        // (which only has one (the main) thread).
        thread_priv->thread->handleCleanup();

        if (thread_priv->thread != me){
          pthread_kill(thread_priv->threadid, SIGUSR2);
        }
      }
      // Wait for all threads to be in the signal handler
      int numtries = 100000;
      bool done = false;
      while(--numtries && !done){
        done = true;
        for (int i = 0;i<numActive;i++){
          Thread_private * thread_priv = active[i];
          if (thread_priv->thread != me){
            if (!thread_priv->is_blocked)
              done = false;
          }
        }
        sched_yield();
        //sleep(1);
      }
      if (!numtries){
        for (int i = 0;i<numActive;i++){
          Thread_private* thread_priv = active[i];
          if ( thread_priv->thread != me && !thread_priv->is_blocked ) {
            fprintf(stderr, "Thread: %s is slow to stop, giving up\n",
                    thread_priv->thread->getThreadName());
            //sleep(1000);
          }
        }
      }
    } // end if( initialized )

    // See Thread.h for why we are doing this.
    if (Thread::getCallExit()) {
      ::exit(code);
    }
  }
  else if ( !initialized ) {
    // This case happens if the thread library is not being used.
    // Just use the normal exit function.
    ::exit(code);
  }
}


/*
 * Handle an abort signal - like segv, bus error, etc.
 */
static
void
#if defined(__sgi)
handle_abort_signals(int sig, int, SigContext* ctx)
#elif defined (__CYGWIN__)
handle_abort_signals(int sig)
#else
handle_abort_signals(int sig, SigContext ctx)
#endif
{
  struct sigaction action;
  sigemptyset(&action.sa_mask);
  action.sa_handler = SIG_DFL;
  action.sa_flags = 0;
  if (sigaction(sig, &action, NULL) == -1) {
    throw ThreadError(std::string("sigaction failed") + strerror(errno));
  }

  Thread* self = Thread::self();
  const char* tname = self?self->getThreadName():"idle or main";
#if defined(__sgi)
#  if defined(_LONGLONG)
  caddr_t addr = (caddr_t)ctx->sc_badvaddr;
#  else
  caddr_t addr = (caddr_t)ctx->sc_badvaddr.lo32;
#  endif
#else
#  if defined(PPC)
  void* addr = (void*)ctx.regs->dsisr;
#  else
#    if defined(_AIX)
  // Not sure if this is correct, but here it is.
  // On IMB SP2 sigcontext is defined in /usr/include/sys/context.h
#      if defined(SCI_64BITS)
  void* addr = (void*)ctx.sc_jmpbuf.jmp_context.except;
#      else
  void* addr = (void*)ctx.sc_jmpbuf.jmp_context.o_vaddr;
#      endif
#    else
  //     void* addr = (void*)ctx.cr2;
  void* addr = 0;
#    endif
#  endif
#endif
  char* signam = Core_Thread_signal_name(sig, addr);
  
  //don't print if the signal was SIGINT because the signal likely came from MPI aborting 
  //and the problem was likely on another processor
  bool print=sig!=SIGINT;

  Uintah::CrashPad::printMessages(std::cout);
  
  if(print)
    fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) caught signal %s\n", 7,7,7,tname, getpid(), signam);

  SCIRun::WAIT_FOR_DEBUGGER(true);

  Thread::niceAbort(NULL,print);
  
  action.sa_handler = (SIG_HANDLER_T)handle_abort_signals;
  action.sa_flags = 0;
  if (sigaction(sig, &action, NULL) == -1) {
    throw ThreadError(std::string("sigaction failed") + strerror(errno));
  }
} // handle_abort_signals()


void
Thread::print_threads()
{
  FILE* fp = stderr;
  for (int i = 0;i<numActive;i++)
  {
    Thread_private* p = active[i];
    const char* tname = p->thread?p->thread->getThreadName():"???";
    // NOTE(boulos): On Darwin, pthread_t is an opaque type and so
    // using it as a thread_id (despite the man page) is a bad idea.
    // For this purpose (printing out some unique identifier) it
    // should be fine but requires a C-style cast.
    long unsigned int tid = (long unsigned int)(p->threadid);
    fprintf(fp, " %lu: %s (", tid, tname);
    if (p->thread)
    {
      if (p->thread->isDaemon())
        fprintf(fp, "daemon, ");
      if (p->thread->isDetached())
        fprintf(fp, "detached, ");
    }
    fprintf(fp, "state = %s", Thread::getStateString(p->state));
    for (int i = 0;i<p->bstacksize;i++)
    {
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
#ifdef __CYGWIN__
// Cygwin's version doesn't take a sigcontext
handle_quit(int sig)
#else
handle_quit(int sig, SigContext /*ctx*/)
#endif
{
  // Try to acquire a lock.  If we can't, then assume that somebody
  // else already caught the signal...
  Thread* self = Thread::self();
  if (self == 0)
    return; // This is an idle thread...
  if (!(control_c_sema.tryDown()))
  {
    control_c_sema.down();
    control_c_sema.up();
  }

  // Otherwise, we got the semaphore and handle the interrupt
  const char* tname = self?self->getThreadName():"main?";

  // Kill all of the threads...
  char* signam = Core_Thread_signal_name(sig, 0);
  int pid = getpid();
  
  //don't print if the signal was SIGINT because the signal likely came from MPI aborting 
  //and the problem was likely on another processor
  bool print=sig!=SIGINT;
  
  Uintah::CrashPad::printMessages(std::cout);

  if(print)
    fprintf(stderr, "Thread \"%s\"(pid %d) caught signal %s\n", tname, pid, signam);

  SCIRun::WAIT_FOR_DEBUGGER(true);

  Thread::niceAbort(NULL, print); // Enter the monitor
  control_c_sema.up();
}


/*
 * Handle siguser1 - for stop/resume
 */
static
void
handle_siguser2(int)
{
  Thread* self = Thread::self();
  if (!self)
  {
    // This can happen if the thread is just started and hasn't had
    // the opportunity to call setspecific for the thread id yet
    for (int i = 0;i<numActive;i++)
      if (pthread_self() == active[i]->threadid)
        self = active[i]->thread;
  }
  self->priv_->is_blocked = true;
  self->priv_->block_sema.down();
  self->priv_->is_blocked = false;
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
  action.sa_flags = 0;

  action.sa_handler = (SIG_HANDLER_T)handle_abort_signals;
  if (sigaction(SIGILL, &action, NULL) == -1)
    throw ThreadError(std::string("SIGILL failed") + strerror(errno));
  if (sigaction(SIGABRT, &action, NULL) == -1)
    throw ThreadError(std::string("SIGABRT failed") + strerror(errno));
  if (sigaction(SIGTRAP, &action, NULL) == -1)
    throw ThreadError(std::string("SIGTRAP failed") + strerror(errno));
  if (sigaction(SIGBUS, &action, NULL) == -1)
    throw ThreadError(std::string("SIGBUS failed") + strerror(errno));
  if (sigaction(SIGSEGV, &action, NULL) == -1)
    throw ThreadError(std::string("SIGSEGV failed") + strerror(errno));

  action.sa_handler = (SIG_HANDLER_T)handle_quit;
  if (sigaction(SIGQUIT, &action, NULL) == -1)
    throw ThreadError(std::string("SIGQUIT failed") + strerror(errno));
  if (sigaction(SIGINT, &action, NULL) == -1)
    throw ThreadError(std::string("SIGINT failed") + strerror(errno));

  action.sa_handler = (SIG_HANDLER_T)handle_siguser2;
  if (sigaction(SIGUSR2, &action, NULL) == -1)
    throw ThreadError(std::string("SIGUSR2 failed") + strerror(errno));
}


static void
exit_handler()
{
  Thread * self = Thread::self();

  // Self appears to be able to be NULL if the program has already
  // mostly shutdown before the atexit() function (this function)
  // kicks in.

  if( exiting || self == NULL ) {
    return;
  }
  Thread_shutdown( self );
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

  if (initialized)
    return;
  if (exiting)
    abort(); // Something really weird happened!

  CleanupManager::initialize();

  if (!getenv("THREAD_NO_ATEXIT"))
    atexit(exit_handler);

  pthread_mutex_init(&sched_lock, NULL);

  if (pthread_key_create(&thread_key, NULL) != 0)
  {
    throw ThreadError("pthread_key_create:  Out of resources.");
  }

  initialized = true;
  ThreadGroup::s_default_group = new ThreadGroup("default group", 0);
  mainthread = new Thread(ThreadGroup::s_default_group, "main");
  //  mainthread->priv_=new Thread_private(false);
  mainthread->priv_ = new Thread_private(true);
  mainthread->priv_->thread = mainthread;
  mainthread->priv_->state = RUNNING;
  mainthread->priv_->bstacksize = 0;
  mainthread->priv_->is_blocked = false;
  mainthread->priv_->threadid = pthread_self();
  mainthread->priv_->ismain = true;

  //  mainthread->priv_->block_sema.down();

  for (int i = 0;i<MAXBSTACK;i++)
    mainthread->priv_->blockstack[i]=bstack_init;
  if (pthread_setspecific(thread_key, mainthread) != 0)
  {
    throw ThreadError("pthread_setspecific:  Failed.");
  }

  lock_scheduler();
  active[numActive]=mainthread->priv_;
  numActive++;
  unlock_scheduler();
  if (!getenv("THREAD_NO_CATCH_SIGNALS"))
    install_signal_handlers();
  numProcessors();  //initialize the processor count;
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
} // namespace SCIRun


Mutex::Mutex(const char* name)
  : name_(name)
{
  // DO NOT CALL INITIALIZE in this CTOR!
  if (this == 0){
    fprintf(stderr, "WARNING: creation of null mutex\n");
  }

  priv_ = new Mutex_private;

#ifdef PTHREAD_MUTEX_ERRORCHECK_NP
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr); // always returns zero
  if (pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK_NP) != 0)
    throw ThreadError("pthread_mutexattr_settype:  Bad kind.");
  pthread_mutex_init(&priv_->mutex, &attr); // always returns zero
  pthread_mutexattr_destroy(&attr);  // usually noop
#else
  pthread_mutex_init(&priv_->mutex, NULL); // always returns zero
#endif
}


Mutex::~Mutex()
{
  // RHE 3.0 pthread_mutex_destroy returns EBUSY if you unlock an
  // already unlocked thread first.  So we force a lock/unlock before
  // destroying.  Probably better to just remove the unlock altogether
  // but this breaks the shutdown behavior.  TODO: This fix doesn't
  // really work.  Race condition can relock between our unlock and
  // destroy calls.
  pthread_mutex_trylock(&priv_->mutex);
  pthread_mutex_unlock(&priv_->mutex);
  if (pthread_mutex_destroy(&priv_->mutex) != 0)
  {
    // EBUSY
    fprintf(stderr,"Mutex::~Mutex: Warning: Mutex \"%s\" currently locked.\n",
            name_);
    priv_ = 0;
    return;
    // EBUSY
    //throw ThreadError("pthread_mutex_destroy: Mutex currently locked.");
  }
  delete priv_;
  priv_ = 0;
}

void
Mutex::unlock()
{
  int status = pthread_mutex_unlock(&priv_->mutex);
  if (status)
  {
    switch (status)
    {
    case EINVAL:
      ThreadError("pthread_mutex_unlock:  Uninitialized lock.");
      break;

    case EPERM:
      ThreadError("pthread_mutex_unlock:  Calling thread did not lock.");
      break;

    default:
      ThreadError("pthread_mutex_unlock:  Unknown error.");
    }
  }
}

void
Mutex::lock()
{
  Thread* t = Thread::isInitialized()?Thread::self():0;
  int oldstate = -1;
  Thread_private* p = 0;
  if (t){
    p = t->priv_;
    oldstate = Thread::push_bstack(p, Thread::BLOCK_MUTEX, name_);
  }

#if defined( __APPLE__ ) || defined ( _AIX )
  // Temporary hack:
  // On OSX and AIX, this call may come before the constructor (for
  // static vars) for some reason.  To solve this problem we allocate
  // priv_ and init it if the constructor was not called yet.
  // Note:  This would appear to cause a deadlock or crash
  // if we lock on priv_ and then call the constructor to replace it.
  if ( !priv_ ) {
    priv_=new Mutex_private;
    pthread_mutex_init(&priv_->mutex, NULL);
  }
#endif

  int status = pthread_mutex_lock(&priv_->mutex);
  if (status)
  {
    switch (status)
    {
    case EINVAL:
      throw ThreadError("pthread_mutex_lock:  Uninitialized lock.");
      break;

    case EDEADLK:
      throw ThreadError("pthread_mutex_lock:  Calling thread already holds this lock.");
      break;

    default:
      throw ThreadError("pthread_mutex_lock:  Unknown error.");
    }
  }

  if (t)
  {
    Thread::pop_bstack(p, oldstate);
  }
}


bool
Mutex::tryLock()
{
  int status = pthread_mutex_trylock(&priv_->mutex);
  switch (status)
  {
  case 0:
    return true;

  case EBUSY:
    return false;

  default: // EINVAL
    throw ThreadError("pthread_mutex_trylock:  Uninitialized lock.");
  }
}


namespace SCIRun {
struct RecursiveMutex_private {
  pthread_mutex_t mutex;
};
} // namespace SCIRun


RecursiveMutex::RecursiveMutex(const char* name)
  : name_(name)
{
  if (!Thread::initialized)
    Thread::initialize();
  priv_ = new RecursiveMutex_private;

  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  if (pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE) != 0)
    throw ThreadError("pthread_mutexattr_settype: unknown kind.");
  pthread_mutex_init(&priv_->mutex, &attr);
  pthread_mutexattr_destroy(&attr);
}


RecursiveMutex::~RecursiveMutex()
{
  // RHE 3.0 pthread_mutex_destroy returns EBUSY if you unlock an
  // already unlocked thread first.  So we force a lock/unlock before
  // destroying.  Probably better to just remove the unlock altogether
  // but this breaks the shutdown behavior.  TODO: This fix doesn't
  // really work.  Race condition can relock between our unlock and
  // destroy calls.
  pthread_mutex_trylock(&priv_->mutex);
  pthread_mutex_unlock(&priv_->mutex);
  if (pthread_mutex_destroy(&priv_->mutex) != 0)
  {
    // EBUSY
    fprintf(stderr, "RecursiveMutex::~RecursiveMutex: Warning:  Mutex currently locked.\n");
    priv_ = 0;
    return;
    //throw ThreadError("pthread_mutex_destroy: Mutex currently locked.");
  }
  delete priv_;
  priv_=0;
}


void
RecursiveMutex::unlock()
{
  const int status = pthread_mutex_unlock(&priv_->mutex);
  if (status)
  {
    switch (status)
    {
    case EINVAL:
      ThreadError("pthread_mutex_unlock:  Uninitialized lock.");
      break;

    case EPERM:
      ThreadError("pthread_mutex_unlock:  Calling thread did not lock.");
      break;

    default:
      ThreadError("pthread_mutex_unlock:  Unknown error.");
    }
  }
}


void
RecursiveMutex::lock()
{
  Thread* self = Thread::self();
  int oldstate = 0;
  Thread_private* p = NULL;
  if (self) {
    p = Thread::self()->priv_;
    oldstate = Thread::push_bstack(p, Thread::BLOCK_ANY, name_);
  }
  const int status = pthread_mutex_lock(&priv_->mutex);
  if (status)
  {
    switch (status)
    {
    case EINVAL:
      throw ThreadError("pthread_mutex_lock:  Uninitialized lock.");
      break;

    case EDEADLK:
      throw ThreadError("pthread_mutex_lock:  Calling thread already holds this lock.");
      break;

    default:
      throw ThreadError("pthread_mutex_lock:  Unknown error.");
    }
  }

  if (self) Thread::pop_bstack(p, oldstate);
}


namespace SCIRun {
#if defined (__APPLE__)
struct Semaphore_private {
  Semaphore_private(const char *name, int value);
  Mutex mutex_;
  int   cnt_;
  ConditionVariable cv_;
};

Semaphore_private::Semaphore_private(const char *name, int value) :
  mutex_(name),
  cnt_(value),
  cv_(name)
{
}

#else
struct Semaphore_private {
  sem_type sem;
};
#endif
} // namespace SCIRun



#if defined(__APPLE__)

Semaphore::Semaphore(const char *name,int value)
  : name_(name)
{
  priv_ = new Semaphore_private(name,value);
}


Semaphore::~Semaphore()
{
  if (priv_)
  {
    delete priv_;
    priv_ = 0;
  }
}


void
Semaphore::down(int count)
{
  for (int p = 0 ; p < count; p++)
  {
    priv_->mutex_.lock();
    priv_->cnt_--;
    if (priv_->cnt_ < 0) priv_->cv_.wait(priv_->mutex_);
    priv_->mutex_.unlock();
  }
}


bool
Semaphore::tryDown()
{
  priv_->mutex_.lock();
  if (priv_->cnt_ > 0)
  {
    priv_->cnt_--;
    priv_->mutex_.unlock();
    return(true);
  }
  priv_->mutex_.unlock();
  return(false);
}


void
Semaphore::up(int count)
{
  for (int p = 0;p < count; p++)
  {
    priv_->mutex_.lock();
    priv_->cv_.conditionBroadcast();
    priv_->cnt_++;
    priv_->mutex_.unlock();
  }
}

#else

Semaphore::Semaphore(const char* name, int value)
  : name_(name)
{
  if (!Thread::initialized)
    Thread::initialize();
  priv_=new Semaphore_private;

#if defined(_AIX)
  priv_->sem =
    (msemaphore*) mmap(NULL,sizeof(msemaphore),
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS | MAP_VARIABLE, -1, 0 );
#endif

  if ( !SEM_INIT_SUCCESS( SEM_INIT(&(priv_->sem), 0, value) ) )
    throw ThreadError(std::string("SEM_INIT: ") + strerror(errno));
}


Semaphore::~Semaphore()
{
#if !defined(_AIX)
  // Dd: Don't know exactly what to do about this for AIX...
  int val;
  sem_getvalue(&(priv_->sem),&val);
  while (val<=0)
  {
    SEM_UNLOCK(&(priv_->sem));
    sem_getvalue(&(priv_->sem),&val);
  }
  if (SEM_DESTROY(&priv_->sem) != 0)
  {
    throw ThreadError(std::string("sem_destroy: ")
                      +strerror(errno));
    perror("Sem destroy" );
  }
  delete priv_;
  priv_=0;
#endif
}


void
Semaphore::up(int count)
{
  for (int i = 0;i<count;i++)
  {
    if (SEM_UNLOCK(&priv_->sem) != 0)
      throw ThreadError(std::string("SEM_UNLOCK: ") + strerror(errno));
  }
}


void
Semaphore::down(int count)
{
  Thread* self = Thread::self();
  int oldstate = 0;
  Thread_private* p = NULL;
  if (self) {
    p = Thread::self()->priv_;
    oldstate = Thread::push_bstack(p, Thread::BLOCK_SEMAPHORE, name_);
  }
  for (int i = 0;i<count;i++)
  {
    if (SEM_LOCK(&priv_->sem) != 0)
    {
      perror("sem lock");
      throw ThreadError(std::string("SEM_LOCK: ") + strerror(errno));
    }
  }
  if (self) Thread::pop_bstack(p, oldstate);
}


bool
Semaphore::tryDown()
{
  if (SEM_TRYLOCK(&priv_->sem) != 0)
  {
    if (errno == EAGAIN)
      return false;
    throw ThreadError(std::string("SEM_TRYLOCK: ") + strerror(errno));
  }
  return true;
}

#endif

namespace SCIRun {
struct ConditionVariable_private {
  pthread_cond_t cond;
};
} // namespace SCIRun


ConditionVariable::ConditionVariable(const char* name)
  : name_(name)
{
  if (!Thread::initialized)
    Thread::initialize();
  priv_ = new ConditionVariable_private;
  pthread_cond_init(&priv_->cond, 0);
}


ConditionVariable::~ConditionVariable()
{
  if (pthread_cond_destroy(&priv_->cond) != 0)
  {
    ThreadError("pthread_cond_destroy:  Threads are currently waiting on this condition.");
  }
  delete priv_;
  priv_=0;
}


void
ConditionVariable::wait(Mutex& m)
{
  Thread* self = Thread::self();
  if (self) {
    Thread_private* p = Thread::self()->priv_;
    int oldstate = Thread::push_bstack(p, Thread::BLOCK_ANY, name_);

    pthread_cond_wait(&priv_->cond, &m.priv_->mutex);

    Thread::pop_bstack(p, oldstate);
  } else {

    pthread_cond_wait(&priv_->cond, &m.priv_->mutex);
  }
}


bool
ConditionVariable::timedWait(Mutex& m, const struct timespec* abstime)
{
  Thread* self = Thread::self();
  int oldstate = 0;
  Thread_private* p = NULL;
  if (self) {
    p = Thread::self()->priv_;
    oldstate = Thread::push_bstack(p, Thread::BLOCK_ANY, name_);
  }
  bool success;
  if (abstime){
    int err = pthread_cond_timedwait(&priv_->cond, &m.priv_->mutex, abstime);
    if (err != 0){
      if (err == ETIMEDOUT)
        success = false;
      else
        throw ThreadError("pthread_cond_timedwait:  Interrupted by a signal.");
    } else {
      success = true;
    }
  } else {
    pthread_cond_wait(&priv_->cond, &m.priv_->mutex);
    success = true;
  }
  if (self) Thread::pop_bstack(p, oldstate);
  return success;
}


void
ConditionVariable::conditionSignal()
{
  pthread_cond_signal(&priv_->cond);
}


void
ConditionVariable::conditionBroadcast()
{
  pthread_cond_broadcast(&priv_->cond);
}

#ifdef __ia64__

using SCIRun::Barrier;

namespace SCIRun {
struct Barrier_private {
  Barrier_private();

  //  long long amo_val;
  char pad0[128];
  __int64 amo_val;
  char pad1[128];
  volatile int flag;
  char pad2[128];
};
} // namespace SCIRun


using SCIRun::Barrier_private;


Barrier_private::Barrier_private()
{
  flag = 0;
  amo_val = 0;
}


Barrier::Barrier(const char* name)
  : name_(name)
{
  if (!Thread::isInitialized())
  {
    if (getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "Barrier: %s\n", name);
    Thread::initialize();
  }
  priv_=new Barrier_private;
}


Barrier::~Barrier()
{
  delete priv_;
  priv_=0;
}


void
Barrier::wait(int n)
{
  Thread* self = Thread::self();
  int oldstate;
  Thread_private* p;
  if (self) {
    p = Thread::self()->priv_;
    oldstate = Thread::push_bstack(p, Thread::BLOCK_BARRIER, name_);
  }
  int gen = priv_->flag;
  __int64 val = __sync_fetch_and_add_di(&(priv_->amo_val),1);
  if (val == n-1){
    priv_->amo_val = 0;
    priv_->flag++;
  }
  while(priv_->flag==gen)
    /* spin */ ;
  if (self) Thread::pop_bstack(p, oldstate);
}


using SCIRun::AtomicCounter;


namespace SCIRun {
struct AtomicCounter_private {
  AtomicCounter_private();

  // These variables used only for non fectchop implementation
  //  long long amo_val;
  char pad0[128];
  __int64 amo_val;
  char pad1[128];
};
} // namespace SCIRun

using SCIRun::AtomicCounter_private;


AtomicCounter_private::AtomicCounter_private()
{
}


AtomicCounter::AtomicCounter(const char* name)
  : name_(name)
{
  if (!Thread::isInitialized())
  {
    if (getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "AtomicCounter: %s\n", name);
    Thread::initialize();
  }
  priv_=new AtomicCounter_private;
  priv_->amo_val = 0;
}


AtomicCounter::AtomicCounter(const char* name, int value)
  : name_(name)
{
  if (!Thread::isInitialized())
  {
    if (getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "AtomicCounter: %s\n", name);
    Thread::initialize();
  }
  priv_=new AtomicCounter_private;
  priv_->amo_val = value;
}


AtomicCounter::~AtomicCounter()
{
  delete priv_;
  priv_=0;
}


AtomicCounter::operator int() const
{
  return (int)(priv_->amo_val);
}

// Preincrement
int
AtomicCounter::operator++()
{
  __int64 val = __sync_fetch_and_add_di(&(priv_->amo_val),1);
  return (int)val+1;
}


// Postincrement
int
AtomicCounter::operator++(int)
{
  __int64 val = __sync_fetch_and_add_di(&(priv_->amo_val),1);
  return (int)val;
}

// Predecrement
int
AtomicCounter::operator--()
{
  __int64 val = __sync_fetch_and_add_di(&(priv_->amo_val),-1);
  return (int)val-1;
}

// Postdecrement
int
AtomicCounter::operator--(int)
{
  __int64 val = __sync_fetch_and_add_di(&(priv_->amo_val),-1);
  return (int)val;
}


void
AtomicCounter::set(int v)
{
  priv_->amo_val = v;
}

#endif // end #ifdef __ia64__
