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
 */

// This is brutal, but effective.  We want this file to have access
// to the private members of Thread, without having to explicitly
// declare friendships in the class.
#define private public
#define protected public
#include <Core/Thread/Thread.h>
#undef private
#undef protected
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ThreadError.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/WorkQueue.h>
#include <Core/Thread/Thread_unix.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <sys/signal.h>
#include <sys/time.h>
#include <sys/types.h>

#include <stack>

/*
 * This Thread implementation is for no threads.  Creating a Thread will throw
 * an exception, and all other functions will do nothing.
 */

#include <Core/Thread/CrowdMonitor_default.cc>
#include <Core/Thread/RecursiveMutex_default.cc>


extern "C" int __ateachexit(void(*)());

#define MAXBSTACK 10
#define MAXTHREADS 4000

#define N_POOLMUTEX 301

namespace SCIRun {

static void install_signal_handlers();


/*
 * Intialize threads for irix
 */
void
Thread::initialize()
{
  printf("I called Thread::initialize\n");
  install_signal_handlers();
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
Thread::exit()
{
  printf("I called Thread::exit\n");
  ::exit(0);
}

void
Thread::checkExit()
{
  printf("I called Thread::checkExit\n");
  ::exit(0);
}

void
Thread::exitAll(int code)
{
  printf("I called Thread::exitAll\n");

  ::exit(code);
}

/*
 * Startup the given thread...
 */
void
Thread::os_start(bool stopped)
{
  printf("I called Thread::os_start\n");

}

void
Thread::stop()
{
  printf("I called Thread::stop\n");
}

void
Thread::resume()
{
  printf("I called Thread::resume\n");
}

void
Thread::detach()
{
  printf("I called Thread::resume\n");

}

void
Thread::join()
{
  printf("I called Thread::join\n");

}

void
ThreadGroup::gangSchedule()
{
  printf("I called Thread::gangSchedule\n");
}

/*
 * Thread block stack manipulation
 */
int
Thread::push_bstack(Thread_private* p, Thread::ThreadState state,
		    const char* name)
{
  printf("I called Thread::push_bstack\n");
  return 0;
}

void
Thread::pop_bstack(Thread_private* p, int oldstate)
{
  printf("I called Thread::pop_bstack\n");
}

/*
 * Signal handling cruft
 */
void
Thread::print_threads()
{
  printf("I called Thread::print_threads\n");

}

typedef void (*SIG_HANDLER_T)(int);

/*
 * Handle sigquit - usually sent by control-C
 */
static
void
handle_quit(int sig, int /* code */, sigcontext)
{
  if(sig==SIGINT){
    // Print out the thread states...
    Thread::niceAbort();
  } else {
    exit(1);
  }
}

/*
 * Handle an abort signal - like segv, bus error, etc.
 */
static
void
handle_abort_signals(int sig, int /* code */, sigcontext context)
{
#if 0
  printf ("Ready to handle_abort_signals\n");
  fflush(stdout);
  printf("%d\n", (int)*(int*)0x400144080);
  fflush(stdout);
#endif
  struct sigaction action;
  sigemptyset(&action.sa_mask);
  action.sa_handler=SIG_DFL;
  action.sa_flags=0;
  if(sigaction(sig, &action, NULL) == -1)
    throw ThreadError(std::string("sigaction (SIG_DFL) failed")
		      +strerror(errno));

  Thread* self=Thread::self();
  const char* tname=self?self->getThreadName():"idle or main";
#if defined(__sgi)
#  if defined(_LONGLONG)
     caddr_t addr=(caddr_t)ctx->sc_badvaddr;
#  else
     caddr_t addr=(caddr_t)ctx->sc_badvaddr.lo32;
#  endif
#else
#  if defined(PPC)
    void* addr=(void*)ctx.regs->dsisr;
#  else
#    if defined(_AIX)
       // Not sure if this is correct, but here it is.
       // On IMB SP2 sigcontext is defined in /usr/include/sys/context.h
#      if defined(SCI_64BITS)
         void* addr=(void*)ctx.sc_jmpbuf.jmp_context.except;
#      else
         void* addr=(void*)ctx.sc_jmpbuf.jmp_context.o_vaddr;
#      endif
#    else
//     void* addr=(void*)ctx.cr2;
       void* addr=0;
#    endif
#  endif
#endif
  char* signam=Core_Thread_signal_name(sig, addr);
  fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) caught signal %s\n", 7,7,7,tname, getpid(), signam);
  Thread::niceAbort();

  action.sa_handler=(SIG_HANDLER_T)handle_abort_signals;
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

  action.sa_handler=(SIG_HANDLER_T)handle_abort_signals;
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

  action.sa_handler=(SIG_HANDLER_T)handle_quit;
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
  return NULL;
}


int
Thread::numProcessors()
{
  return 1;
}

/*
 * Yield the CPU
 */
void
Thread::yield()
{
}

/*
 * Migrate the thread to a CPU.
 */
void
Thread::migrate(int proc)
{
}

/*
 * Mutex implementation
 */
Mutex::Mutex(const char* name)
  : name_(name)
{
}

Mutex::~Mutex()
{
}

void
Mutex::lock()
{
}

bool
Mutex::tryLock()
{
  return true;
}

void
Mutex::unlock()
{
}

/*
 * Semaphore implementation
 */

Semaphore::Semaphore(const char* name, int count)
  : name_(name)
{
}

Semaphore::~Semaphore()
{
}

void
Semaphore::down(int count)
{
}

bool
Semaphore::tryDown()
{
  return true;
}

void
Semaphore::up(int count)
{
}

Barrier::Barrier(const char* name)
  : name_(name)
{
}

Barrier::~Barrier()
{
}

void
Barrier::wait(int n)
{
}

struct AtomicCounter_private {
  AtomicCounter_private();

  // These variables used only for non fectchop implementation
  Mutex lock;
  int value;
};

AtomicCounter_private::AtomicCounter_private()
  : lock("AtomicCounter lock")
{
}

AtomicCounter::AtomicCounter(const char* name)
  : name_(name)
{
  priv_=new AtomicCounter_private;
  priv_->value=0;
}

AtomicCounter::AtomicCounter(const char* name, int value)
  : name_(name)
{
  priv_=new AtomicCounter_private;
  priv_->value=value;
}

AtomicCounter::~AtomicCounter()
{
  delete priv_;
  priv_=0;
}

AtomicCounter::operator int() const
{
  return priv_->value;
}

int
AtomicCounter::operator++()
{
  int ret=++priv_->value;
  return ret;
}

int
AtomicCounter::operator++(int)
{
  int ret=priv_->value++;
  return ret;
}

int
AtomicCounter::operator--()
{
  int ret=--priv_->value;	
  return ret;
}

int
AtomicCounter::operator--(int)
{
  int ret=priv_->value--;
  return ret;
}

void
AtomicCounter::set(int v)
{
    priv_->value=v;
}

struct ConditionVariable_private {
  int num_waiters;
  bool pollsema;
};

ConditionVariable::ConditionVariable(const char* name)
  : name_(name)
{
  priv_=new ConditionVariable_private();
  priv_->num_waiters=0;
  priv_->pollsema=false;
}

ConditionVariable::~ConditionVariable()
{
  delete priv_;
  priv_=0;
}

void
ConditionVariable::wait(Mutex& m)
{
}

bool
ConditionVariable::timedWait(Mutex& m, const struct timespec* abstime)
{
  return true;
}

void
ConditionVariable::conditionSignal()
{
}

void
ConditionVariable::conditionBroadcast()
{
}

} // End namespace SCIRun
