/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  Thread_win32.cc: win32 threads implementation of the thread library
 *
 *  Written by:
 *   Author: Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   Date: November 1999
 *
 */

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Thread_unix.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/ThreadError.h>

// version of windows
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0500
#endif


#define __REENTRANT
#define _WINSOCK2API_
#include <windows.h>
#include <excpt.h>
#include <signal.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <map>
#undef private
#undef public

using std::cerr;
using std::endl;
using std::map;
using std::string;

#define MAX(x,y) ((x>y)?x:y)
#define MIN(x,y) ((x<y)?x:y)

#define MAXBSTACK 10
#define MAXTHREADS 4000

namespace SCIRun {
  // a naive way to get Thread::self when the compiler 
  // doesn't support thread-local storage
  map<int, Thread*> threadids; 

  // NOTE - Do not give semaphores or mutexes names - there's a good chance that
  // if they have the same name they will conflict with each other (multiple semaphores
  // with the same name act as one semaphore).
  struct Thread_private {
  public:
    HANDLE t;                       // native thread
    Thread* thread;
    int threadid;
    Thread::ThreadState state;
    int bstacksize;
    const char* blockstack[MAXBSTACK];
    HANDLE done;                    // native semaphore
    HANDLE delete_ready;            // native semaphore
    HANDLE main_sema;
    HANDLE control_c_sema;

    static void initialize() { Thread::initialize(); }
  };
  
  struct Mutex_private {
    HANDLE lock;
  };
  
  struct Semaphore_private {
    HANDLE hSema;
  };

  struct CrowdMonitor_private {
    ConditionVariable write_waiters;
    ConditionVariable read_waiters;
    Mutex lock;
    int num_readers_waiting;
    int num_writers_waiting;
    int num_readers;
    int num_writers;
    CrowdMonitor_private();
    ~CrowdMonitor_private();
  };

  // implemented with help from the article at 
  // http://www.cs.wustl.edu/~schmidt/win32-cv-1.html
  struct ConditionVariable_private {
    Mutex waiters_count_lock_;
    int waiters_count_;     // num waiting threads
    HANDLE sema_;           // used to queue threads
    HANDLE waiters_done_;   // auto-reset event used by broadcast/signal
                            // to wait for waiting threads
    size_t was_broadcast_;  // whether we're broadcasting or signalling
    ConditionVariable_private(const char* name);
    ~ConditionVariable_private();
  };

  struct Barrier_private {
    Mutex mutex;
    ConditionVariable cond0;
    ConditionVariable cond1;
    int cc;
    int nwait;
    Barrier_private();
    ~Barrier_private();
  };

  struct AtomicCounter_private {
    Mutex lock;
    int value;
    AtomicCounter_private();
    ~AtomicCounter_private();
  };

  struct ThreadLocalMemory {
    Thread* current_thread;
  };

/*  Thread-local storage - not all compilers support it
#if 1
  // this is the VC way to give thread-local storage
__declspec(thread) ThreadLocalMemory* thread_local;
#else
__thread ThreadLocalMemory* thread_local; // __attribute__ ((tls_model ("initial-exec")));
#endif
*/
bool exiting=false;

  char* threadError()
  {
    char* lpMsgBuf;
    FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
      NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &lpMsgBuf, 0, NULL);

    return lpMsgBuf;

  }

Mutex::Mutex(const char* name)
{
	priv_ = scinew Mutex_private;
	priv_->lock = CreateMutex(NULL,0,0);
	if (priv_->lock == 0)
	{
	  throw ThreadError(std::string("CreateMutex failed")
			    +threadError());
	}

	if(getenv("THREAD_SHOWINIT"))
	  fprintf(stderr, "Mutex: %s\n", name);
	int length = strlen(name);
	name_ = new char[length+1];
	sprintf((char*)name_,"%s",name);
}

Mutex::~Mutex()
{
    CloseHandle(priv_->lock);
    delete[] (char*)name_;
    delete priv_;
    priv_=0;
}

void Mutex::lock()
{
	WaitForSingleObject(priv_->lock,INFINITE);
}

void Mutex::unlock()
{
	ReleaseMutex(priv_->lock);
}

bool Mutex::tryLock()
{
	int check = WaitForSingleObject(priv_->lock,0);
	if (check == WAIT_OBJECT_0)
		return 1;
	else if (check == WAIT_TIMEOUT)
		return 0;
	else 
	{
	  throw ThreadError(std::string("Mutex::try_lock failed")
			    +threadError());
	}
	return 0; // never happens
}

static Thread_private* active[MAXTHREADS];
static int numActive;
static HANDLE sched_lock;              
__declspec(dllexport) HANDLE main_sema;         
//static HANDLE control_c_sema;

Semaphore::Semaphore(const char* name,int count)
{
	priv_ = scinew Semaphore_private;
	priv_->hSema = CreateSemaphore(NULL,count,MAX(1000,MIN(2*count,100)),0);
	if (priv_->hSema == 0)
	{
	  throw ThreadError(std::string("CreateSemaphore failed")
			    +threadError());
	}
	if(getenv("THREAD_SHOWINIT"))
	  fprintf(stderr, "Semaphore: %s\n", name);
	int length = strlen(name);
	name_ = new char[length+1];
	sprintf((char*)name_,"%s",name);
}

Semaphore::~Semaphore()
{
    CloseHandle(priv_->hSema);
    delete[] (char*)name_;
    delete priv_;
    priv_=0;
}

void Semaphore::down(int dec)
{
	int check;
	for (int loop = 0;loop<dec;loop++) {
		check = WaitForSingleObject(priv_->hSema,INFINITE);
		if (check != WAIT_OBJECT_0)
		{
			if (check == WAIT_ABANDONED);
			else if (check == WAIT_TIMEOUT);
			else if (check == (int) WAIT_FAILED)
			{
				throw ThreadError(std::string("WaitForSingleObject failed")
						  +threadError());
			}
			else;
		}
	}
}

bool Semaphore::tryDown()
{
	int check = WaitForSingleObject(priv_->hSema,0);
	if (check == WAIT_OBJECT_0)
		return 0;
	else if (check == WAIT_TIMEOUT)
		return 1;
	else 
	{
	  throw ThreadError(std::string("WaitForSingleObject failed")
			    +threadError());
	}
	return 0; // never happens
}

void Semaphore::up(int inc)
{
	long count;
	ReleaseSemaphore(priv_->hSema,inc,&count);
}


AtomicCounter_private::AtomicCounter_private()
    : lock("AtomicCounter lock")
{
}

AtomicCounter_private::~AtomicCounter_private()
{
}

AtomicCounter::AtomicCounter(const char* name)
    : name_(name)
{
  if(!Thread::isInitialized()){
    if(getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "AtomicCounter: %s\n", name);
    Thread_private::initialize();
  }
  priv_=new AtomicCounter_private;
}

AtomicCounter::AtomicCounter(const char* name, int value)
    : name_(name)
{
  priv_=new AtomicCounter_private;
  priv_->value=value;
  if(getenv("THREAD_SHOWINIT"))
    fprintf(stderr, "AtomicCounter: %s\n", name);

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
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  int ret=++priv_->value;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

int
AtomicCounter::operator++(int)
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  int ret=priv_->value++;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

int
AtomicCounter::operator--()
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  int ret=--priv_->value;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

int
AtomicCounter::operator--(int)
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  int ret=priv_->value--;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

void
AtomicCounter::set(int v)
{
  int oldstate=Thread::couldBlock(name_);
  priv_->lock.lock();
  priv_->value=v;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
}


CrowdMonitor_private::CrowdMonitor_private()
  : write_waiters("CrowdMonitor write condition"),
    read_waiters("CrowdMonitor read condition"),
    lock("CrowdMonitor lock")
{
  num_readers_waiting=0;
  num_writers_waiting=0;
  num_readers=0;
  num_writers=0;
}

CrowdMonitor_private::~CrowdMonitor_private()
{
}

CrowdMonitor::CrowdMonitor(const char* name)
  : name_(name)
{
  if(!Thread::isInitialized()){
    if(getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "CrowdMonitor: %s\n", name);

    Thread_private::initialize();
  }
  priv_=new CrowdMonitor_private;
}

CrowdMonitor::~CrowdMonitor()
{
  delete priv_;
  priv_=0;
}

void
CrowdMonitor::readLock()
{
  int oldstate=Thread::couldBlock(name_);
  priv_->lock.lock();
  while(priv_->num_writers > 0){
    priv_->num_readers_waiting++;
    int s=Thread::couldBlock(name_);
    priv_->read_waiters.wait(priv_->lock);
    Thread::couldBlockDone(s);
    priv_->num_readers_waiting--;
  }
  priv_->num_readers++;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
}

void
CrowdMonitor::readUnlock()
{
  priv_->lock.lock();
  priv_->num_readers--;
  if(priv_->num_readers == 0 && priv_->num_writers_waiting > 0)
    priv_->write_waiters.conditionSignal();
  priv_->lock.unlock();
}

void
CrowdMonitor::writeLock()
{
  int oldstate=Thread::couldBlock(name_);
  priv_->lock.lock();
  while(priv_->num_writers || priv_->num_readers){
    // Have to wait...
    priv_->num_writers_waiting++;
    int s=Thread::couldBlock(name_);
    priv_->write_waiters.wait(priv_->lock);
    Thread::couldBlockDone(s);
    priv_->num_writers_waiting--;
  }
  priv_->num_writers++;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
} 

void
CrowdMonitor::writeUnlock()
{
  priv_->lock.lock();
  priv_->num_writers--;
  if(priv_->num_writers_waiting)
    priv_->write_waiters.conditionSignal(); // Wake one of them up...
  else if(priv_->num_readers_waiting)
    priv_->read_waiters.conditionBroadcast(); // Wake all of them up...
  priv_->lock.unlock();
}

Barrier_private::Barrier_private()
  : mutex("Barrier lock"),
    cond0("Barrier condition 0"), cond1("Barrier condition 1"),
    cc(0), nwait(0)
{
}

Barrier_private::~Barrier_private()
{
}

Barrier::Barrier(const char* name)
  : name_(name)
{
  if(!Thread::isInitialized()){
    if(getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "Barrier: %s\n", name);
    Thread_private::initialize();
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
  int oldstate=Thread::couldBlock(name_);
  priv_->mutex.lock();
  ConditionVariable& cond=priv_->cc?priv_->cond0:priv_->cond1;
  priv_->nwait++;
  if(priv_->nwait == n){
    // Wake everybody up...
    priv_->nwait=0;
    priv_->cc=1-priv_->cc;
    cond.conditionBroadcast();
  } else {
    cond.wait(priv_->mutex);
  }
  priv_->mutex.unlock();
  Thread::couldBlockDone(oldstate);
}

ConditionVariable_private::ConditionVariable_private(const char* name)
  : waiters_count_lock_(name), waiters_count_(0), was_broadcast_(0)
{
  sema_ = CreateSemaphore(NULL,0,50,NULL);
  if (sema_ == 0) {
    throw ThreadError(std::string("CreateSemaphore failed")
		      +threadError());
  }
  
  waiters_done_ = CreateEvent(NULL, FALSE, FALSE, NULL);
  if (waiters_done_ == 0) {
    throw ThreadError(std::string("CreateEvent failed")
		      +threadError());
  }

}

ConditionVariable_private::~ConditionVariable_private()
{
}

ConditionVariable::ConditionVariable(const char* name)
  : name_(name)
{
  if (!Thread::isInitialized()) 
    Thread_private::initialize();
  if(getenv("THREAD_SHOWINIT"))
    fprintf(stderr, "ConditionVariable: %s\n", name);
  priv_ = new ConditionVariable_private(name);
  
}

ConditionVariable::~ConditionVariable()
{
  delete priv_;
  priv_ = 0;
}

void
ConditionVariable::wait(Mutex& m)
{
  timedWait(m, 0);
}

bool
ConditionVariable::timedWait(Mutex& m, const struct timespec* abstime)
{
  bool success = true;
  DWORD waittime = 0;
  if (abstime) {
    // convert abstime to milliseconds
  } else {
    waittime = INFINITE;
  }

  Thread_private* p = Thread::self()->priv_;
  int oldstate = Thread::push_bstack(p, Thread::BLOCK_ANY, name_);

 priv_->waiters_count_lock_.lock();
  priv_->waiters_count_++;
 priv_->waiters_count_lock_.unlock();

  //release m and wait on sema until signal or broadcast called by other thread
  int err = SignalObjectAndWait(m.priv_->lock, priv_->sema_, waittime, FALSE);
  if (err != WAIT_OBJECT_0) {
    if (err == WAIT_TIMEOUT)
      success = false;
    else
      throw ThreadError(std::string("SignalObjectAndWait failed: ")
			+threadError());
  }

 priv_->waiters_count_lock_.lock();
  priv_->waiters_count_--;

  // check to see if we're the last waiter after broadcast
  int last_waiter = priv_->was_broadcast_ && priv_->waiters_count_ == 0;
 priv_->waiters_count_lock_.unlock();
  
  if (last_waiter) {
    // signal waiters_done and wait for mutex (to ensure fairness)
    err = SignalObjectAndWait(priv_->waiters_done_, m.priv_->lock, INFINITE, FALSE);
    if (err != WAIT_OBJECT_0)
      throw ThreadError(std::string("SignalObjectAndWait failed: ")
			+threadError());
  }
  else
    m.lock(); 
  Thread::pop_bstack(p, oldstate);

  
  return success;
}

void
ConditionVariable::conditionSignal()
{
  priv_->waiters_count_lock_.lock();
  int have_waiters = priv_->waiters_count_ > 0;
 priv_->waiters_count_lock_.unlock(); 
 
 if (have_waiters)
   ReleaseSemaphore(priv_->sema_, 1, 0);
}

void
ConditionVariable::conditionBroadcast()
{
  priv_->waiters_count_lock_.lock();
  int have_waiters = 0;
  
  if (priv_->waiters_count_ > 0) {
    // broadcast
    priv_->was_broadcast_ = 1;
    have_waiters = 1;
  }

  if (have_waiters) {
    // wake up waiting threads
    ReleaseSemaphore(priv_->sema_, priv_->waiters_count_, 0);
    
    priv_->waiters_count_lock_.unlock();
    
    // wait for threads to acquire semaphore
    WaitForSingleObject(priv_->waiters_done_, INFINITE);
    priv_->was_broadcast_ = 0;
  }
  else 
    priv_->waiters_count_lock_.unlock();
}

static void lock_scheduler()
{
	if(WaitForSingleObject(sched_lock,INFINITE)!=WAIT_OBJECT_0) {
	  throw ThreadError(std::string("lock_scheduler failed")
			    +threadError());
	}
}

static void unlock_scheduler()
{
	if(!ReleaseMutex(sched_lock)) {
	  throw ThreadError(std::string("unlock_scheduler failed")
			    +threadError());
	}
}

/*
 * Handle an abort signal - like segv, bus error, etc.
 */
static
void
handle_abort_signals(int sig)
{
    Thread* self=Thread::self();
    signal(sig, SIG_DFL);

    const char* tname=self?self->getThreadName():"idle or main";
    char* signam=Core_Thread_signal_name(sig, 0);
    fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) caught signal %s\n", 7,7,7,tname, (int)GetCurrentThreadId(), signam);
    fflush(stderr);
    Thread::niceAbort();

    signal(sig, handle_abort_signals);
}

void
Thread::print_threads()
{
    FILE* fp=stderr;
    for(int i=0;i<numActive;i++){
	Thread_private* p=active[i];
	const char* tname=p->thread?p->thread->getThreadName():"???";
	fprintf(fp, " %d: %s (", p->threadid, tname);
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

LONG WINAPI exception_handler(LPEXCEPTION_POINTERS ptr)
{
  Thread* self=Thread::self();
  const char* tname=self?self->getThreadName():"idle or main";
  char* signam=Core_Thread_signal_name(SIGSEGV, 0);
  fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) caught signal %s\n", 7,7,7,tname, (int)GetCurrentThreadId(), signam);
  fflush(stderr);
  Thread::niceAbort(ptr->ContextRecord);
  return 0;
}

/*
 * Setup signals for the current thread
 */
static
void
install_signal_handlers()
{
    signal(SIGILL, handle_abort_signals);
    signal(SIGABRT, handle_abort_signals);
    signal(SIGTERM, handle_abort_signals);
    signal(SIGFPE, handle_abort_signals);
    signal(SIGSEGV, handle_abort_signals);
    signal(SIGBREAK, handle_abort_signals);
    signal(SIGINT, handle_abort_signals);

    // NT doesn't generate SIGILL, SIGTERM, or SIGSEGV, so this should catch them
    SetUnhandledExceptionFilter(exception_handler);
}

#if 0
static void exit_handler()
{
    if(exiting)
        return;

    // Wait forever...
	cerr << "waiting forever" << endl;
	HANDLE wait;
	wait = CreateSemaphore(0,0,10,0);
	if(WaitForSingleObject(wait,INFINITE)!=WAIT_OBJECT_0) {
	  throw ThreadError(std::string("WaitForSingleObject (exit_handler) failed")
			    +threadError());
	}
	cerr << "didn't wait forever!" << endl;
}
#endif
void Thread::initialize()
{
	// atexit() has a semantic bug in win32, so we block in main.cc instead.
    //atexit(exit_handler);

	sched_lock = CreateMutex(0,0,0);
	if (!sched_lock) {
	  throw ThreadError(std::string("CreateMutex (sched_lock) failed")
			    +threadError());
	}

    ThreadGroup::s_default_group=new ThreadGroup("default group", 0);
    Thread* mainthread=new Thread(ThreadGroup::s_default_group, "main");
    mainthread->priv_=new Thread_private;
    mainthread->priv_->thread=mainthread;
	  mainthread->priv_->threadid = GetCurrentThreadId();
    mainthread->priv_->state=RUNNING;
    mainthread->priv_->bstacksize=0;

    threadids[mainthread->priv_->threadid] = mainthread;

	cerr << "mainthread id = " << mainthread->priv_->threadid << endl;

	//thread_local = new ThreadLocalMemory;
	//thread_local->current_thread = mainthread;

        mainthread->priv_->done = CreateSemaphore(0,0,10,0);
	if (!mainthread->priv_->done) {
	  throw ThreadError(std::string("CreateSemaphore (done) failed")
			    +threadError());
	}

	mainthread->priv_->delete_ready = CreateSemaphore(0,0,10,0);
	if (!mainthread->priv_->delete_ready) {
	  throw ThreadError(std::string("CreateSemaphore (delete_ready) failed")
			    +threadError());
	}

	mainthread->priv_->main_sema = CreateSemaphore(0,2,MAXTHREADS,0);
	if (!mainthread->priv_->main_sema) {
	  throw ThreadError(std::string("CreateSemaphore (main_sema) failed")
			    +threadError());
	}

	mainthread->priv_->control_c_sema = CreateSemaphore(0,1,MAXTHREADS,0);
	if (!mainthread->priv_->control_c_sema) {
	  throw ThreadError(std::string("CreateSemaphore (control_c_sema) failed")
			    +threadError());
	}

    lock_scheduler();
    active[numActive]=mainthread->priv_;
    numActive++;
    unlock_scheduler();

//#if 0
    if(!getenv("THREAD_NO_CATCH_SIGNALS")) {
	install_signal_handlers();
    }
//#endif
    Thread::initialized = true;

}

void Thread_run(Thread* t)
{
  // in the try-except functionality, we have access to the entire stack trace in the except () clause
  // (it has access to the stack before executing the exception handler while it determines where to land
  // the exception
#ifdef _MSC_VER 
  __try {
#endif
    t->run_body();
#ifdef _MSC_VER
  } __except(Thread::niceAbort(((LPEXCEPTION_POINTERS)GetExceptionInformation())->ContextRecord), 
      EXCEPTION_CONTINUE_SEARCH) {

  }
#endif
}

void Thread::migrate(int proc)
{
    // Nothing for now...
}

void Thread_shutdown(Thread* thread, bool actually_exit)
{
    Thread_private* priv=thread->priv_;
    char name[256];
    strcpy(name, thread->threadname_);

    ReleaseSemaphore(priv->done, 1, 0);

    // Wait to be deleted...
	if (WaitForSingleObject(priv->delete_ready,INFINITE)!=WAIT_OBJECT_0) {
	  throw ThreadError(std::string("WaitForSingleObject (delete_ready) failed")
			    +threadError());
	}

	
	delete thread;

    // Allow this thread to run anywhere...
    if(thread->cpu_ != -1)
	thread->migrate(-1);

    lock_scheduler();
    //thread_local->current_thread = 0;
    priv->thread=0;
    CloseHandle(priv->done);
    CloseHandle(priv->delete_ready);

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

    // This can't be done in checkExit, because of a potential race
    // condition.
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

    if(priv->threadid == 0){
	priv->state=Thread::PROGRAM_EXIT;
	if (WaitForSingleObject(main_sema,INFINITE)!=WAIT_OBJECT_0) {
	  throw ThreadError(std::string("WaitForSingleObject (main_sema) failed")
			    +threadError());
	}
    }
    // get rid of the tls hack
    threadids.erase(GetCurrentThreadId());

    if (actually_exit)
      ExitThread(0);
}

unsigned long run_threads(void* priv_v)
{
    Thread_private* priv=(Thread_private*)priv_v;
    threadids[priv->threadid] = priv->thread;
    //thread_local = new ThreadLocalMemory;
    //thread_local->current_thread = priv->thread;
    priv->state=Thread::RUNNING;
    Thread_run(priv->thread);
    priv->state=Thread::SHUTDOWN;
    Thread_shutdown(priv->thread, false);
    return 0; // Never reached
}

// matches the callback from CreateThread, use the helper so we can friend it
unsigned long __stdcall start_threads(void* priv_v)
{
  return run_threads(priv_v);
}

Thread* Thread::self()
{
  //return thread_local->current_thread;
  int threadid = GetCurrentThreadId();
  map<int, Thread*>::iterator iter = threadids.find(threadid);
  if (iter == threadids.end()) {
    printf("Thread::self failed: This thread: %d\n", threadid);
    printf("Available threads: ");
    for (iter = threadids.begin(); iter != threadids.end(); iter++) {
      printf("%d ",iter->first);
    }
    printf("\n");
    return 0;
  }
  return iter->second;
}

void Thread::exitAll(int code)
{
    exiting=true;
    ::exit(code);
}

void Thread::exit()
{
  Thread* self = Thread::self();
  Thread_shutdown(self, true);
}

void Thread::os_start(bool stopped)
{
    if(!Thread::initialized)
	Thread_private::initialize();

    priv_=new Thread_private;

    priv_->done = CreateSemaphore(0,0,100,0);
	if (!priv_->done) {
	  throw ThreadError(std::string("CreateSemaphore (done) failed")
			    +threadError());
	}
	priv_->delete_ready = CreateSemaphore(0,0,100,0);
	if (!priv_->delete_ready) {
	  throw ThreadError(std::string("CreateSemaphore (delete_ready) failed")
			    +threadError());
	}

    priv_->state=STARTUP;
    priv_->bstacksize=0;
    priv_->thread=this;
    priv_->threadid=0;
    priv_->main_sema = main_sema;

    lock_scheduler();
	priv_->t = CreateThread(0,stacksize_,start_threads,priv_,(stopped?CREATE_SUSPENDED:0),(unsigned long*)&priv_->threadid);
  threadids[priv_->threadid] = this;
	if (!priv_->t) {
	  throw ThreadError(std::string("CreateThread failed")
			    +threadError());
	}
    active[numActive]=priv_;
    numActive++;
    unlock_scheduler();
}

void Thread::checkExit()
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

void Thread::detach()
{
  ReleaseSemaphore(priv_->delete_ready,1,NULL);
  detached_=true;
}

void Thread::stop()
{
  SuspendThread(priv_->t);
}

void Thread::resume()
{
  ResumeThread(priv_->t);
}

void Thread::join()
{
  // wait for thread to be done - get the handle and release it
  WaitForSingleObject(priv_->done,INFINITE);
  ReleaseSemaphore(priv_->done,1,NULL);
  ReleaseSemaphore(this->priv_->delete_ready,1,NULL);
}

void Thread::yield()
{
	Sleep(0);
}

int Thread::numProcessors()
{
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	return sysinfo.dwNumberOfProcessors;
}

int Thread::push_bstack(Thread_private* p, Thread::ThreadState state,
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

void Thread::pop_bstack(Thread_private* p, int oldstate)
{
    p->bstacksize--;
    p->state=(ThreadState)oldstate;
} 

void
Thread::set_affinity(int cpu)
{
}

void ThreadGroup::gangSchedule()
{
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


} // end namespace SCIRun


