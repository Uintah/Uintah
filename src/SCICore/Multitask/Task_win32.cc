
/*
 *  Task_win32.cc:  Task implementation for win32
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1996
 *
 *  Modified for win32 by Chris Moulding 12/98
 *
 *  Copyright (C) 1996 SCI Group
 */


#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <stdio.h>
#include <string.h>
#define _WIN32_WINNT 0x400
#include <windows.h>
#include <winbase.h>
#include <process.h>
#undef _WIN32_WINNT

#define DEFAULT_STACK_LENGTH 64*1024
#define INITIAL_STACK_LENGTH 48*1024
#define DEFAULT_SIGNAL_STACK_LENGTH 16*1024

#define NOT_FINISHED(what) cerr << what << ": Not Finished " << __FILE__ << " (line " << __LINE__ << ") " << endl

extern "C" int Task_try_lock(unsigned long*);

namespace SCICore {
namespace Multitask {

static int aborting=0;
static int exit_code;
static int exiting=0;
static char* progname;
static __declspec(thread) void* selfkey;

struct TaskPrivate
{
	HANDLE hThread;
};

TaskPrivate bob;

#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x<y)?y:x)

struct TaskArgs {
    int arg;
    Task* t;
};

class MainTask : public Task {
public:
    MainTask();
    virtual ~MainTask();
    int body(int);
};


#define MAXTASKS 1000
static int ntasks=0;
static Task* tasks[MAXTASKS];
Mutex* sched_lock;


unsigned __stdcall runbody( void* vargs )
{
    TaskArgs* args=(TaskArgs*)vargs;
    int arg=args->arg;
    Task* t=args->t;
    delete args;

    t->startup(arg);
    return 0;
}

int Task::startup(int task_arg)
{
	selfkey = (void*)this;

    int retval=body(task_arg);
    Task::taskexit(this, retval);
    return 0; // Never reached.
}

// We are done..
void Task::taskexit(Task*, int retval)
{
	ExitThread(retval); 
} 

// Fork a thread
void Task::activate(int task_arg)
{
    if(activated){
	cerr << "Error: task is being activated twice!" << endl;
	Task::exit_all(-1);
    }

    activated=1;
    priv=scinew TaskPrivate;
    TaskArgs* args=scinew TaskArgs;
    args->arg=task_arg;
    args->t=this;
	
	sched_lock->lock();
	
	unsigned int threadid;
	priv->hThread = (HANDLE)_beginthreadex(NULL, DEFAULT_STACK_LENGTH, runbody, (void*)args, NULL, &threadid);
	printf("threadID %d matches thread name %s\n",threadid,name);
	if (strcmp(name,"TCLTask")==0) 
	  SetThreadPriority(priv->hThread,THREAD_PRIORITY_ABOVE_NORMAL);
	if (priv->hThread == 0)
	{
		int check = GetLastError();
		exit(1);
	}
    tasks[ntasks++]=this;

    sched_lock->unlock();
}

Task* Task::self()
{
    Task* t=(Task*)selfkey;
    return t;
}


void Task::exit_all(int)
{
    fprintf(stderr, "Task::exit_all not done!\n");
    exit(1);
}


void Task::initialize(char* pn)
{
    progname=strdup(pn);

	selfkey = NULL;

    sched_lock=new Mutex;

    Task* maintask=scinew MainTask;
    maintask->activated=1;
    maintask->priv=scinew TaskPrivate;

	selfkey = (void*)maintask;
}

int Task::nprocessors()
{
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	return sysinfo.dwNumberOfProcessors;
}

void Task::main_exit()
{
    ExitThread(0);
}


void Task::yield()
{
	Sleep(0);
}

void Task::cancel_itimer(int which_timer)
{
}

int Task::start_itimer(const TaskTime& start, const TaskTime& interval,
                       void (*handler)(void*), void* cbdata)
{
	return 0;
}

//
// Semaphore implementation
//
struct Semaphore_private {
	HANDLE hSema;
};



Semaphore::Semaphore(int count)
{
	priv = scinew Semaphore_private;
	priv->hSema = CreateSemaphore(NULL,count,MAX(10,MIN(2*count,100)),NULL);
	if (priv->hSema == 0)
	{
		int check = GetLastError();
		exit(1);
	}
}

Semaphore::~Semaphore()
{
	CloseHandle(priv->hSema);
	delete priv;
}

void Semaphore::down()
{
	int check = WaitForSingleObject(priv->hSema,INFINITE);
	if (check != WAIT_OBJECT_0)
	{
		if (check == WAIT_ABANDONED);
		else if (check == WAIT_TIMEOUT);
		else if (check == WAIT_FAILED)
		{
			check = GetLastError();
		}
		else;
	}
}

int Semaphore::try_down()
{
	int check = WaitForSingleObject(priv->hSema,0);
	if (check == WAIT_OBJECT_0)
		return 0;
	else if (check == WAIT_TIMEOUT)
		return 1;
	else 
	{
		cerr << "ERROR: Semaphore::try_down()" << endl;
		exit(1);
	}
	return 0; // never happens
}

void Semaphore::up()
{
	long count;
	ReleaseSemaphore(priv->hSema,1,&count);
}

struct Mutex_private {
    HANDLE lock;
};

Mutex::Mutex()
{
	priv = scinew Mutex_private;
	priv->lock = CreateMutex(NULL,0,NULL);
	if (priv->lock == 0)
	{
		int check = GetLastError();
		exit(1);
	}
}

Mutex::~Mutex()
{
	CloseHandle(priv->lock);
	delete priv;
}

void Mutex::lock()
{
	WaitForSingleObject(priv->lock,INFINITE);
}

void Mutex::unlock()
{
	ReleaseMutex(priv->lock);
}

int Mutex::try_lock()
{
	int check = WaitForSingleObject(priv->lock,0);
	if (check == WAIT_OBJECT_0)
		return 1;
	else if (check == WAIT_TIMEOUT)
		return 0;
	else 
	{
		cerr << "ERROR: Mutex::try_lock()" << endl;
		exit(1);
	}
	return 0; // never happens
}

//
// Condition variable implementation
//

#define SCHMIDT_COND 0

#if SCHMIDT_COND

typedef struct
{
  int waiters_count_;

  CRITICAL_SECTION waiters_count_lock_;

  HANDLE sema_;

  HANDLE waiters_done_;

  size_t was_broadcast_;

} pthread_cond_t;


typedef HANDLE pthread_mutex_t;
typedef void pthread_condattr_t;

int pthread_cond_init (pthread_cond_t *cv, const pthread_condattr_t *){  
  cv->waiters_count_ = 0;
  cv->was_broadcast_ = 0;
  cv->sema_ = CreateSemaphore (NULL,       // no security
                                0,          // initially 0
                                0x7fffffff, // max count
                                NULL);      // unnamed 
  InitializeCriticalSection (&cv->waiters_count_lock_);
  cv->waiters_done_ = CreateEvent (NULL,  // no security
                                   FALSE, // auto-reset
                                   FALSE, // non-signaled initially
                                   NULL); // unnamed
  return 0;
}

int pthread_cond_wait (pthread_cond_t *cv, pthread_mutex_t *external_mutex){  
  EnterCriticalSection (&cv->waiters_count_lock_);  
  cv->waiters_count_++;
  LeaveCriticalSection (&cv->waiters_count_lock_);
  SignalObjectAndWait (*external_mutex, cv->sema_, INFINITE, FALSE);
  EnterCriticalSection (&cv->waiters_count_lock_);
  int last_waiter = cv->was_broadcast_ && cv->waiters_count_ == 0;
  LeaveCriticalSection (&cv->waiters_count_lock_);
  if (last_waiter)
    SignalObjectAndWait (cv->waiters_done_, *external_mutex, INFINITE, FALSE);
  else       
	WaitForSingleObject (*external_mutex,INFINITE);
  return 0;
}

int pthread_cond_signal (pthread_cond_t *cv){
  EnterCriticalSection (&cv->waiters_count_lock_);
  int have_waiters = cv->waiters_count_ > 0;
  LeaveCriticalSection (&cv->waiters_count_lock_);
  if (have_waiters)
    ReleaseSemaphore (cv->sema_, 1, 0);
  return 0;
}

int pthread_cond_broadcast (pthread_cond_t *cv){
  EnterCriticalSection (&cv->waiters_count_lock_);  
  int have_waiters = 0;
  if (cv->waiters_count_ > 0) {
    cv->was_broadcast_ = 1;    
	have_waiters = 1;  
  }  
  if (have_waiters) {
    ReleaseSemaphore (cv->sema_, cv->waiters_count_, 0);
    LeaveCriticalSection (&cv->waiters_count_lock_);
    cv->was_broadcast_ = 0;  
  }  
  else
    LeaveCriticalSection (&cv->waiters_count_lock_);
  return 0;
}

void pthread_cond_destroy(pthread_cond_t* cv)
{
	CloseHandle(cv->sema_);
	CloseHandle(cv->waiters_done_);
	DeleteCriticalSection(&(cv->waiters_count_lock_));
}

//
// Condition variable implementation
//
struct ConditionVariable_private {
    pthread_cond_t cond;
};

ConditionVariable::ConditionVariable()
{
    priv=scinew ConditionVariable_private;
    pthread_cond_init(&priv->cond, NULL);
}

ConditionVariable::~ConditionVariable()
{
    if(priv){
	pthread_cond_destroy(&priv->cond);
	delete priv;
    }
}

void ConditionVariable::wait(Mutex& mutex)
{
    if(pthread_cond_wait(&priv->cond, &mutex.priv->lock) != 0)
	perror("pthread_cond_wait");
}

void ConditionVariable::cond_signal()
{
    if(pthread_cond_signal(&priv->cond) != 0)
	perror("pthread_cond_signal");
}

void ConditionVariable::broadcast()
{
    if(pthread_cond_broadcast(&priv->cond) != 0)
	perror("pthread_cond_broadcast");
}


#else // SCHMIDT_COND

class ConditionVariable_private {
public:
    ConditionVariable_private();
    int nwaiters;
    Mutex mutex;
    Semaphore semaphore;
};

ConditionVariable_private::ConditionVariable_private()
: semaphore(0), nwaiters(0)
{
}

ConditionVariable::ConditionVariable()
{
    priv=scinew ConditionVariable_private;
}

ConditionVariable::~ConditionVariable()
{
    if(priv){
	delete priv;
    }
}

void ConditionVariable::wait(Mutex& mutex)
{
    priv->mutex.lock();
    priv->nwaiters++;
    priv->mutex.unlock();
    mutex.unlock();
    // Block until woken up by signal or broadcast
    priv->semaphore.down();
    mutex.lock();
}

void ConditionVariable::cond_signal()
{
    priv->mutex.lock();
    if(priv->nwaiters > 0){
	priv->nwaiters--;
	priv->semaphore.up();
    }
    priv->mutex.unlock();
}

void ConditionVariable::broadcast()
{
    priv->mutex.lock();
    while(priv->nwaiters > 0){
	priv->nwaiters--;
	priv->semaphore.up();
    }
    priv->mutex.unlock();
}

#endif // SCHMIDT_COND

struct Barrier_private {
    Mutex lock;
    ConditionVariable cond;
    int count;
};

Barrier::Barrier()
{
    priv=new Barrier_private;
    priv->count=0;
}

Barrier::~Barrier()
{
	delete priv;
}

void Barrier::wait(int n)
{
    priv->lock.lock();
    int orig_count=priv->count++;
    if(priv->count==n){
        priv->count=0; // Reset the counter...
        priv->lock.unlock();
        priv->cond.broadcast();
    } else {
        priv->cond.wait(priv->lock);
	priv->lock.unlock();
    }
}

MainTask::MainTask()
: Task("main()", 1)
{
}

MainTask::~MainTask()
{
}

int MainTask::body(int)
{
    cerr << "Error - main's body called!\n";
    exit(-1);
    return 0;
}

} // end of namespace Multitask
} // end of namespace SCICore

