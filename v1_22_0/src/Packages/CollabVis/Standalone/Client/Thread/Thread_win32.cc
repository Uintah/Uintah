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
 *  Thread_win32.cc: win32 threads implementation of the thread library
 *
 *  Written by:
 *   Author: Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   Date: November 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/ThreadError.h>
#include <afxwin.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

using std::cerr;
using std::endl;

#define MAX(x,y) ((x>y)?x:y)
#define MIN(x,y) ((x<y)?x:y)


#define MAXBSTACK 10
#define MAXTHREADS 4000

namespace SCIRun {
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
	};

	struct Mutex_private {
	HANDLE lock;
	};

	struct Semaphore_private {
	HANDLE hSema;
	};

    }
}

struct ThreadLocalMemory {
    Thread* current_thread;
};

__declspec(thread) ThreadLocalMemory* thread_local;
bool exiting=false;

Mutex::Mutex(const char* name)
{
	priv_ = scinew Mutex_private;
	priv_->lock = CreateMutex(NULL,0,name);
	if (priv_->lock == 0)
	{
		int check = GetLastError();
		::exit(1);
	}

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
		cerr << "ERROR: Mutex::try_lock()" << endl;
		::exit(1);
	}
	return 0; // never happens
}

static Thread_private* active[MAXTHREADS];
static int numActive;
static bool initialized;
static HANDLE sched_lock;              
__declspec(dllexport) HANDLE main_sema;         
static HANDLE control_c_sema;

Semaphore::Semaphore(const char* name,int count)
{
	priv_ = scinew Semaphore_private;
	priv_->hSema = CreateSemaphore(NULL,count,MAX(10,MIN(2*count,100)),name);
	if (priv_->hSema == 0)
	{
		int check = GetLastError();
		::exit(1);
	}
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
			else if (check == WAIT_FAILED)
			{
				check = GetLastError();
				cerr << "Uh oh.  One of the WaitForSingleObject()'s failed\n" << endl;
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
		cerr << "ERROR: Semaphore::try_down()" << endl;
		::exit(1);
	}
	return 0; // never happens
}

void Semaphore::up(int inc)
{
	long count;
	ReleaseSemaphore(priv_->hSema,inc,&count);
}


static void lock_scheduler()
{
	if(WaitForSingleObject(sched_lock,INFINITE)!=WAIT_OBJECT_0) {
		cerr << "lock_scheduler failed" << endl;
		::exit(-1);
	}
}

static void unlock_scheduler()
{
	if(!ReleaseMutex(sched_lock)) {
		cerr << "unlock_scheduler failed" << endl;
		::exit(-1);
	}
}

static void exit_handler()
{
	cerr << "main thread has entered exit_handler" << endl;
    if(exiting)
        return;

    // Wait forever...
	cerr << "waiting forever" << endl;
	HANDLE wait;
	wait = CreateSemaphore(0,0,10,"wait");
	if(WaitForSingleObject(wait,INFINITE)!=WAIT_OBJECT_0) {
		cerr << "WaitForSingleObject() failed for semaphore named wait" << endl;
		::exit(-10);
	}
	cerr << "didn't wait forever!" << endl;
}

void Thread::initialize()
{
	// atexit() has a semantic bug in win32, so we block in main.cc instead.
    //atexit(exit_handler);

	sched_lock = CreateMutex(0,0,"sched_lock");
	if (!sched_lock) {
		cerr << "unable to create mutex named sched_lock" << endl;
		::exit(-1);
	}

    ThreadGroup::s_default_group=new ThreadGroup("default group", 0);
    Thread* mainthread=new Thread(ThreadGroup::s_default_group, "main");
    mainthread->priv_=new Thread_private;
    mainthread->priv_->thread=mainthread;
	mainthread->priv_->threadid = GetCurrentThreadId();
    mainthread->priv_->state=RUNNING;
    mainthread->priv_->bstacksize=0;

	cerr << "mainthread id = " << mainthread->priv_->threadid << endl;

	thread_local = new ThreadLocalMemory;
	thread_local->current_thread = mainthread;

	mainthread->priv_->done = CreateSemaphore(0,0,10,"done");
	if (!mainthread->priv_->done) {
		cerr << "unable to create semaphore name done" << endl;
		::exit(-1);
	}

	mainthread->priv_->delete_ready = CreateSemaphore(0,0,10,"delete_ready");
	if (!mainthread->priv_->delete_ready) {
		cerr << "unable to create semaphore name delete_ready" << endl;
		::exit(-1);
	}

	mainthread->priv_->main_sema = CreateSemaphore(0,2,MAXTHREADS,"main_sema");
	if (!mainthread->priv_->main_sema) {
		cerr << "unable to create semaphore name main_sema" << endl;
		::exit(-1);
	}

	mainthread->priv_->control_c_sema = CreateSemaphore(0,1,MAXTHREADS,"control_c_sema");
	if (!mainthread->priv_->control_c_sema) {
		cerr << "unable to create semaphore name control_c_sema" << endl;
		::exit(-1);
	}

    lock_scheduler();
    active[numActive]=mainthread->priv_;
    numActive++;
    unlock_scheduler();

#if 0
    if(!getenv("THREAD_NO_CATCH_SIGNALS"))
	install_signal_handlers();
#endif

    initialized=true;
}

void Thread_run(Thread* t)
{
    t->run_body();
}

void Thread::migrate(int proc)
{
    // Nothing for now...
}

static void Thread_shutdown(Thread* thread)
{
    Thread_private* priv=thread->priv_;

	if (WaitForSingleObject(priv->done,INFINITE)!=WAIT_OBJECT_0) {
		cerr << "WaitForSingleObject() failed on semaphore named done" << endl;
		::exit(-1);
	}

    delete thread;

    // Wait to be deleted...
	if (WaitForSingleObject(priv->delete_ready,INFINITE)!=WAIT_OBJECT_0) {
		cerr << "WaitForSingleObject() failed on semaphore named delete_ready" << endl;
		::exit(-1);
	}

    // Allow this thread to run anywhere...
    if(thread->cpu_ != -1)
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
	if (WaitForSingleObject(main_sema,INFINITE)!=WAIT_OBJECT_0) {
		cerr << "WaitForSingleObject() failed on semaphore named main_sema" << endl;
		::exit(-1);
	}
	}
    ::exit(0);
}

unsigned long __stdcall run_threads(void* priv_v)
{
    Thread_private* priv=(Thread_private*)priv_v;
	thread_local = new ThreadLocalMemory;
	thread_local->current_thread = priv->thread;
    priv->state=Thread::RUNNING;
    Thread_run(priv->thread);
    priv->state=Thread::SHUTDOWN;
    Thread_shutdown(priv->thread);
    return 0; // Never reached
}

Thread* Thread::self()
{
    return thread_local->current_thread;
}

void Thread::exitAll(int code)
{
    exiting=true;
    ::exit(code);
}

void Thread::os_start(bool stopped)
{
    if(!initialized)
	Thread::initialize();

    priv_=new Thread_private;

	priv_->done = CreateSemaphore(0,0,100,"done");
	if (!priv_->done) {
		cerr << "CreateSemaphore failed" << endl;
		::exit(-1);
	}
	priv_->delete_ready = CreateSemaphore(0,0,100,"delete_ready");
	if (!priv_->delete_ready) {
		cerr << "CreateSemaphore failed" << endl;
		::exit(-1);
	}

    priv_->state=STARTUP;
    priv_->bstacksize=0;
    priv_->thread=this;
    priv_->threadid=0;
	priv_->main_sema = main_sema;

    lock_scheduler();
	priv_->t = CreateThread(0,0,run_threads,priv_,(stopped?CREATE_SUSPENDED:0),(unsigned long*)&priv_->threadid);
	if (!priv_->t) {
		cerr << "CreateSemaphore failed" << endl;
		::exit(-1);
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
	long last;
	ReleaseSemaphore(priv_->delete_ready,1,&last);
    detached_=true;
#if 0
    if(pthread_detach(priv_->threadid) != 0)
	throw ThreadError(std::string("pthread_detach failed")
			  +strerror(errno));
#endif
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
	WaitForSingleObject(priv_->t,INFINITE);
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
} // End namespace SCIRun

void ThreadGroup::gangSchedule()
{





