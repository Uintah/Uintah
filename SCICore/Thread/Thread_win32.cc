
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

#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/ThreadGroup.h>

using SCICore::Thread::Mutex;
using SCICore::Thread::Semaphore;
using SCICore::Thread::Thread;
using SCICore::Thread::ThreadGroup;

#define MAXBSTACK 10

namespace SCICore {
    namespace Thread {
	struct Thread_private {
	    Thread* thread;
	    int threadid;
	    Thread::ThreadState state;
	    int bstacksize;
	    const char* blockstack[MAXBSTACK];
	    Semaphore done;
	    Semaphore delete_ready;
	    Semaphore block_sema;
	};
    }
}

Mutex::Mutex(const char* blah)
{
}

Mutex::~Mutex()
{
}

void Mutex::unlock()
{
}

void Mutex::lock()
{
}

bool Mutex::tryLock()
{
	return false;
}

Semaphore::Semaphore(const char* blah1,int blah2)
{
}

Semaphore::~Semaphore()
{
}

void Semaphore::down(int blah)
{
}

void Semaphore::up(int blah)
{
}

Thread* Thread::self()
{
	return 0;
}

void Thread::exitAll(int blah)
{
}

void Thread::os_start(bool blah)
{
}

void Thread::initialize()
{
}

void Thread::checkExit()
{
}

int Thread::push_bstack(SCICore::Thread::Thread_private* blah1,ThreadState blah2,const char* blah3)
{
	return 0;
}

void Thread::pop_bstack(SCICore::Thread::Thread_private* blah1,int blah2)
{
}

void Thread::stop()
{
}

void Thread::resume()
{
}

void Thread::join()
{
}

void Thread::detach()
{
}

void ThreadGroup::gangSchedule()
{
}






