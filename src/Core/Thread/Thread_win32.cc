
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
#include <SCICore/Malloc/Allocator.h>
#include <afxwin.h>
#include <string.h>
#include <stdio.h>
#include <iostream.h>

#define MAX(x,y) ((x>y)?x:y)
#define MIN(x,y) ((x<y)?x:y)

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

	struct Mutex_private {
	HANDLE lock;
	};

	struct Semaphore_private {
	HANDLE hSema;
	};

    }
}


Mutex::Mutex(const char* name)
{
	d_priv = scinew Mutex_private;
	d_priv->lock = CreateMutex(NULL,0,NULL);
	if (d_priv->lock == 0)
	{
		int check = GetLastError();
		exit(1);
	}

	int length = strlen(name);
	d_name = new char[length+1];
	sprintf((char*)d_name,"%s",name);
}

Mutex::~Mutex()
{
	CloseHandle(d_priv->lock);
	delete[] (char*)d_name;
	delete d_priv;
}

void Mutex::lock()
{
	WaitForSingleObject(d_priv->lock,INFINITE);
}

void Mutex::unlock()
{
	ReleaseMutex(d_priv->lock);
}

bool Mutex::tryLock()
{
	int check = WaitForSingleObject(d_priv->lock,0);
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

Semaphore::Semaphore(const char* name,int count)
{
	d_priv = scinew Semaphore_private;
	d_priv->hSema = CreateSemaphore(NULL,count,MAX(10,MIN(2*count,100)),NULL);
	if (d_priv->hSema == 0)
	{
		int check = GetLastError();
		exit(1);
	}
	int length = strlen(name);
	d_name = new char[length+1];
	sprintf((char*)d_name,"%s",name);
}

Semaphore::~Semaphore()
{
	CloseHandle(d_priv->hSema);
	delete[] (char*)d_name;
	delete d_priv;
}

void Semaphore::down(int dec)
{
	int check;
	for (int loop = 0;loop<dec;loop++) {
		check = WaitForSingleObject(d_priv->hSema,INFINITE);
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
	int check = WaitForSingleObject(d_priv->hSema,0);
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

void Semaphore::up(int inc)
{
	long count;
	ReleaseSemaphore(d_priv->hSema,inc,&count);
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






