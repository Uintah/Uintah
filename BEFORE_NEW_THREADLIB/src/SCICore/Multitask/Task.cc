//static char *id="@(#) $Id$";

/*
 *  Task.cc: Implementation of architecture independant parts of Task library
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Multitask/Task.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Multitask/ITC.h>
#include <iostream.h>

namespace SCICore {
namespace Multitask {

// Task constructure
Task::Task(const char* name, int detached, int priority)
: name(name), priority(priority), detached(detached), timers(0),
  timer_id(100), ntimers(0)
{
    activated=0;
}

Task::~Task()
{
}

const char* Task::get_name()
{
    return name;
}

TaskTime::TaskTime(int secs, int usecs)
: secs(secs), usecs(usecs)
{
}

TaskTime::TaskTime(float secs)
: secs((int)secs), usecs((int)((secs-(int)secs)*1000.))
{
}

TaskTime::TaskTime(double secs)
: secs((int)secs), usecs((int)((secs-(int)secs)*1000.))
{
}

TaskTime::TaskTime()
{
}

TaskInfo::TaskInfo(int ntasks)
: ntasks(ntasks), tinfo(scinew Info[ntasks])
{
}

TaskInfo::~TaskInfo()
{
    delete[] tinfo;
}

class SCICORESHARE Multiprocess : public Task {
public:
  Multiprocess* next;
  Semaphore sema1;
  Semaphore sema2;
  void (*starter)(void*, int);
  void* userdata;
  int processor;
  Multiprocess();
  virtual int body(int);
};

Multiprocess::Multiprocess()
: Task("Multiprocess helper"), sema1(0), sema2(0)
{
}

int Multiprocess::body(int)
{
    for(;;){
	sema1.down();
	(*starter)(userdata, processor);
	sema2.up();
    }
}

static Multiprocess* mpworkers;
static Mutex workerlock;

#ifdef __sgi
  #include <sys/sysmp.h>
#elif defined(_WIN32)
  #define _WIN32_WINNT 0x0400  
  #include <windows.h>
  #include <stdio.h>
  #include <SCICore/share/share.h>
  #define PTHREAD HANDLE
  #undef _WIN32_WINNT
#endif

void Task::multiprocess(int nprocessors, void (*starter)(void*, int),
			void* userdata, bool block)
{
	int i;
    if(!block){
	for(i=0;i<nprocessors;i++){
	    Multiprocess* w=new Multiprocess;
	    w->activate(0);
	    w->starter=starter;
	    w->userdata=userdata;
	    w->processor=i;
	    w->sema1.up();
#ifdef _WIN32
		SetThreadIdealProcessor(GetCurrentThread(),i);
#elif defined __sgi
	    sysmp(MP_MUSTRUN, i);
#endif
	    w->next=0;
	}
	return;
    }
    workerlock.lock();

    Multiprocess* workers=0;
    for(i=0;i<nprocessors;i++){
      Multiprocess* w=mpworkers;
      if(!w){
	w=new Multiprocess;
	w->activate(0);
      } else {
	mpworkers=w->next;
      }
      w->next=workers;
      workers=w;
    }
    workerlock.unlock();

    Multiprocess* w=workers;
    for(i=0;i<nprocessors;i++){
      w->starter=starter;
      w->userdata=userdata;
      w->processor=i;
      w->sema1.up();
#ifdef _WIN32
	  SetThreadIdealProcessor(GetCurrentThread(),i);
#elif defined __sgi
      sysmp(MP_MUSTRUN, i);
#endif
      w=w->next;
    }

    w=workers;
    for(i=0;i<nprocessors;i++){
      w->sema2.down();
      w=w->next;
    }

    workerlock.lock();
    mpworkers=workers;
    workerlock.unlock();
}

} // End namespace Multitask
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:38  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:07  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:11:01  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//
