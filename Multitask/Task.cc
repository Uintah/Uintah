
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

#include <Multitask/Task.h>
#include <Malloc/Allocator.h>
#include <Multitask/ITC.h>
#include <iostream.h>

// Task constructure
Task::Task(char* name, int detached, int priority)
: name(name), priority(priority), detached(detached), timers(0),
  timer_id(100), ntimers(0)
{
    activated=0;
}

Task::~Task()
{
}

char* Task::get_name()
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

class Multiprocess : public Task {
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
  while(1){
    sema1.down();
    (*starter)(userdata, processor);
    sema2.up();
  }
  return 0;
}

static Multiprocess* mpworkers;
static Mutex workerlock;

#include <sys/sysmp.h>

void Task::multiprocess(int nprocessors, void (*starter)(void*, int),
			void* userdata, bool block)
{
    if(!block){
	for(int i=0;i<nprocessors;i++){
	    Multiprocess* w=new Multiprocess;
	    w->activate(0);
	    w->starter=starter;
	    w->userdata=userdata;
	    w->processor=i;
	    w->sema1.up();
	    sysmp(MP_MUSTRUN, i);
	    w->next=0;
	}
	return;
    }
    workerlock.lock();

    Multiprocess* workers=0;
    Multiprocess* lastw=0;
    for(int i=0;i<nprocessors;i++){
      Multiprocess* w=mpworkers;
      if(!w){
	w=new Multiprocess;
	w->activate(0);
      } else {
	mpworkers=w->next;
      }
      w->next=workers;
      workers=w;
      if(i==0)
	lastw=w;
    }
    workerlock.unlock();

    Multiprocess* w=workers;
    for(i=0;i<nprocessors;i++){
      w->starter=starter;
      w->userdata=userdata;
      w->processor=i;
      w->sema1.up();
      sysmp(MP_MUSTRUN, i);
      w=w->next;
    }

    w=workers;
    for(i=0;i<nprocessors;i++){
      w->sema2.down();
      w=w->next;
    }

    workerlock.lock();
    lastw=mpworkers;
    mpworkers=workers;
    workerlock.unlock();
}

