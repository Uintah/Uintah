
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
    void (*starter)(void*, int);
    void* userdata;
    int processor;
    Multiprocess(void (*starter)(void*, int), void* userdata, int processor);
    virtual int body(int);
};

Multiprocess::Multiprocess(void (*starter)(void*, int), void* userdata, int processor)
: Task("Multiprocess helper"), 
  starter(starter), userdata(userdata), processor(processor)
{
}

int Multiprocess::body(int)
{
    cerr << "started: " << processor << endl;
    (*starter)(userdata, processor);
    return 0;
}

void Task::multiprocess(int nprocessors, void (*starter)(void*, int), void* userdata)
{
    for(int i=0;i<nprocessors;i++){
	Multiprocess* mp=new Multiprocess(starter, userdata, i);
	cerr << "starting: " << i << endl;
	mp->activate(0);
    }
}

