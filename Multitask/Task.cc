
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
#include <iostream.h>

// Task constructure
Task::Task(const clString& name, int detached, int priority)
: name(name), priority(priority), detached(detached)
{
    activated=0;
}

Task::~Task()
{
}

clString Task::get_name()
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

TaskInfo::TaskInfo(int ntasks)
: ntasks(ntasks), tinfo(new Info[ntasks])
{
}

TaskInfo::~TaskInfo()
{
    delete[] tinfo;
}

