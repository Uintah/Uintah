
/*
 *  SimpleTask.h:  Convenient way to generate a simple class for a task
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

// No include guards - this can by done more than once...

#include <Multitask/Task.h>

// To use this, define SIMPLE_TASK_NAME to the name of the class that
// you want to build, and #include this header.  The class definition
// for a simple task type will be generated automatically.

#ifndef SIMPLE_TASK_NAME
#error Must define SIMPLE_TASK_NAME before including SimpleTask
#endif

class SIMPLE_TASK_NAME : public Task {
public:
    inline SIMPLE_TASK_NAME(const clString& n, int detached=1, int prio=Task::DEFAULT_PRIORITY,
			    Processor* wh=0)
    	: Task(n, detached, prio, wh) { }
    virutal ~SIMPLE_TASK_NAME();
    virtual int body(int);
};

SIMPLE_TASK_NAME::~SIMPLE_TASK_NAME()
{
}
