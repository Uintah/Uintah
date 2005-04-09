 
/*
 *  TCLTask.h:  Handle TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_TCLTask_h
#define SCI_project_TCLTask_h 1

#include <Multitask/Task.h>

class TCLTask : public Task {
    int argc;
    char** argv;
protected:
    virtual int body(int);
public:
    TCLTask(int argc, char* argv[]);
    virtual ~TCLTask();
    static void lock();
    static void unlock();
};

#endif
