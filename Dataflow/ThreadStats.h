
/*
 *  ThreadStats.h: Thread information visualizer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Jul 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ThreadStats_h
#define SCI_project_ThreadStats_h 1

#include <TCL.h>
class TaskInfo;

class ThreadStats : public TCL {
    TaskInfo* info;
    TaskInfo* oldinfo;
    int maxstacksize;
public:
    ThreadStats();
    ~ThreadStats();

    void init_tcl();
    virtual void tcl_command(TCLArgs&, void*);
};

#endif
