
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

#include <Core/TclInterface/TCL.h>

namespace SCIRun {

class SCICORESHARE ThreadStats : public TCL {
    int maxstacksize;
public:
    ThreadStats();
    ~ThreadStats();

    void init_tcl();
    virtual void tcl_command(TCLArgs&, void*);
};

} // End namespace SCIRun


#endif
