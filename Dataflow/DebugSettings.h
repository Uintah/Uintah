
/*
 *  DebugSettings.h: Debug settings visualizer
 *
 *  Written by:
 *   James T. Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_DebugSettings_h
#define SCI_project_DebugSettings_h 1

#include <TCL/TCL.h>

class DebugSettings : public TCL {
public:
    DebugSettings();
    ~DebugSettings();

    void init_tcl();
    virtual void tcl_command(TCLArgs&, void*);
};

#endif
