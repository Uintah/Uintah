
/*
 *  TCLInit.cc: Initialize TCL stuff..
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Jul 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/TclInterface/MemStats.h>
#include <Core/TclInterface/DebugSettings.h>
#include <Core/TclInterface/ThreadStats.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCL.h>

namespace SCIRun {

void TCL::initialize()
{
    DebugSettings* debugs=scinew DebugSettings;
    debugs->init_tcl();
    MemStats* memstats=scinew MemStats;
    memstats->init_tcl();
    ThreadStats* threadstats=scinew ThreadStats;
    threadstats->init_tcl();
}

} // End namespace SCIRun

