
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

#include <Dataflow/MemStats.h>
#include <Dataflow/DebugSettings.h>
#include <Dataflow/NetworkEditor.h>
#include <Dataflow/ThreadStats.h>
#include <Malloc/Allocator.h>
#include <TCL/TCL.h>

void TCL::initialize()
{
    DebugSettings* debugs=scinew DebugSettings;
    debugs->init_tcl();
    MemStats* memstats=scinew MemStats;
    memstats->init_tcl();
    ThreadStats* threadstats=scinew ThreadStats;
    threadstats->init_tcl();
}
