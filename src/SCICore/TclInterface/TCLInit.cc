
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

#include <TclInterface/MemStats.h>
#include <TclInterface/DebugSettings.h>
#include <TclInterface/ThreadStats.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCL.h>

namespace SCICore {
namespace TclInterface {

void TCL::initialize()
{
    DebugSettings* debugs=scinew DebugSettings;
    debugs->init_tcl();
    MemStats* memstats=scinew MemStats;
    memstats->init_tcl();
    ThreadStats* threadstats=scinew ThreadStats;
    threadstats->init_tcl();
}

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:16  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
