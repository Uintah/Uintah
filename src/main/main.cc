//static char *id="@(#) $Id$";

/*
 *  uintah.cc: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Modified for distributed SCIRun by:
 *   Michelle Miller
 *   May 1998
 *
 *  Copyright (C) 1999 U of U
 */

#include <PSECore/Dataflow/Network.h>
#include <PSECore/Dataflow/NetworkEditor.h>
//#include <PSECore/Distributed/SlaveController.h>
#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/TclInterface/GuiServer.h>
#include <SCICore/TclInterface/GuiManager.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Containers/String.h>

#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

using SCICore::TclInterface::TCLTask;
using SCICore::TclInterface::GuiManager;
using SCICore::TclInterface::GuiServer;
using SCICore::Thread::Thread;
using namespace SCICore::Containers;
using namespace PSECore::Dataflow;

using PSECore::Dataflow::Network;
using PSECore::Dataflow::NetworkEditor;

namespace PSECore {
  namespace Dataflow {
    extern bool global_remote;
  }
}

namespace SCICore {
  namespace TclInterface {
    void set_guiManager( GuiManager * );
  }
}

//extern void SCICore::TclInterface::set_guiManager( GuiManager* );

int global_argc;
char** global_argv;

#ifndef PSECORETCL
#error You must set PSECORETCL to the PSECore/Tcl path
#endif

#ifndef SCICORETCL
#error You must set SCICORETCL to the SCICore/Tcl path
#endif

#ifndef DEFAULT_PACKAGE_PATH
#error You must set a DEFAULT_PACKAGE_PATH or life is pretty dull
#endif

// master creates slave by rsh "sr -slave hostname portnumber"
int main(int argc, char** argv)
{
    global_argc=argc;
    global_argv=argv;

    // Start up TCL...
    TCLTask* tcl_task = new TCLTask(argc, argv);
    Thread* t=new Thread(tcl_task, "TCL main event loop");
    t->detach();
    tcl_task->mainloop_waitstart();

    // Set up the TCL environment to find core components
    clString result;
    TCL::eval("global PSECoreTCL SCICoreTCL",result);
    TCL::eval("set PSECoreTCL "PSECORETCL,result);
    TCL::eval("set SCICoreTCL "SCICORETCL,result);
    TCL::eval("lappend auto_path "SCICORETCL,result);
    TCL::eval("lappend auto_path "PSECORETCL,result);

    // Create initial network
    // We build the Network with a 1, indicating that this is the
    // first instantiation of the network class, and this network
    // should read the command line specified files (if any)
    Network* net=new Network(1);

    // Fork off task for the network editor.  It is a detached
    // task, and the Task* will be deleted by the task manager
    NetworkEditor* gui_task=new NetworkEditor(net);

    // Load in the default packages
    if(getenv("PACKAGE_PATH")!=NULL)
      packageDB.loadPackage(getenv("PACKAGE_PATH"));
    else
      packageDB.loadPackage(DEFAULT_PACKAGE_PATH);

    // Activate the network editor and scheduler.  Arguments and return
    // values are meaningless
    Thread* t2=new Thread(gui_task, "Scheduler");
    t2->setDaemon(true);
    t2->detach();

    // startup master-side GuiServer, even when no slave..
    // XXX: Is this a remnant of dist'd stuff?  Michelle?
    GuiServer* gui_server = new GuiServer;
    Thread* t3=new Thread(gui_server, "GUI server thread");
    t3->setDaemon(true);
    t3->detach();

    // Now activate the TCL event loop
    tcl_task->release_mainloop();

    // Never reached
    return 0;
}

//
// $Log$
// Revision 1.4  1999/08/28 17:54:54  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/17 06:40:15  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:37  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//
