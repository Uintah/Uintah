//static char *id="@(#) $Id$

/*
 *  scirun.cc: Main program for project
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
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/ModuleList.h>
#include <Dataflow/Network.h>
#include <Dataflow/NetworkEditor.h>
#include <Distributed/SlaveController.h>
#include <Multitask/Task.h>
#include <TclInterface/GuiServer.h>
#include <TclInterface/GuiManager.h>
#include <TclInterface/TCLTask.h>
#include <TclInterface/TCL.h>

#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

using SCICore::Multitask::Task;
using SCICore::TclInterface::TCLTask;
using SCICore::TclInterface::GuiManager;
using SCICore::TclInterface::GuiServer;

using PSECommon::Dataflow::Network;
using PSECommon::Dataflow::NetworkEditor;
using PSECommon::Distributed::SlaveController;

namespace PSECommon {
  namespace Dataflow {
    extern bool global_remote;
  }
}

namespace SCICore {
  namespace TclInterface {
    void set_guiManager( GuiManager * );
  }
}

//extern void set_guiManager (GuiManager*);

int global_argc;
char** global_argv;

// master creates slave by rsh "sr -slave hostname portnumber"
int main(int argc, char** argv)
{
    // Initialize the multithreader
    Task::initialize(argv[0]);
    global_argc=argc;
    global_argv=argv;

    // Start up TCL...
    TCLTask* tcl_task = new TCLTask(argc, argv);
    tcl_task->activate(0);
    tcl_task->mainloop_waitstart();

    // Create initial network
    // We build the Network with a 1, indicating that this is the
    // first instantiation of the network class, and this network
    // should read the command line specified files (if any)
    Network* net=new Network(1);

    // Fork off task for the network editor.  It is a detached
    // task, and the Task* will be deleted by the task manager

    // if slave, then create daemon, not scheduler
    if (argc > 1 && strcmp (argv[1], "-slave") == 0) {
	PSECommon::Dataflow::global_remote = true;
	SlaveController* remote_task = new SlaveController(net,argv[2],argv[3]);
	remote_task->activate(0);  

	// startup slave-side GuiManager client
	SCICore::TclInterface::set_guiManager (new GuiManager(argv[2], argv[3]));
    }
    else {
     	NetworkEditor* gui_task=new NetworkEditor(net);

    	// Activate the network editor and scheduler.  Arguments and return
    	// values are meaningless
    	gui_task->activate(0);

 	// startup master-side GuiServer, even when no slave..
    	GuiServer* gui_server = new GuiServer;
    	gui_server->activate(0);
    }

    // Now activate the TCL event loop
    tcl_task->release_mainloop();

    // This will wait until all tasks have completed before exiting
    Task::main_exit();

    // Never reached
    return 0;
}

//
// $Log$
// Revision 1.1  1999/07/27 16:58:46  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 22:25:36  dav
// trying to update all
//
//
