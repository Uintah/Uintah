
/*
 *  main.cc: Main program for project
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
#include <Dataflow/SlaveController.h>
#include <Multitask/Task.h>
#include <TCL/GuiServer.h>
#include <TCL/GuiManager.h>
#include <TCL/TCLTask.h>

#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

extern void set_guiManager (GuiManager*);
extern bool global_remote;

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
    if (strcmp (argv[1], "-slave") == 0) {
	global_remote = true;
	SlaveController* remote_task = new SlaveController(net,argv[2],argv[3]);
	remote_task->activate(0);  

	// startup slave-side GuiManager client
	set_guiManager (new GuiManager(argv[2], argv[3]));
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

