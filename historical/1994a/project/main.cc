
/*
 *  main.cc: Main program for project
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/ArgProcessor.h>
#include <Multitask/Task.h>
#include <ColorManager.h>
#include <Dialbox.h>
#include <ModuleList.h>
#include <MtXEventLoop.h>
#include <Network.h>
#include <NetworkEditor.h>
#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

MtXEventLoop* evl;

int main(int argc, char** argv)
{
    // The possible arguments are registered with the ArgProcessor
    // class.  This hands that class the list of arguments, and
    // the rest is handled "automagically".
    ArgProcessor::process_args(argc, argv);

    // If the environmental variable SCI_DATA is set, change
    // to that directory
    char* datadir=getenv("SCI_DATA");
    if(datadir){
#if REALLY_CHDIR
	cerr << "Changing to directory: " << datadir << endl;
	if(chdir(datadir) == -1){
	    perror("chdir");
	    cerr << "Cannot change to directory: " << datadir << endl;
	}
#endif
	static char dirbuf[1000];
	sprintf(dirbuf, "PWD=%s", datadir);
	putenv(dirbuf);
    }

    // Initialize the multithreader
    TaskManager::initialize(argv[0]);

    // Fork off a task for the Event loop handler...
    evl=new MtXEventLoop();
    evl->activate(0);

    // Wait until it gets started
    evl->wait_start();

    // Find the display and create a ColorManager for the default
    // colormap
    Display* display=evl->get_display();
    Screen* screen=evl->get_screen();
    ColorManager* color_manager=new ColorManager(display,
						 DefaultColormapOfScreen(screen));
    
    // Create initial network
    // We build the Network with a 1, indicating that this is the
    // first instantiation of the network class, and this network
    // should read the command line specified files (if any)
    Network* net=new Network(1);

    // Fork off task for the network editor.  It is a detached
    // task, and the Task* will be deleted by the task manager
    NetworkEditor* gui_task=new NetworkEditor(net, display, color_manager);

    // Activate the network editor and scheduler.  Arguments and return
    // values are meaningless
    gui_task->activate(0);

    // Fork the dial box manager
    Dialbox* dialtask=new Dialbox(color_manager);
    dialtask->activate(0);

    // This will wait until all tasks have completed before exiting
    TaskManager::main_exit();

    // Never reached
    return 0;
}

