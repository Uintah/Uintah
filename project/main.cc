
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

#include <Classlib/Args.h>
#include <Classlib/ArgProcessor.h>
#include <Multitask/Task.h>
#include <stdlib.h>

static Arg_flag nogui("nogui", "Turn off Network GUI display");

int main(int argc, char** argv)
{
    // The possible arguments are registered with the ArgProcessor
    // class.  This hands that class the list of arguments, and
    // the rest is handled "automagically".
    ArgProcessor::process_args(argc, argv);

    // Initialize the multithreader
    TaskManager::initialize();

    // Create initial network
    // We build the Network with a 1, indicating that this is the
    // first instantiation of the network class, and this network
    // should read the command line specified files (if any)
#if 0
    Network* net=new Network(1);
    
    // Fork off task for scheduler
    // This is a detached task, and will be deleted by the task manager
    Task* sched_task=new Scheduler(net);
    sched_task->activate();

    // Fork off task for Network editor
    if(!nogui_flag.is_set()){
	// This is a detached task, and will be deleted by the task manager
	Task* gui_task=new NetworkEditor(net);
	gui_task->activate();
    }
#endif

    // This will wait until all tasks have completed before exiting
    TaskManager::main_exit();
    // Never reached
    return 0;
}
