
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
#include <ModuleList.h>
#include <Network.h>
#include <NetworkEditor.h>
#include <Scheduler.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    // The possible arguments are registered with the ArgProcessor
    // class.  This hands that class the list of arguments, and
    // the rest is handled "automagically".
    ArgProcessor::process_args(argc, argv);

    // Initialize the multithreader
    TaskManager::initialize();

    // Build the list of known modules...
    ModuleList::initialize_list();

    // Create initial network
    // We build the Network with a 1, indicating that this is the
    // first instantiation of the network class, and this network
    // should read the command line specified files (if any)
    Network* net=new Network(1);
    
    // Fork off task for the scheduler and the network editor
    // and tell them about each other.  They are both detached
    // tasks, and the Task* will be deleted by the task manager
    Scheduler* sched_task=new Scheduler(net);
    NetworkEditor* gui_task=new NetworkEditor(net);
    sched_task->set_gui(gui_task);
    gui_task->set_sched(sched_task);

    // Activate the tasks...  Arguments and return values are
    // meaningless
    sched_task->activate(0);
    gui_task->activate(0);

    // This will wait until all tasks have completed before exiting
    TaskManager::main_exit();

    // Never reached
    return 0;
}
