 
/*
 *  TCLTask.h:  Handle TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <TCLTask.h>
#include <TCL.h>
#include <Multitask/ITC.h>
#include "../tcl/tcl7.3/tcl.h"
#include "../tcl/tk3.6/tk.h"

extern "C" int tkMain(int argc, char** argv);
extern Tcl_Interp* the_interp;

static Mutex* tlock=0;
static Task* owner;
static int lock_count;

static void do_lock()
{
    if(owner == Task::self()){
	lock_count++;
	return;
    }
    tlock->lock();
    lock_count=1;
    owner=Task::self();
}

static void do_unlock()
{
    if(--lock_count == 0){
	owner=0;
	tlock->unlock();
    }
}

TCLTask::TCLTask(int argc, char* argv[])
: Task("TCLTask", 1), argc(argc), argv(argv)
{
    if(!tlock)
	tlock=new Mutex;
    Tcl_SetLock(do_lock, do_unlock);
    Tk_SetSelectProc((Tk_SelectProc*)Task::mtselect);
    tkMain(argc, argv);
    TCL::initialize();
}

TCLTask::~TCLTask()
{
}

int TCLTask::body(int)
{
    Tk_MainLoop();
    Tcl_Eval(the_interp, "exit");
    return 0;
}

void TCLTask::lock()
{
    do_lock();
}

void TCLTask::unlock()
{
    do_unlock();
}

void TCLTask::wait_start()
{
    extern int Tk_Started;
    while(!Tk_Started){
	Task::yield();
    }
}
