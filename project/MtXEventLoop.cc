
/*
 *  MtXEventLoop.h: The Event loop thread
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <MtXEventLoop.h>
#include <Classlib/ArgProcessor.h>
#include <Multitask/ITC.h>
#include <iostream.h>
#include <stdio.h> // For perror
#include <stdlib.h>
#include <unistd.h>

#define READ 0
#define WRITE 1

static String fallback_resources[] = {
    "*fontList: screen14",
    "*background: pink3",
    NULL
    };


MtXEventLoop::MtXEventLoop()
: Task("Xt Event Loop", 1) // Detached...
{
    started=0;
    lock_count=0;
}

MtXEventLoop::~MtXEventLoop()
{
}

XtAppContext MtXEventLoop::get_app()
{
    return context;
}

Display* MtXEventLoop::get_display()
{
    return display;
}

Screen* MtXEventLoop::get_screen()
{
    return screen;
}

void MtXEventLoop::wait_start()
{
    while(!started){
	Task::yield();
    }
}

void do_process_token(XtPointer ud, int* source, XtInputId*)
{
    MtXEventLoop* that=(MtXEventLoop*)ud;
    that->process_token(*source);
}

int MtXEventLoop::body(int)
{
    eventloop_taskid=Task::self();
    XtToolkitInitialize();
    context=XtCreateApplicationContext();
    XtAppSetFallbackResources(context, fallback_resources);
    int x_argc;
    char** x_argv;
    ArgProcessor::get_x_args(x_argc, x_argv);
    clString progname(ArgProcessor::get_program_name());
    display=XtOpenDisplay(context, NULL, progname(),
			  "sci", NULL, 0, &x_argc, x_argv);
    if(!display){
	cerr << "Error opening display\n";
	exit(-1);
    }
    screen=DefaultScreenOfDisplay(display);

    // Setup token...
    mutex=new Mutex;
    sema=new Semaphore(0);
    pipe(&pipe_fd[0]);
    XtAppAddInput(context, pipe_fd[READ], (XtPointer)XtInputReadMask,
		  do_process_token, this);

    // Go into main loop;
    started=1;
    XtAppMainLoop(context);

    // Never reached...
    return 0;
}

void MtXEventLoop::process_token(int source)
{
    mutex->lock();
    // Read the input to find the requestor...
    Semaphore* who;
    if(read(source, (void*)&who, sizeof(Semaphore*)) != sizeof(Semaphore*)){
	perror("read");
	exit(-1);
    }
    mutex->unlock();
    // Signal their semaphore to wake them up...
    who->up();

    // Wait until they are done...
    sema->down();
}

void MtXEventLoop::lock()
{
    // See if we have the token already....
    if(token_owner == Task::self()){
	// Just increment the count...
	lock_count++;
	return;
    }
    Semaphore mysema(0);
    mutex->lock();
    Semaphore* myaddr=&mysema;
    if(write(pipe_fd[WRITE], (void*)&myaddr, sizeof(Semaphore*)) != sizeof(Semaphore)){
	perror("write");
	exit(-1);
    }
    mutex->unlock();

    // Block until the Event loop says that it is OK
    mysema.down();
    // We have the token now...
    lock_count++;
    token_owner=Task::self();
}

void MtXEventLoop::unlock()
{
    // Unblock the event loop thread...
    if(--lock_count==0){
	// Reset the owner before we give it up...
	token_owner=eventloop_taskid;
	sema->up();
    }
}
