
/*
 *  Thread: The thread class
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <SCICore/Thread/Thread.h>
#include <SCICore/Exceptions/Exception.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/ThreadError.h>
#include <SCICore/Thread/ThreadGroup.h>
#include <errno.h>
#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#define THREAD_DEFAULT_STACKSIZE 64*1024

using SCICore::Thread::Thread;
using SCICore::Thread::ThreadGroup;

Thread::~Thread()
{
    if(d_runner){
        d_runner->d_my_thread=0;
        delete d_runner;
    }
}

Thread::Thread(ThreadGroup* g, const char* name)
{
    d_group=g;
    g->addme(this);
    d_threadname=name;
    d_daemon=false;
    d_detached=false;
    d_runner=0;
    d_cpu=-1;
    d_stacksize=THREAD_DEFAULT_STACKSIZE;
}

void
Thread::run_body()
{
    try {
	d_runner->run();
    } catch(const ThreadError& e){
	fprintf(stderr, "Caught unhandled Thread error:\n%s\n",
		e.message().c_str());
	Thread::niceAbort();
    } catch(const SCICore::Exceptions::Exception& e){
	fprintf(stderr, "Caught unhandled exception:\n%s\n",
		e.message().c_str());
	Thread::niceAbort();
    } catch(...){
	fprintf(stderr, "Caught unhandled exception of unknown type\n");
	Thread::niceAbort();
    }
}

Thread::Thread(Runnable* runner, const char* name,
	       ThreadGroup* group, ActiveState state)
    : d_runner(runner), d_threadname(name), d_group(group)
{
    if(d_group == 0){
        if(!ThreadGroup::s_default_group)
	    Thread::initialize();
        d_group=ThreadGroup::s_default_group;
    }

    d_runner->d_my_thread=this;
    d_group->addme(this);
    d_daemon=false;
    d_cpu=-1;
    d_detached=false;
    d_stacksize=THREAD_DEFAULT_STACKSIZE;
    switch(state){
    case Activated:
	os_start(false);
	d_activated=true;
	break;
    case Stopped:
	os_start(true);
	d_activated=true;
	break;
    case NotActivated:
	d_activated=false;
	d_priv=0;
	break;
    }
}

void
Thread::activate(bool stopped)
{
    if(d_activated)
	throw ThreadError("Thread is already activated");
    d_activated=true;
    os_start(stopped);
}

ThreadGroup*
Thread::getThreadGroup()
{
    return d_group;
}

void
Thread::setDaemon(bool to)
{
    d_daemon=to;
    checkExit();
}

bool
Thread::isDaemon() const
{
    return d_daemon;
}

bool
Thread::isDetached() const
{
    return d_detached;
}

const char*
Thread::getThreadName() const
{
    return d_threadname;
}

ThreadGroup*
Thread::parallel(const ParallelBase& helper, int nthreads,
		 bool block, ThreadGroup* threadGroup)
{
    ThreadGroup* newgroup=new ThreadGroup("Parallel group",
    				      threadGroup);
    for(int i=0;i<nthreads;i++){
        char buf[50];
        sprintf(buf, "Parallel thread %d of %d", i, nthreads);
        new Thread(new ParallelHelper(&helper, i), buf,
		   newgroup, Thread::Stopped);
    }
    newgroup->gangSchedule();
    newgroup->resume();
    if(block){
        newgroup->join();
        delete newgroup;
        return 0;
    } else {
        newgroup->detach();
    }
    return newgroup;
}

void
Thread::niceAbort()
{
    for(;;){
        char action;
        Thread* s=Thread::self();
	print_threads();
	fprintf(stderr, "\n");
	fprintf(stderr, "Abort signalled by pid: %d\n", getpid());
	fprintf(stderr, "Occured for thread:\n \"%s\"", s->d_threadname);
	fprintf(stderr, "resume(r)/dbx(d)/cvd(c)/kill thread(k)/exit(e)? ");
	fflush(stderr);
	char buf[100];
	while(read(fileno(stdin), buf, 100) <= 0){
	    if(errno != EINTR){
		fprintf(stderr, "\nCould not read response, exiting\n");
		buf[0]='e';
		exitAll(1);
	    }
	}
	action=buf[0];
        char command[500];
        switch(action){
        case 'r': case 'R':
	    return;
        case 'd': case 'D':
	    sprintf(command, "winterm -c dbx -p %d &", getpid());
	    system(command);	
	    break;
        case 'c': case 'C':
	    sprintf(command, "cvd -pid %d &", getpid());
	    system(command);	
	    break;
        case 'k': case 'K':
	    exit();
	    break;
        case 'e': case 'E':
	    exitAll(1);
	    break;
        default:
	    break;
        }
    }
}

int
Thread::couldBlock(const char* why)
{
    Thread_private* p=Thread::self()->d_priv;
    return push_bstack(p, BLOCK_ANY, why);
}

void
Thread::couldBlockDone(int restore)
{
    Thread_private* p=Thread::self()->d_priv;
    pop_bstack(p, restore);
}

unsigned long
Thread::getStackSize() const
{
    return d_stacksize;
}

void
Thread::setStackSize(unsigned long stacksize)
{
    if(d_activated)
	throw ThreadError("Cannot change stack size on a running thread");
    d_stacksize=stacksize;
}

/*
 * Return the statename for p
 */
const char*
Thread::getStateString(ThreadState state)
{
    switch(state) {
    case STARTUP:
	return "startup";
    case RUNNING:
	return "running";
    case IDLE:
	return "idle";
    case SHUTDOWN:
	return "shutting down";
    case BLOCK_SEMAPHORE:
	return "blocking on semaphore";
    case PROGRAM_EXIT:
	return "waiting for program exit";
    case JOINING:
	return "joining with thread";
    case BLOCK_MUTEX:
	return "blocking on mutex";
    case BLOCK_ANY:
	return "blocking";
    case DIED:
	return "died";
    case BLOCK_BARRIER:
	return "spinning in barrier";
    default:
	return "UNKNOWN";
    }
}

//
// $Log$
// Revision 1.7  1999/08/29 00:47:01  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.6  1999/08/28 03:46:50  sparker
// Final updates before integration with PSE
//
// Revision 1.5  1999/08/25 22:36:01  sparker
// More thread library updates - now compiles
//
// Revision 1.4  1999/08/25 19:00:51  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:38:00  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
