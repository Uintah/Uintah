
/* REFERENCED */
static char *id="$Id$";

/*
 *  Thread: The thread class
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
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/ThreadGroup.h>
#include <errno.h>
#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

SCICore::Thread::Thread::~Thread()
{
    if(d_runner){
        d_runner->d_my_thread=0;
        delete d_runner;
    }
}

SCICore::Thread::Thread::Thread(ThreadGroup* g, const char* name)
{
    d_group=g;
    g->addme(this);
    d_threadname=name;
    d_daemon=false;
    d_detached=false;
    d_runner=0;
    d_cpu=-1;
}

void
SCICore::Thread::Thread::run_body()
{
    d_runner->run();
}

SCICore::Thread::Thread::Thread(Runnable* runner, const char* name,
	       ThreadGroup* group, bool stopped)
    : d_runner(runner), d_threadname(name), d_group(group)
{
    if(group == 0){
        if(!ThreadGroup::s_default_group)
	    Thread::initialize();
        group=ThreadGroup::s_default_group;
    }

    d_runner->d_my_thread=this;
    d_group->addme(this);
    d_daemon=false;
    d_cpu=-1;
    d_detached=false;
    os_start(stopped);
}

SCICore::Thread::ThreadGroup*
SCICore::Thread::Thread::getThreadGroup()
{
    return d_group;
}

void
SCICore::Thread::Thread::setDaemon(bool to)
{
    d_daemon=to;
    checkExit();
}

bool
SCICore::Thread::Thread::isDaemon() const
{
    return d_daemon;
}

bool
SCICore::Thread::Thread::isDetached() const
{
    return d_detached;
}

const char*
SCICore::Thread::Thread::getThreadName() const
{
    return d_threadname;
}

SCICore::Thread::ThreadGroup*
SCICore::Thread::Thread::parallel(const ParallelBase& helper, int nthreads,
		 bool block, ThreadGroup* threadGroup)
{
    ThreadGroup* newgroup=new ThreadGroup("Parallel group",
    				      threadGroup);
    for(int i=0;i<nthreads;i++){
        char buf[50];
        sprintf(buf, "Parallel thread %d of %d", i, nthreads);
        new Thread(new ParallelHelper(&helper, i), buf,
		   newgroup, true);
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
SCICore::Thread::Thread::niceAbort()
{
    for(;;){
        char action;
        Thread* s=Thread::self();
#if 0
        if(s->d_abortHandler){
	    //action=s->d_abortHandler->threadAbort(s);
        } else {
#endif
	    fprintf(stderr, "Abort signalled by pid: %d\n", getpid());
	    fprintf(stderr, "Occured for thread:\n \"%s\"", s->d_threadname);
	    fprintf(stderr, "resume(r)/dbx(d)/cvd(c)/kill thread(k)/exit(e)? ");
	    fflush(stderr);
	    char buf[100];
	    while(read(fileno(stdin), buf, 100) <= 0){
		if(errno != EINTR){
		    fprintf(stderr, "\nCould not read response, exiting\n");
		    buf[0]='e';
		    break;
		}
	    }
	    action=buf[0];
#if 0
        }
#endif
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
SCICore::Thread::Thread::couldBlock(const char* why)
{
    Thread_private* p=Thread::self()->d_priv;
    return push_bstack(p, BLOCK_ANY, why);
}

void
SCICore::Thread::Thread::couldBlockDone(int restore)
{
    Thread_private* p=Thread::self()->d_priv;
    pop_bstack(p, restore);
}

/*
 * Return the statename for p
 */
const char*
SCICore::Thread::Thread::getStateString(ThreadState state)
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
