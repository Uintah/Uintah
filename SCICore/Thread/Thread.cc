
/* REFERENCED */
static char *id="$Id$";

/*
 *  Thread.h: The thread class
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include "Thread.h"
#include "ThreadGroup.h"
#include "Parallel.h"
#include <Tester/RigorousTest.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream.h>
#include <errno.h>
#include <stdio.h>


Thread::~Thread()
{
    if(d_runner){
        d_runner->d_myThread=0;
        delete d_runner;
    }
}

Thread::Thread(ThreadGroup* g, const std::string& name)
{
    d_group=g;
    g->addme(this);
    d_threadname=name;
    d_daemon=false;
    d_detached=false;
    d_cpu=-1;
    d_priority=5;
    d_runner=0;
}

void Thread::run_body()
{
    d_runner->run();
}

Thread::Thread(Runnable* runner, const std::string& name,
	       ThreadGroup* group, bool stopped)
    : d_runner(runner), d_threadname(name), d_group(group)
{
    if(group == 0){
        if(!ThreadGroup::s_defaultGroup)
	    Thread::initialize();
        group=ThreadGroup::s_defaultGroup;
    }

    d_runner->d_myThread=this;
    d_group->addme(this);
    d_daemon=false;
    d_detached=false;
    d_cpu=-1;
    d_priority=5;
    os_start(stopped);
}

ThreadGroup* Thread::threadGroup()
{
    return d_group;
}

void Thread::setDaemon(bool to)
{
    d_daemon=to;
    checkExit();
}

int Thread::getPriority() const
{
    return d_priority;
}

bool Thread::isDaemon() const
{
    return d_daemon;
}

bool Thread::isDetached() const
{
    return d_detached;
}

const std::string& Thread::threadName() const
{
    return d_threadname;
}

Thread_private* Thread::getPrivate() const
{
    return d_priv;
}

ThreadGroup* Thread::parallel(const ParallelBase& helper, int nthreads,
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

void Thread::niceAbort()
{
    for(;;){
        char action;
        Thread* s=Thread::currentThread();
#if 0
        if(s->d_abortHandler){
	    //action=s->d_abortHandler->threadAbort(s);
        } else {
#endif
	    fprintf(stderr, "Abort signalled by pid: %d\n", getpid());
	    fprintf(stderr, "Occured for thread:\n \"%s\"", s->d_threadname.c_str());
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
    	exit(1);
    	break;
        case 'e': case 'E':
    	exitAll(1);
    	break;
        default:
    	break;
        }
    }
}

//
// $Log$
// Revision 1.3  1999/08/25 02:38:00  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

