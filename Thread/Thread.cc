
#include "Thread.h"
#include "ThreadGroup.h"
#include "ThreadError.h"
#include "ThreadEvent.h"
#include "Parallel.h"
#include "ThreadTopology.h"
#include "ThreadListener.h"
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream.h>
#include <errno.h>



Thread::~Thread() {
    if(runner){
        runner->mythread=0;
        delete runner;
    }
}

Thread::Thread(ThreadGroup* g, char* name) {
    group=g;
    g->addme(this);
    threadname=strdup(name);
    daemon=false;
    detached=false;
    cpu=-1;
    priority=5;
    abort_handler=0;
}

int Thread::nlisteners;
ThreadListener** Thread::listeners;
void Thread::event(Thread* t, ThreadEvent event) {
    for(int i=0;i<nlisteners;i++){
        listeners[i]->send_event(t, event);
    }
}

void Thread::run_body(){
    runner->run();
    //Thread::event(this, Thread::THREAD_DONE);
}

Thread::Thread(Runnable* runner, const char* name, ThreadGroup* group, bool stopped)
      : runner(runner), threadname(strdup(name)), group(group) {
    if(group == 0){
        if(!ThreadGroup::default_group)
    	Thread::initialize();
        group=ThreadGroup::default_group;
    }

    runner->mythread=this;
    group->addme(this);
    daemon=false;
    detached=false;
    cpu=-1;
    priority=5;
    os_start(stopped);
    abort_handler=0;
}

ThreadGroup* Thread::threadGroup() {
    return group;
}

void Thread::setDaemon(bool to) {
    daemon=to;
    check_exit();
}

int Thread::getPriority() const {
    return priority;
}

bool Thread::isDaemon() const {
    return daemon;
}

bool Thread::isDetached() const {
    return detached;
}

void Thread::error(char* error) {
    fprintf(stderr, "\n\nThread Error: %s\n ", error);
    Thread::niceAbort();
}

const char* Thread::threadName() const {
    return threadname;
}

Thread_private* Thread::getPrivate() const {
    return priv;
}

ThreadGroup* Thread::parallel(const ParallelBase& helper, int nthreads,
				 bool block, ThreadGroup* threadGroup) {
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

void Thread::niceAbort() {
    for(;;){
        char action;
        Thread* s=Thread::currentThread();
        if(s->abort_handler){
	    action=s->abort_handler->thread_abort(s);
        } else {
	    fprintf(stderr, "Abort signalled by pid: %d\n", getpid());
	    fprintf(stderr, "Occured for thread:\n \"%s\"", s->threadname);
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
        }
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

