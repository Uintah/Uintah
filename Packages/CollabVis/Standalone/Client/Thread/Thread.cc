/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Thread/Thread.h>
#include <Core/Exceptions/Exception.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/ThreadError.h>
#include <Core/Thread/ThreadGroup.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#ifdef __sgi
#include <libexc.h>
#endif

#define THREAD_DEFAULT_STACKSIZE 64*1024*2


namespace SCIRun {

class ParallelHelper : public Runnable {
  const ParallelBase* helper;
  int proc;
public:
  ParallelHelper(const ParallelBase* helper, int proc)
    : helper(helper), proc(proc) {}
  virtual ~ParallelHelper() {}
  virtual void run() {
    ParallelBase* cheat=(ParallelBase*)helper;
    cheat->run(proc);
  }
};

bool Thread::initialized = false;

bool Thread::isInitialized()
{
  return initialized;
}

Thread::~Thread()
{
    if(runner_){
        runner_->my_thread_=0;
        delete runner_;
    }
    free(const_cast<char *>(threadname_));
}

Thread::Thread(ThreadGroup* g, const char* name)
{
    group_=g;
    g->addme(this);
    threadname_=strdup(name);
    daemon_=false;
    detached_=false;
    runner_=0;
    cpu_=-1;
    stacksize_=THREAD_DEFAULT_STACKSIZE;
}

void
Thread::run_body()
{
    try {
	runner_->run();
    } catch(const ThreadError& e){
	fprintf(stderr, "Caught unhandled Thread error:\n%s\n",
		e.message());
	Thread::niceAbort();
    } catch(const Exception& e){
	fprintf(stderr, "Caught unhandled exception:\n%s\n",
		e.message());
	Thread::niceAbort();
    } catch(...){
	fprintf(stderr, "Caught unhandled exception of unknown type\n");
	Thread::niceAbort();
    }
}

Thread::Thread(Runnable* runner, const char* name,
	       ThreadGroup* group, ActiveState state)
    : runner_(runner), threadname_(strdup(name)), group_(group)
{
  //cerr << "In Thread::Thread" << endl;
    if(group_ == 0){
        if(!ThreadGroup::s_default_group)
	    Thread::initialize();
        group_=ThreadGroup::s_default_group;
    }

    runner_->my_thread_=this;
    group_->addme(this);
    daemon_=false;
    cpu_=-1;
    detached_=false;
    stacksize_=THREAD_DEFAULT_STACKSIZE;
    switch(state){
    case Activated:
	os_start(false);
	activated_=true;
	break;
    case Stopped:
	os_start(true);
	activated_=true;
	break;
    case NotActivated:
	activated_=false;
	priv_=0;
	break;

    }
}

void
Thread::activate(bool stopped)
{
  //cerr << "In Thread::activate" << endl;
    if(activated_)
	throw ThreadError("Thread is already activated");
    activated_=true;
    os_start(stopped);
}

ThreadGroup*
Thread::getThreadGroup()
{
    return group_;
}

Runnable*
Thread::getRunnable()
{
    return runner_;
}

void
Thread::setDaemon(bool to)
{
    daemon_=to;
    checkExit();
}

bool
Thread::isDaemon() const
{
    return daemon_;
}

bool
Thread::isDetached() const
{
    return detached_;
}

const char*
Thread::getThreadName() const
{
    return threadname_;
}

ThreadGroup*
Thread::parallel(const ParallelBase& helper, int nthreads,
		 bool block, ThreadGroup* threadGroup)
{
  ThreadGroup* newgroup=new ThreadGroup("Parallel group", threadGroup);
  if(!block){
    // Extra synchronization to make sure that helper doesn't
    // get destroyed before the threads actually start
    helper.wait_=new Semaphore("Thread::parallel startup wait", 0);
  }
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
    helper.wait_->down(nthreads);
    delete helper.wait_;
    newgroup->detach();
  }
  return newgroup;
}

void
Thread::niceAbort()
{
#ifndef _WIN32
#ifdef __sgi
  // Use -lexc to print out a stack trace
  static const int MAXSTACK = 100;
  static const int MAXNAMELEN = 1000;
  __uint64_t addrs[MAXSTACK];
  char* cnames_str = new char[MAXSTACK*MAXNAMELEN];
  char* names[MAXSTACK];
  for(int i=0;i<MAXSTACK;i++)
    names[i]=cnames_str+i*MAXNAMELEN;
  int nframes = trace_back_stack(0, addrs, names, MAXSTACK, MAXNAMELEN);
  if(nframes == 0){
    fprintf(stderr, "Backtrace not available!\n");
  } else {
    fprintf(stderr, "Backtrace:\n");
    for(int i=0;i<nframes;i++)
      fprintf(stderr, "0x%p: %s\n", (void*)addrs[i], names[i]);
  }
#endif	// __sgi

  char* smode = getenv("SCI_SIGNALMODE");
  if (!smode)
    smode = "ask";
	
  Thread* s=Thread::self();
  print_threads();
  fprintf(stderr, "\n");
  fprintf(stderr, "Abort signalled by pid: %d\n", getpid());
  fprintf(stderr, "Occured for thread:\n \"%s\"", s->threadname_);
	
  for (;;) {
    if (strcasecmp(smode, "ask") == 0) {
      char buf[100];
      fprintf(stderr, "resume(r)/dbx(d)/cvd(c)/kill thread(k)/exit(e)? ");
      fflush(stderr);
      while(read(fileno(stdin), buf, 100) <= 0){
	if(errno != EINTR){
	  fprintf(stderr, "\nCould not read response, sleeping for 20 seconds\n");
	  sleep(20);
	  buf[0]='e';
	  exitAll(1);
	}
      }
      switch (buf[0]) {
      case 'r': case 'R':
	smode = "resume";
	break;
      case 'd': case 'D':
	smode = "dbx";
	break;
      case 'c': case 'C':
	smode = "cvd";
	break;
      case 'k': case 'K':
	smode = "kill";
	break;
      case 'e': case 'E':
	smode = "exit";
	break;
      default:
	break;
      }
    }

    if (strcasecmp(smode, "resume") == 0) {
      return;
    } else if (strcasecmp(smode, "dbx") == 0) {
      char command[500];
      if(getenv("SCI_DBXCOMMAND")){
	sprintf(command, getenv("SCI_DBXCOMMAND"), getpid());
      } else {
#ifdef __sgi
	sprintf(command, "winterm -c dbx -p %d &", getpid());
#else
	sprintf(command, "xterm -e gdb %d &", getpid());
#endif
      }
      system(command);
      smode = "ask";
    } else if (strcasecmp(smode, "cvd") == 0) {
      char command[500];
      sprintf(command, "cvd -pid %d &", getpid());
      system(command);
      smode = "ask";
    } else if (strcasecmp(smode, "kill") == 0) {
      exit();
    } else if (strcasecmp(smode, "exit") == 0) {
      exitAll(1);
    } else {
      fprintf(stderr, "Unrecognized option, exiting\n");
      smode = "exit";
    }
  }
#endif	// _WIN32
}

int
Thread::couldBlock(const char* why)
{
  if(!initialized)
    Thread::initialize();
  Thread_private* p=Thread::self()->priv_;
  return push_bstack(p, BLOCK_ANY, why);
}

void
Thread::couldBlockDone(int restore)
{
  Thread_private* p=Thread::self()->priv_;
  pop_bstack(p, restore);
}

unsigned long
Thread::getStackSize() const
{
  return stacksize_;
}

void
Thread::setStackSize(unsigned long stacksize)
{
  if(activated_)
    throw ThreadError("Cannot change stack size on a running thread");
  stacksize_=stacksize;
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

} // End namespace SCIRun
