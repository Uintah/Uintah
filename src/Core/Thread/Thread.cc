
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
#include <SCICore/Exceptions/Exception.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Runnable.h>
#include <SCICore/Thread/ThreadError.h>
#include <SCICore/Thread/ThreadGroup.h>
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

using SCICore::Thread::ParallelBase;
using SCICore::Thread::Runnable;
using SCICore::Thread::Thread;
using SCICore::Thread::ThreadGroup;

namespace SCICore {
namespace Thread {

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

}
}

using SCICore::Thread::ParallelHelper;

Thread::~Thread()
{
    if(d_runner){
        d_runner->d_my_thread=0;
        delete d_runner;
    }
    free(const_cast<char *>(d_threadname));
}

Thread::Thread(ThreadGroup* g, const char* name)
{
    d_group=g;
    g->addme(this);
    d_threadname=strdup(name);
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
		e.message());
	Thread::niceAbort();
    } catch(const SCICore::Exceptions::Exception& e){
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
    : d_runner(runner), d_threadname(strdup(name)), d_group(group)
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

Runnable*
Thread::getRunnable()
{
    return d_runner;
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
  ThreadGroup* newgroup=new ThreadGroup("Parallel group", threadGroup);
  if(!block){
    // Extra synchronization to make sure that helper doesn't
    // get destroyed before the threads actually start
    helper.d_wait=new Semaphore("Thread::parallel startup wait", 0);
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
    helper.d_wait->down(nthreads);
    delete helper.d_wait;
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
  fprintf(stderr, "Occured for thread:\n \"%s\"", s->d_threadname);
	
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
	sprintf(command, "xterm -e dbx -p %d &", getpid());
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
