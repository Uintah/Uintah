/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Thread/Time.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(_AIX)
// Needed for strcasecmp on aix 4.3 (on 5.1 we don't need this.)
// currently blue is 4.3.
#  include <strings.h>
#endif

#include <cstring>
#include <sys/types.h>
#ifdef _WIN32
#include <windows.h>
#include <winnt.h>
#include <io.h>
#include <process.h>
#include <imagehlp.h>
#include <psapi.h>
#define SCI_OK_TO_INCLUDE_SCI_ENVIRONMENT_DEFS_H
#include <sci_defs/environment_defs.h> // for SCIRUN_OBJDIR. can't use sci_getenv lest we create a circular dependency
#undef SCI_OK_TO_INCLUDE_SCI_ENVIRONMENT_DEFS_H
#define strcasecmp stricmp //native windows doesn't have strcasecmp
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE
#include <unistd.h>
#endif
#ifdef HAVE_EXC
#include <libexc.h>
#elif defined(__GNUC__) && defined(__linux)
#include <execinfo.h>
#endif





// provide "C" interface to exitAll
extern "C" { 
  void SCISHARE exit_all_threads(int rc) {
    SCIRun::Thread::exitAll(rc);
  }
}

namespace SCIRun {

class ParallelHelper : public Runnable {
  ParallelBase &helper_;
  int proc_;
public:
  ParallelHelper(ParallelBase& helper, int proc)
    : helper_(helper), proc_(proc) {}
  virtual ~ParallelHelper() {}
  virtual void run() {
    helper_.run(proc_);
  }
};

bool Thread::initialized = false;
const char* Thread::defaultAbortMode = "ask";
bool Thread::callExit = true;


bool Thread::isInitialized()
{
  return initialized;
}

void Thread::setDefaultAbortMode(const char* abortMode)
{
  defaultAbortMode = abortMode;
}

Thread::~Thread()
{
  if(runner_) {
    runner_->my_thread_=0;
    if(runner_->delete_on_exit) {
      delete runner_;
    }
  }
  free(const_cast<char *>(threadname_));
}

Thread::Thread( ThreadGroup * g, const char * name )
{
  g->addme( this );
  group_      = g;
  threadname_ = strdup(name);
  daemon_     = false;
  detached_   = false;
  runner_     = 0;
  cpu_        = -1;
  myid_         = 0;
  stacksize_  = Thread::DEFAULT_STACKSIZE;
  abortCleanupFunc_ = NULL;
}

void
Thread::run_body()
{
  try {
    runner_->run();
  } catch(const ThreadError& e) {
    fprintf(stderr, "Caught unhandled Thread error:\n%s\n", e.message());
    Thread::niceAbort();
  } catch(const Exception& e) {
    fprintf(stderr, "Caught unhandled exception:\n%s\n",e.message());
    const char *trace = e.stackTrace();
    if (trace) {
      fprintf(stderr, "Exception %s", trace);
    }
    Thread::niceAbort();
  } catch(const std::string &e) {
    fprintf(stderr, "Caught unhandled string exception:\n%s\n", e.c_str());
    Thread::niceAbort();
  } catch(const char *&e) {
    fprintf(stderr, "Caught unhandled char exception:\n%s\n", e);
    Thread::niceAbort();
#ifndef _MSC_VER 
    // catch these differently with MS compiler, we can get the whole stack trace, but it must be done with
    // an MS-specific exception handler in a different function
  } catch(...) {
    fprintf(stderr, "Caught unhandled exception of unknown type\n");
    Thread::niceAbort();
#endif
  }
}

Thread::Thread( Runnable* runner, const char* name,
                ThreadGroup* group, ActiveState state,
                unsigned long stacksize ) :
  runner_(runner),
  threadname_(strdup(name)),
  group_(group),
  stacksize_(stacksize),
  daemon_(false),
  detached_(false),
  cpu_(-1)
{
  if(group_ == 0) {
    if( !ThreadGroup::s_default_group )
      Thread::initialize();
    group_=ThreadGroup::s_default_group;
  }

  runner_->my_thread_=this;
  group_->addme(this);
  switch(state) {
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
  if(activated_) {
	throw ThreadError("Thread is already activated");
  }
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
Thread::parallel(ParallelBase& helper, int nthreads,
		 bool block, ThreadGroup* threadGroup)
{
  if (block && nthreads <= 1) {
    helper.run(0);
    return 0;
  }

  ThreadGroup* newgroup=new ThreadGroup("Parallel group", threadGroup);
  if(!block) {
    // Extra synchronization to make sure that helper doesn't
    // get destroyed before the threads actually start
    helper.wait_=new Semaphore("Thread::parallel startup wait", 0);
  }
  for(int i=0;i<nthreads;i++) {
    char buf[50];
    sprintf(buf, "Parallel thread %d of %d", i, nthreads);
    new Thread(new ParallelHelper(helper, i), buf,
	       newgroup, Thread::Stopped);
  }
  newgroup->gangSchedule();
  newgroup->resume();
  if(block) {
    newgroup->join();
    delete newgroup;
    return 0;
  } 
  else {
    helper.wait_->down(nthreads);
    delete helper.wait_;
    newgroup->detach();
  }
  return newgroup;
}

void
Thread::handleCleanup()
{
  Thread::ptr2cleanupfunc funcPtr = Thread::self()->getCleanupFunction();
  if( funcPtr != NULL ) {

    // printf( "Handling thread cleanup for thread '%s'.\n", threadname_ );

    (*funcPtr)();

    // handleCleanup() should only be called once, but just in case... 
    Thread::self()->setCleanupFunction( NULL );
  }
}

void
Thread::niceAbort(void* context /* = 0 */, bool print /*= true */)
{
  if(print) {
    fprintf( stderr, "%s", getStackTrace(context).c_str() );
  }

  const char* smode = getenv("SCI_SIGNALMODE");
  if (!smode)
    smode = defaultAbortMode; //"e"; 
	
  Thread* s=Thread::self();
  if(print)
  {
    print_threads();
    fprintf(stderr, "\n");
    fprintf(stderr, "Abort signalled by pid: %d\n", getpid());
    if(s)
      fprintf(stderr, "Occured for thread: \"%s\"\n", s->threadname_);
    else
      fprintf(stderr, "With NULL thread pointer.\n");
  }

  for (;;) {
    if (strcasecmp(smode, "ask") == 0) {
      char buf[100];
      fprintf(stderr, "resume(r)/dbx(d)/cvd(c)/kill thread(k)/exit(e)? ");
      fflush(stderr);
      while(read(fileno(stdin), buf, 100) <= 0){
	if(errno != EINTR){
	  fprintf(stderr, "\nCould not read response, sleeping for 20 seconds.\n");
          Time::waitFor(20.0);
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

#if defined( REDSTORM )
      printf("Error: running debugger at exception is not supported on RedStorm\n");
#else
      char command[500];
      if(getenv("SCI_DBXCOMMAND")){
	sprintf(command, getenv("SCI_DBXCOMMAND"), getpid());
      } else {
	sprintf(command, "xterm -e gdb %d &", getpid());
      }
      system(command);
      smode = "ask";
#endif
    } else if (strcasecmp(smode, "cvd") == 0) {
#if defined( REDSTORM )
      printf("Error: running debugger at exception is not supported on RedStorm\n");
#else
      char command[500];
      sprintf(command, "cvd -pid %d &", getpid());
      system(command);
      smode = "ask";
#endif
    } else if (strcasecmp(smode, "kill") == 0) {
      exit();
    } else if (strcasecmp(smode, "exit") == 0) {
      exitAll(1);
    } else {
      fprintf(stderr, "Unrecognized option, exiting\n");
      smode = "exit";
    }
  } // end for(;;)
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

void
Thread::setCleanupFunction( Thread::ptr2cleanupfunc func )
{
  abortCleanupFunc_ = func;
}

Thread::ptr2cleanupfunc
Thread::getCleanupFunction()
{
  return abortCleanupFunc_;
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
