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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_AIX)
// Needed for strcasecmp on aix 4.3 (on 5.1 we don't need this.)
// currently blue is 4.3.
#  include <strings.h>
#endif

#include <string.h>
#include <sys/types.h>
#ifdef _WIN32
#include <windows.h>
#include <winnt.h>
#include <io.h>
#include <process.h>
#include <imagehlp.h>
#include <tlhelp32.h>
#else
#include <unistd.h>
#endif
#ifdef HAVE_EXC
#include <libexc.h>
#elif defined(__GNUC__) && defined(__linux)
#include <execinfo.h>
#endif



#define THREAD_DEFAULT_STACKSIZE 64*1024*2

// provide "C" interface to exitAll
extern "C" { 
void exit_all_threads(int rc) {
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

bool Thread::isInitialized()
{
  return initialized;
}

Thread::~Thread()
{
    if(runner_){
        runner_->my_thread_=0;
	if(runner_->delete_on_exit)
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
Thread::parallel(ParallelBase& helper, int nthreads,
		 bool block, ThreadGroup* threadGroup)
{
  if (block && nthreads <= 1)
  {
    helper.run(0);
    return 0;
  }

  ThreadGroup* newgroup=new ThreadGroup("Parallel group", threadGroup);
  if(!block){
    // Extra synchronization to make sure that helper doesn't
    // get destroyed before the threads actually start
    helper.wait_=new Semaphore("Thread::parallel startup wait", 0);
  }
  for(int i=0;i<nthreads;i++){
    char buf[50];
    sprintf(buf, "Parallel thread %d of %d", i, nthreads);
    new Thread(new ParallelHelper(helper, i), buf,
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
Thread::niceAbort(void* Context /* = 0 */)
{
#ifdef HAVE_EXC
  static const int MAXSTACK = 100;
  // Use -lexc to print out a stack trace
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
#elif defined(__GNUC__) && defined(__linux)
  static const int MAXSTACK = 100;
  static void *addresses[MAXSTACK];
  int n = backtrace( addresses, MAXSTACK );
  if (n == 0){
    fprintf(stderr, "Backtrace not available!\n");
  } else {
    fprintf(stderr, "Backtrace:\n");
    char **names = backtrace_symbols( addresses, n );
    for ( int i = 0; i < n; i++ )
    {
      fprintf (stderr, "%s\n", names[i]);
    } 
    free(names);
  } 
#elif defined(_WIN32)
  // fix for IA64, and AMD64 later
  // inits of sf.* and 1st param to stackwalk64 are for i386 currently
  // use ImageNtHeader, look in PE header


  // setup initial structs
  int i = 0;
  static const int MAXNAMELEN = 1000;
  static const int IMGSYMLEN = sizeof(IMAGEHLP_SYMBOL);
  static IMAGEHLP_SYMBOL* image_sym = 0;
  char name[MAXNAMELEN]; // undecorated name

  // this is where the symbol info will be stored
  if (image_sym == 0) {
    image_sym = (IMAGEHLP_SYMBOL *) malloc( IMGSYMLEN + MAXNAMELEN );
  }
  memset( image_sym, 0, IMGSYMLEN + MAXNAMELEN );
  image_sym->SizeOfStruct = IMGSYMLEN;
  image_sym->MaxNameLength = MAXNAMELEN;

  // thread context and stack frame
  CONTEXT context;
  if (Context == 0) {
    memset(&context, 0, sizeof(CONTEXT));
    context.ContextFlags = CONTEXT_FULL;

    GetThreadContext(GetCurrentThread(), &context); 
  }
  else {
    context = *(CONTEXT*)Context;
  }
  STACKFRAME sf;
  memset(&sf, 0, sizeof(STACKFRAME));
  sf.AddrPC.Offset = context.Eip; // for X86
  sf.AddrPC.Mode = AddrModeFlat;
  sf.AddrFrame.Offset = context.Ebp; // for X86
  sf.AddrFrame.Mode = AddrModeFlat;
  HANDLE hProc = GetCurrentProcess();

  static bool first = true;

  if (first) {
    SymInitialize(hProc, 0, true);
    first = false;
  }

  bool success = StackWalk(IMAGE_FILE_MACHINE_I386, hProc, GetCurrentThread(), &sf, 0, 0, SymFunctionTableAccess, SymGetModuleBase, 0) && sf.AddrPC.Offset != 0;

  while (success) {
    DWORD offset = 0;
    if (SymGetSymFromAddr(hProc, sf.AddrPC.Offset, &offset, image_sym)) {
      UnDecorateSymbolName(image_sym->Name, name, MAXNAMELEN, UNDNAME_COMPLETE);
      fprintf(stderr, "#%d %s\n", i, name);
    }
    else {
      fprintf(stderr, "#%d ???\n", i);
      static int bad = 0;
      bad++;
      if (bad > 25)
        break;
    }  
    i++;
    success = StackWalk(IMAGE_FILE_MACHINE_I386, hProc, GetCurrentThread(), &sf, 0, 0, SymFunctionTableAccess, SymGetModuleBase, 0) && sf.AddrPC.Offset != 0;
  }
  fflush(stderr);
#endif

  char* smode = getenv("SCI_SIGNALMODE");
  if (!smode)
    smode = "ask";
	
  Thread* s=Thread::self();
  print_threads();
  fprintf(stderr, "\n");
  fprintf(stderr, "Abort signalled by pid: %d\n", getpid());
  if(s)
    fprintf(stderr, "Occured for thread: \"%s\"\n", s->threadname_);
  else
    fprintf(stderr, "With NULL thread pointer.\n");

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
