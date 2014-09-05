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
 *  Exception.h: Base exception class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Core/Exceptions/Exception.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sgi_stl_warnings_on.h>
#include <string.h>
#ifdef HAVE_EXC
#include <libexc.h>
#elif defined(__GNUC__) && defined(__linux)
#include <execinfo.h>
#endif


#if defined(_AIX)
// Needed for strcasecmp on aix 4.3 (on 5.1 we don't need this.)
// currently blue is 4.3.
#  include <strings.h>
#endif

namespace SCIRun {

using namespace std;

Exception::Exception()
{
  ostringstream stacktrace;
  static const int MAXSTACK = 100;

#ifdef HAVE_EXC
  // Use -lexc to print out a stack trace
  static const int MAXNAMELEN = 1000;
  __uint64_t addrs[MAXSTACK];
  char* cnames_str = new char[MAXSTACK*MAXNAMELEN];
  char* names[MAXSTACK];
  for(int i=0;i<MAXSTACK;i++)
    names[i]=cnames_str+i*MAXNAMELEN;
  int nframes = trace_back_stack(0, addrs, names, MAXSTACK, MAXNAMELEN);
  if(nframes == 0){
    stacktrace << "Backtrace not available!\n";
  } else {
    stacktrace << "Backtrace:\n";
    stacktrace.flags(ios::hex);
    // Skip the first procedure (us)
    for(int i=1;i<nframes;i++)
      stacktrace << "0x" << (void*)addrs[i] << ": " << names[i] << '\n';
  }
#elif defined(__GNUC__) && defined(__linux)
  static void *addresses[MAXSTACK];
  int n = backtrace( addresses, MAXSTACK );
  if (n < 2){
    stacktrace << "Backtrace not available!\n";
  } else {
    stacktrace << "Backtrace:\n";
    stacktrace.flags(ios::hex);
    char **names = backtrace_symbols( addresses, n );
    for ( int i = 2; i < n; i++ )
      stacktrace << names[i] << '\n';
    free(names);
  }
#endif
  stacktrace_ = strdup(stacktrace.str().c_str());
}

Exception::~Exception()
{
  if(stacktrace_)
    free((char*)stacktrace_);
}

// This is just to fool the compiler so that it will not complain about
// "loop expressions are constant"  (SGI mipspro)  See Exception.h for
// use - Steve.
bool Exception::alwaysFalse()
{
  return false;
}


void Exception::sci_throw(const Exception& exc)
{
  // This is a function invoked by the SCI_THROW macro.  It
  // can be useful for tracking down fatal errors, since the
  // normal exception mechanism will unwind the stack before
  // you can get anything useful out of it.

  // Set this environment variable if you want to have a default
  // response to the question below.  Value values are:
  // ask, dbx, cvd, throw, abort
  char* emode = getenv("SCI_EXCEPTIONMODE");
  if(!emode)
    emode = "dbx"; // Default exceptionmode

  // If the mode is not "throw", we print out a message
  if(strcasecmp(emode, "throw") != 0){
    cerr << "\n\nAn exception was thrown.  Msg: " << exc.message() << "\n";
    if(exc.stacktrace_){
      cerr << exc.stacktrace_;
    }
    // Print out the exception type (clasname) and the message
    cerr << "\nException type: " << exc.type() << '\n';
    cerr << "Exception message: " << exc.message() << '\n';
  }
  // See what we should do
  for(;;){
    if(strcasecmp(emode, "ask") == 0){
      // Ask the user
      cerr << "\nthrow(t)/dbx(d)/cvd(c)/abort(a)? ";
      emode=0;
      while(!emode){
	char action;
	char buf[100];
	while(read(fileno(stdin), buf, 100) <= 0){
	  if(errno != EINTR){
	    cerr <<  "\nCould not read response, throwing exception\n";
	    emode = "throw";
	    break;
	  }
	}
	action=buf[0];
	switch(action){
	case 't': case 'T':
	  emode="throw";
	  break;
	case 'd': case 'D':
	  emode="dbx";
	  break;
	case 'c': case 'C':
	  emode="cvd";
	  break;
	case 'a': case 'A':
	  emode="abort";
	  break;
	default:
	  break;
	}
      }
    }

    if(strcasecmp(emode, "throw") == 0) {
      // We cannot throw from here, so we just return and the
      // exception will be thrown by the SCI_THROW macro
      return;
    } else if(strcasecmp(emode, "dbx") == 0){
      // Fire up the debugger
      char command[100];
      if(getenv("SCI_DBXCOMMAND")){
	sprintf(command, getenv("SCI_DBXCOMMAND"), getpid());
      } else {
#ifdef HAVE_EXC
	sprintf(command, "winterm -c dbx -p %d &", getpid());
#else
	sprintf(command, "xterm -e gdb %d&", getpid());
#endif
      }
      cerr << "Starting: " << command << '\n';
      system(command);
      emode="ask";
    } else if(strcasecmp(emode, "cvd") == 0){
      // Fire up the slow, fancy debugger
      char command[100];
      sprintf(command, "cvd -pid %d &", getpid());
      cerr << "Starting: " << command << '\n';
      system(command);
      emode="ask";
    } else if(strcasecmp(emode, "abort") == 0){
      // This will trigger the thread library, but we cannot
      // directly call the thread library here or it would create
      // a circular dependency
      abort();
    } else {
      cerr << "Unknown exception mode: " << emode << ", aborting\n";
      abort();
    }
  }
}

} // End namespace SCIRun
