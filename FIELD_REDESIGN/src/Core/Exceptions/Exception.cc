
/*
 *  Exception.h: Base exception class
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <SCICore/Exceptions/Exception.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#ifdef __sgi
#include <libexc.h>
#include <sstream>
#include <string.h>
#endif

using namespace std;
using SCICore::Exceptions::Exception;

Exception::Exception()
{
#ifdef __sgi
   ostringstream stacktrace;
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
      stacktrace << "Backtrace not available!\n";
   } else {
      stacktrace << "Backtrace:\n";
      stacktrace.flags(ios::hex);
      // Skip the first procedure (us)
      for(int i=1;i<nframes;i++)
	 stacktrace << "0x" << (void*)addrs[i] << ": " << names[i] << '\n';
   }
   d_stacktrace = strdup(stacktrace.str().c_str());
#else
   d_stacktrace = 0;
#endif
}

Exception::~Exception()
{
   if(d_stacktrace)
      free((char*)d_stacktrace);
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
	if(exc.d_stacktrace){
	   cerr << exc.d_stacktrace;
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
	    sprintf(command, "winterm -c dbx -p %d &", getpid());
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

//
// $Log$
// Revision 1.3.2.3  2000/10/26 17:51:52  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.5  2000/07/27 07:40:46  sparker
// Save the stack trace when an exception is thrown (SGI only)
//
// Revision 1.4  2000/06/08 21:08:43  dav
// added more verbose error message
//
// Revision 1.3  2000/03/24 00:06:29  yarden
// replace stderr with cerr.
// include <stdio,h> for sprintf
//
// Revision 1.1  2000/03/23 10:25:40  sparker
// New exception facility - retired old "Exception.h" classes
//
//
