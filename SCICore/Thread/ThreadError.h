
/*
 *  ThreadError: Exception class for unusual errors in the thread library
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: August 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCICore_Thread_ThreadError_h
#define SCICore_Thread_ThreadError_h

#include <SCICore/Exceptions/Exception.h>

/**************************************
 
CLASS
   ThreadError
   
KEYWORDS
   Exception, Thread
   
DESCRIPTION
   An exception class for serious thread library errors.  They are
   often not recoverable.

PATTERNS


WARNING
   
****************************************/

namespace SCICore {
    namespace Thread {
	class ThreadError : public SCICore::Exceptions::Exception {
	public:
	    //////////
	    // Constructor for the ThreadError class.  Message is
	    // a human readable string that explains the reason for
	    // the error
	    ThreadError(const std::string& message);

	    //////////
	    // Destructor
	    virtual ~ThreadError();

	    //////////
	    // Prints out the message associated with this error
	    virtual std::string message() const;
	protected:
	private:
	    std::string d_message;
	};
    }
}

#endif

//
// $Log$
// Revision 1.4  1999/08/28 03:46:51  sparker
// Final updates before integration with PSE
//
// Revision 1.3  1999/08/25 19:00:51  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
//
