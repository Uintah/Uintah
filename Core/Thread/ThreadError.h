
/*
 *  ThreadError: Exception class for unusual errors in the thread library
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: August 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Thread_ThreadError_h
#define Core_Thread_ThreadError_h

#include <Core/share/share.h>

#include <Core/Exceptions/Exception.h>
#include <string>

namespace SCIRun {
/**************************************
 
CLASS
   ThreadError
   
KEYWORDS
   Exception, Thread
   
DESCRIPTION
   An exception class for serious thread library errors.  They are
   often not recoverable.

****************************************/
	class SCICORESHARE ThreadError : public Exception {
	public:
	    //////////
	    // Constructor for the ThreadError class.  Message is
	    // a human readable string that explains the reason for
	    // the error
	    ThreadError(const std::string& message);

	    //////////
	    // Copy ctor
	    ThreadError(const ThreadError&);

	    //////////
	    // Destructor
	    virtual ~ThreadError();

	    //////////
	    // returns the message associated with this error
	    virtual const char* message() const;

	    //////////
	    // returns the name of this exception (the name of this class)
	    virtual const char* type() const;

	protected:
	private:
	    std::string message_;

	    ThreadError& operator=(const ThreadError&);
	};
} // End namespace SCIRun

#endif

