
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

#include <SCICore/share/share.h>

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace SCICore {
    namespace Thread {
/**************************************
 
CLASS
   ThreadError
   
KEYWORDS
   Exception, Thread
   
DESCRIPTION
   An exception class for serious thread library errors.  They are
   often not recoverable.

****************************************/
	class SCICORESHARE ThreadError : public SCICore::Exceptions::Exception {
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
	    // returns the message associated with this error
	    virtual const char* message() const;

	    //////////
	    // returns the name of this exception (the name of this class)
	    virtual const char* type() const;

	protected:
	private:
	    std::string d_message;
	};
    }
}

#endif

//
// $Log$
// Revision 1.8  2000/03/23 10:21:27  sparker
// Use libexc to print out stack straces on the SGI
// Added "name" method to ThreadError to match exception base class
// Fixed a compiler warning in Thread_irix.cc
//
// Revision 1.7  1999/09/24 19:18:31  moulding
// removed an '=' from the end of the include guard #define
//
// Revision 1.6  1999/09/24 18:55:08  moulding
// added SCICORESHARE, for win32, to class declarations
//
// Revision 1.5  1999/09/02 16:52:44  sparker
// Updates to cocoon documentation
//
// Revision 1.4  1999/08/28 03:46:51  sparker
// Final updates before integration with PSE
//
// Revision 1.3  1999/08/25 19:00:51  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
//
