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
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

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

