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
 *  GlobusError.h: Errors due to globus calls
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_GlobusError_h
#define CCA_PIDL_GlobusError_h

#include <Core/CCA/PIDL/PIDLException.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  /**************************************
 
  CLASS
     GlobusError
   
  KEYWORDS
     Exception, Error, globus, PIDL
   
  DESCRIPTION
     Exception class for globus functions.  An unhandled negative return
   code from a globus function will get mapped to this exception.  The
   message is a description of the call, and the code is the result
   returned from globus.

****************************************/
	class GlobusError : public PIDLException {
	public:
	    //////////
	    // Construct the exception with the given reason and the
	    // return code from globus
	    GlobusError(const std::string& msg, int code);

	    //////////
	    // Copy ctor
	    GlobusError(const GlobusError&);

	    //////////
	    // Destructor
	    virtual ~GlobusError();

	    //////////
	    // An explanation message, containing the msg string and the
	    // return code passed into the constructor.
	    const char* message() const;

	    //////////
	    // The name of this class
	    const char* type() const;
	protected:
	private:
	    //////////
	    // The explanation string (usually the name of the offending
	    // call).
	    std::string d_msg;

	    //////////
	    // The globus error code.
	    int d_code;

	    GlobusError& operator=(const GlobusError&);
	};
} // End namespace SCIRun

#endif

