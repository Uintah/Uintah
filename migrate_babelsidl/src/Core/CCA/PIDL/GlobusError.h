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

