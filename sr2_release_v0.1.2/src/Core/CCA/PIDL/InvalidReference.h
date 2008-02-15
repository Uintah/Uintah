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
 *  InvalidReference.h: A "bad" reference to an object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_InvalidReference_h
#define CCA_PIDL_InvalidReference_h

#include "PIDLException.h"
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
/**************************************
 
CLASS
   InvalidReference
   
KEYWORDS
   Exception, Error, PIDL
   
DESCRIPTION
   Exception class for an invalid object reference.  This can result
   from requesting an invalid object from PIDL::objectFrom

****************************************/
	class InvalidReference : public PIDLException {
	public:
	    //////////
	    // Construct the exception with the given explanation
	    InvalidReference(const std::string&);

	    //////////
	    // Copy ctor
	    InvalidReference(const InvalidReference&);

	    //////////
	    // Destructor
	    virtual ~InvalidReference();

	    //////////
	    // Return the explanation
	    const char* message() const;

	    //////////
	    // Return the name of this class
	    const char* type() const;
	protected:
	private:
	    //////////
	    // The explanation string.
	    std::string d_msg;

	    InvalidReference& operator=(const InvalidReference&);
	};
} // End namespace SCIRun

#endif

