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
#include <string>

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

