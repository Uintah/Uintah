
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

#ifndef Component_PIDL_InvalidReference_h
#define Component_PIDL_InvalidReference_h

#include <Core/CCA/Component/PIDL/PIDLException.h>
#include <string>

namespace PIDL {
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
} // End namespace PIDL

#endif

