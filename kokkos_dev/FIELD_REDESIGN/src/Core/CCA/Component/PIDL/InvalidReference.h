
/*
 *  InvalidReference.h: A "bad" reference to an object
 *  $Id$
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

#include <Component/PIDL/PIDLException.h>
#include <string>

namespace Component {
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
    }
}

#endif

//
// $Log$
// Revision 1.6  2000/03/23 20:43:06  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
// Revision 1.5  2000/03/23 10:27:36  sparker
// Added "name" method to match new Exception base class
//
// Revision 1.4  1999/09/24 20:15:58  sparker
// Cocoon documentation updates
//
// Revision 1.3  1999/09/24 06:26:24  sparker
// Further implementation of new Component model and IDL parser, including:
//  - fixed bugs in multiple inheritance
//  - added test for multiple inheritance
//  - fixed bugs in object reference send/receive
//  - added test for sending objects
//  - beginnings of support for separate compilation of sidl files
//  - beginnings of CIA spec implementation
//  - beginnings of cocoon docs in PIDL
//  - cleaned up initalization sequence of server objects
//  - use globus_nexus_startpoint_eventually_destroy (contained in
// 	the globus-1.1-utah.patch)
//
// Revision 1.2  1999/09/17 05:08:07  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.1  1999/08/30 17:39:45  sparker
// Updates to configure script:
//  rebuild configure if configure.in changes (Bug #35)
//  Fixed rule for rebuilding Makefile from Makefile.in (Bug #36)
//  Rerun configure if configure changes (Bug #37)
//  Don't build Makefiles for modules that aren't --enabled (Bug #49)
//  Updated Makfiles to build sidl and Component if --enable-parallel
// Updates to sidl code to compile on linux
// Imported PIDL code
// Created top-level Component directory
// Added ProcessManager class - a simpler interface to fork/exec (not finished)
//
//
