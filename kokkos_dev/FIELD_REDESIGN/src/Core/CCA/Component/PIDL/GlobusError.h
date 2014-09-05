
/*
 *  GlobusError.h: Errors due to globus calls
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

#ifndef Component_PIDL_GlobusError_h
#define Component_PIDL_GlobusError_h

#include <Component/PIDL/PIDLException.h>
#include <string>

namespace Component {
    namespace PIDL {
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
    }
}

#endif

//
// $Log$
// Revision 1.6  2000/03/23 20:43:06  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
// Revision 1.5  2000/03/23 10:27:35  sparker
// Added "name" method to match new Exception base class
//
// Revision 1.4  1999/09/24 20:15:57  sparker
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
