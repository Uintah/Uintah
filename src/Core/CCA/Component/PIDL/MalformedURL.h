
/*
 *  MalformedURL.h: Base class for PIDL Exceptions
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

#ifndef Component_PIDL_MalformedURL_h
#define Component_PIDL_MalformedURL_h

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace Component {
    namespace PIDL {
/**************************************
 
CLASS
   MalformedURL
   
KEYWORDS
   Exception, Error, PIDL, URL
   
DESCRIPTION
   Exception class for an unintelligible URL.  This results from
   a syntax error in the URL.  See InvalidReference for a properly
   formed URL that doesn't map to a valid object.

****************************************/
	class MalformedURL : public SCICore::Exceptions::Exception {
	public:
	    //////////
	    // Contruct the object, giving the offending URL and an
	    // explanation of the error
	    MalformedURL(const std::string& url, const std::string& error);

	    //////////
	    // Destructor
	    virtual ~MalformedURL();

	    //////////
	    // Return a human readable explanation
	    virtual const char* message() const;

	    //////////
	    // Return the name of this class
	    virtual const char* type() const;
	protected:
	private:
	    //////////
	    // The offending URL
	    std::string d_url;

	    //////////
	    // The error explanation
	    std::string d_error;

	    //////////
	    // The "complete" message
	    std::string d_msg;
	};
    }
}

#endif

//
// $Log$
// Revision 1.6  2000/03/23 10:27:36  sparker
// Added "name" method to match new Exception base class
//
// Revision 1.5  1999/09/24 20:15:58  sparker
// Cocoon documentation updates
//
// Revision 1.4  1999/09/24 06:26:25  sparker
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
// Revision 1.3  1999/09/17 05:08:07  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.2  1999/08/31 08:59:00  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.1  1999/08/30 17:39:46  sparker
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
