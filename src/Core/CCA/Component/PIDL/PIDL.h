
/*
 *  PIDL.h: Include a bunch of PIDL files for external clients
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

#ifndef Component_PIDL_PIDL_h
#define Component_PIDL_PIDL_h

#include <Component/PIDL/Object.h>
#include <Component/PIDL/PIDLException.h>
#include <Component/PIDL/URL.h>
#include <Component/PIDL/pidl_cast.h>
#include <string>

namespace Component {
    namespace PIDL {
	class Reference;
	class URL;
	class Wharehouse;

/**************************************
 
CLASS
   PIDL
   
KEYWORDS
   PIDL
   
DESCRIPTION
   A class to encapsulate several static methods for PIDL.
****************************************/
	class PIDL {
	public:
	    //////////
	    // Initialize PIDL
	    static void initialize(int argc, char* argv[]);

	    //////////
	    // Create a base Object class from the given URL
	    static Object objectFrom(const URL&);

	    //////////
	    // Create a base Object class from the given Reference
	    static Object objectFrom(const Reference&);

	    //////////
	    // Go into the main loop which services requests for all
	    // objects.  This will not return until all objects have
	    // been destroyed.
	    static void serveObjects();

	    //////////
	    // Return the URL for the current process.  Individual
	    // objects may be identified by appending their id number
	    // or name to the end of the string.
	    static std::string getBaseURL();

	    //////////
	    // Return the object Wharehouse.  Most clients will not
	    // need to use this.
	    static Wharehouse* getWharehouse();
	protected:
	private:
	    //////////
	    // The wharehouse singleton object
	    static Wharehouse* wharehouse;

	    //////////
	    // Private constructor to prevent creation of a PIDL
	    PIDL();
	};
    }
}

#endif

//
// $Log$
// Revision 1.3  1999/09/24 20:03:35  sparker
// Added cocoon documentation
//
// Revision 1.2  1999/09/17 05:08:08  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.1  1999/08/30 17:39:47  sparker
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
