
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

	class PIDL {
	public:
	    static void initialize(int argc, char* argv[]);
	    static Wharehouse* getWharehouse();
	    static Object objectFrom(const URL&);
	    static Object objectFrom(const Reference&);
	    static void serveObjects();
	    static std::string getBaseURL();
	protected:
	private:
	    static Wharehouse* wharehouse;
	};
    }
}

#endif

//
// $Log$
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
