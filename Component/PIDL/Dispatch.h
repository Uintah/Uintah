
/*
 *  Dispatch.h: A way to hide the globus handler table from outside
 * 	        of PIDL
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

#ifndef Component_PIDL_Dispatch_h
#define Component_PIDL_Dispatch_h

#include <globus_nexus.h>

namespace Component {
    namespace PIDL {
	class Dispatch {
	public:
	    Dispatch(globus_nexus_handler_t* table, int tableSize);
	    ~Dispatch();
	protected:
	    friend class Object_interface;
	    globus_nexus_handler_t* table;
	    int tableSize;
	};
    }
}

#endif
//
// $Log$
// Revision 1.1  1999/08/30 17:39:44  sparker
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
