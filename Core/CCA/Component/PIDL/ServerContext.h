
/*
 *  ServerContext.h: Local class for PIDL that holds the context
 *                   for server objects
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

#ifndef Component_PIDL_ServerContext_h
#define Component_PIDL_ServerContext_h

#include <globus_nexus.h>

namespace Component {
    namespace PIDL {
	class TypeInfo;
	struct ServerContext {
	    const TypeInfo* d_typeinfo;
	    globus_nexus_endpoint_t d_endpoint;
	    int d_objid;
	};
    }
}

#endif

//
// $Log$
// Revision 1.1  1999/08/30 17:39:48  sparker
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
