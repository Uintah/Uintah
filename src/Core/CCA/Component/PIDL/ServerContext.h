
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

#include <Component/PIDL/Object.h>
#include <globus_nexus.h>

namespace Component {
    namespace PIDL {
	class TypeInfo;
/**************************************
 
CLASS
   ServerContext
   
KEYWORDS
   PIDL, Server
   
DESCRIPTION
   One of these objects is associated with each server object.  It provides
   the state necessary for the PIDL internals, including the endpoint
   associated with this object, a pointer to type information, the objects
   id from the object wharehouse, a pointer to the base object class
   (Object_interface), and to the most-derived type (a void*).  Endpoints
   are created lazily from the Object_interface class.
****************************************/
	struct ServerContext {
	    //////////
	    // A pointer to the type information.
	    const TypeInfo* d_typeinfo;

	    //////////
	    // The endpoint associated with this object.
	    globus_nexus_endpoint_t d_endpoint;

	    //////////
	    // The ID of this object from the object wharehouse.  This
	    // id is unique within this process.
	    int d_objid;

	    //////////
	    // A pointer to the object base class.
	    Object_interface* d_objptr;

	    //////////
	    // A pointer to the most derived type.  This is used only by
	    // sidl generated code.
	    void* d_ptr;

	    //////////
	    // A flag, true if the endpoint has been created for this
	    // object.
	    bool d_endpoint_active;

	    //////////
	    // Create the endpoint for this object.
	    void activateEndpoint();
	};
    }
}

#endif

//
// $Log$
// Revision 1.4  1999/09/24 20:03:37  sparker
// Added cocoon documentation
//
// Revision 1.3  1999/09/24 06:26:26  sparker
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
// Revision 1.2  1999/09/21 06:13:00  sparker
// Fixed bugs in multiple inheritance
// Added round-trip optimization
// To support this, we store Startpoint* in the endpoint instead of the
//    object final type.
//
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
