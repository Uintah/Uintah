
/*
 *  Reference.h: A serializable "pointer" to an object
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

#ifndef Component_PIDL_Reference_h
#define Component_PIDL_Reference_h

#include <globus_nexus.h>

namespace Component {
    namespace PIDL {
/**************************************
 
CLASS
   Reference
   
KEYWORDS
   Reference, PIDL
   
DESCRIPTION
   A remote reference.  This class is internal to PIDL and should not
   be used outside of PIDL or sidl generated code.  It contains a nexus
   startpoint and the vtable base offset.
****************************************/
	struct Reference {
	    //////////
	    // Empty constructor.  Initalizes the startpoint to nil
	    Reference();

	    //////////
	    // Copy the reference.  Does NOT copy the startpoint through
	    // globus_nexus_startpoint_copy
	    Reference(const Reference&);

	    //////////
	    // Copy the reference.  Does NOT copy the startpoint through
	    // globus_nexus_startpoint_copy
	    Reference& operator=(const Reference&);

	    //////////
	    // Destructor.  Does not destroy the startpoint.
	    ~Reference();

	    //////////
	    // Return the vtable base
	    int getVtableBase() const;

	    //////////
	    // The startpoint
	    globus_nexus_startpoint_t d_sp;

	    //////////
	    // The vtable base offset
	    int d_vtable_base;
	};
    }
}

#endif

//
// $Log$
// Revision 1.3  1999/09/24 20:03:36  sparker
// Added cocoon documentation
//
// Revision 1.2  1999/09/17 05:08:09  sparker
// Implemented component model to work with sidl code generator
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
