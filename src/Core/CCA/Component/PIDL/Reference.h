
/*
 *  Reference.h: A serializable "pointer" to an object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core/CCA/Component_PIDL_Reference_h
#define Core/CCA/Component_PIDL_Reference_h

#include <globus_nexus.h>

namespace SCIRun {
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
} // End namespace SCIRun

#endif

