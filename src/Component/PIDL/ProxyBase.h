
/*
 *  ProxyBase.h: Base class for all PIDL proxies
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

#ifndef Component_PIDL_ProxyBase_h
#define Component_PIDL_ProxyBase_h

#include <Component/PIDL/Reference.h>

namespace Component {
    namespace PIDL {
	class TypeInfo;

/**************************************
 
CLASS
   ProxyBase
   
KEYWORDS
   Proxy, PIDL
   
DESCRIPTION
   The base class for all proxy objects.  It contains the reference to
   the remote object.  This class should not be used outside of PIDL
   or automatically generated sidl code.
****************************************/
	class ProxyBase {
	public:
	protected:
	    ////////////
	    // Create the proxy from the given reference.
	    ProxyBase(const Reference&);

	    ///////////
	    // Destructor
	    virtual ~ProxyBase();

	    //////////
	    // The reference to the remote object.
	    Reference d_ref;

	    //////////
	    // TypeInfo is a friend so that it can call _proxyGetReference
	    friend class TypeInfo;

	    //////////
	    // Return the internal reference.  If copy is true, the startpoint
	    // will be copied through globus_nexus_startpoint_copy, and
	    // will need to be destroyed with globus_nexus_startpoint_destroy
	    // or globus_nexus_put_startpoint_transfer.
	    void _proxyGetReference(Reference&, bool copy) const;
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
