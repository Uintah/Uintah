
/*
 *  Object.h: Base class for all PIDL distributed objects
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

#ifndef Component_PIDL_Object_proxy_h
#define Component_PIDL_Object_proxy_h

#include <Component/PIDL/ProxyBase.h>
#include <Component/PIDL/Object.h>

namespace Component {
    namespace PIDL {
/**************************************
 
CLASS
   Object_proxy
   
KEYWORDS
   Proxy, Object, PIDL
   
DESCRIPTION
   Internal PIDL class for a proxy to a base object.  This impements
   the Object_interface interface and provides a proxy mechanism for
   remote objects.  Since there are no interesting methods at this level,
   the only interesting thing that we can do is up-cast.
****************************************/
	class Object_proxy : public ProxyBase, public Object_interface {
	public:
	protected:
	    friend class PIDL;
	    //////////
	    // Private constructor from a reference
	    Object_proxy(const Reference&);

	    //////////
	    // Private constructor from a URL
	    Object_proxy(const URL&);

	    //////////
	    // Destructor
	    virtual ~Object_proxy();
	private:
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
