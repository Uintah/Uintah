
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
#include <Component/PIDL/TypeSignature.h>

namespace Component {
    namespace PIDL {
	class ProxyBase {
	public:
	    bool isa(const TypeSignature& ts) const;
	    const Reference& getReference() const;
	protected:
	    ProxyBase(const Reference&);
	    virtual ~ProxyBase();
	    Reference d_ref;

	    Startpoint* getStartpoint() const;
	};
    }
}

#endif


//
// $Log$
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
