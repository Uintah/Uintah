
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

#include <Component/PIDL/ProxyBase.h>
#include <Component/PIDL/GlobusError.h>
#include <globus_nexus.h>

using Component::PIDL::ProxyBase;
using Component::PIDL::Reference;

ProxyBase::ProxyBase(const Reference& ref)
    : d_ref(ref)
{
}

ProxyBase::~ProxyBase()
{
}

void ProxyBase::_proxyGetReference(Reference& ref, bool copy) const
{
    ref=d_ref;
    if(copy){
	if(int gerr=globus_nexus_startpoint_copy(&ref.d_sp, const_cast<globus_nexus_startpoint_t*>(&d_ref.d_sp)))
	    throw GlobusError("startpoint_copy", gerr);
    }
}

//
// $Log$
// Revision 1.3  1999/09/17 05:08:09  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.2  1999/08/31 08:59:01  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
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
