
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

#include <Component/PIDL/Reference.h>
#include <Component/PIDL/TypeInfo.h>

using Component::PIDL::Reference;

Reference::Reference()
{
    d_vtable_base=TypeInfo::vtable_invalid;
}

Reference::Reference(const Reference& copy)
    : d_sp(copy.d_sp), d_vtable_base(copy.d_vtable_base)
{
}

Reference::~Reference()
{
}

Reference& Reference::operator=(const Reference& copy)
{
    d_sp=copy.d_sp;
    d_vtable_base=copy.d_vtable_base;
    return *this;
}

int Reference::getVtableBase() const
{
    return d_vtable_base;
}

//
// $Log$
// Revision 1.4  1999/09/21 06:13:00  sparker
// Fixed bugs in multiple inheritance
// Added round-trip optimization
// To support this, we store Startpoint* in the endpoint instead of the
//    object final type.
//
// Revision 1.3  1999/09/17 05:08:09  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.2  1999/08/31 08:59:02  sparker
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
