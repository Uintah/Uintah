
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

#include <Component/PIDL/Object_proxy.h>
#include <Component/PIDL/GlobusError.h>
#include <Component/PIDL/TypeInfo.h>
#include <Component/PIDL/URL.h>
#include <iostream>
#include <string>

using Component::PIDL::GlobusError;
using Component::PIDL::Object_proxy;
using Component::PIDL::TypeInfo;

Object_proxy::Object_proxy(const Reference& ref)
    : ProxyBase(ref)
{
}

Object_proxy::Object_proxy(const URL& url)
    : ProxyBase(Reference())
{
    std::string s(url.getString());
    d_ref.d_vtable_base=TypeInfo::vtable_methods_start;
    char* str=const_cast<char*>(s.c_str());
    if(int gerr=globus_nexus_attach(str, &d_ref.d_sp)){
	d_ref.d_vtable_base=TypeInfo::vtable_invalid;
	throw GlobusError("nexus_attach", gerr);
    }
}

Object_proxy::~Object_proxy()
{
}

//
// $Log$
// Revision 1.4  1999/09/26 06:12:56  sparker
// Added (distributed) reference counting to PIDL objects.
// Began campaign against memory leaks.  There seem to be no more
//   per-message memory leaks.
// Added a test program to flush out memory leaks
// Fixed other Component testprograms so that they work with ref counting
// Added a getPointer method to PIDL handles
//
// Revision 1.3  1999/09/17 05:08:08  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.2  1999/08/31 08:59:01  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
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
