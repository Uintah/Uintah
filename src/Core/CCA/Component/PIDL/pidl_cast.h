
/*
 *  pidl_cast.h: The equivalent of dynamic_cast for network based objects
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

#ifndef Component_PIDL_pidl_cast_h
#define Component_PIDL_pidl_cast_h

// In global namespace for now...
#include <Component/PIDL/Object.h>
#include <Component/PIDL/TypeInfo.h>
#include <SCICore/Exceptions/InternalError.h>

/**************************************
 
CLASS
   pidl_cast
   
KEYWORDS
   pidl_cast, dynamic_cast
   
DESCRIPTION
   An equivalent to dynamic_cast for (possibly) remote objects.  This
   tries to upcast locally, but will send a remote procedure call
   (in TypeInfo::pidl_cast) if required.
****************************************/
template<class T>
T pidl_cast(const Component::PIDL::Object& obj)
{
    // Try the direct cast before we go remote
    T::interfacetype* iface=dynamic_cast<T::interfacetype*>(obj.getPointer());
    if(iface)
        return iface;

    const Component::PIDL::TypeInfo* typeinfo = T::_getTypeInfo();
    Component::PIDL::Object_interface* result=typeinfo->pidl_cast(obj.getPointer());
    if(result){
	T p=dynamic_cast<T::interfacetype*>(result);
	if(!p)
	    throw SCICore::Exceptions::InternalError("TypeInfo::pidl_cast returned wrong object!");
	return p;
    } else {
	return 0;
    }
}

#endif

//
// $Log$
// Revision 1.4  1999/09/26 06:12:57  sparker
// Added (distributed) reference counting to PIDL objects.
// Began campaign against memory leaks.  There seem to be no more
//   per-message memory leaks.
// Added a test program to flush out memory leaks
// Fixed other Component testprograms so that they work with ref counting
// Added a getPointer method to PIDL handles
//
// Revision 1.3  1999/09/24 20:03:39  sparker
// Added cocoon documentation
//
// Revision 1.2  1999/09/17 05:08:11  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.1  1999/08/30 17:39:50  sparker
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
