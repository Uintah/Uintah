
/*
 *  TypeInfo.h: internal representation for a type.
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

#ifndef Component_PIDL_TypeInfo_h
#define Component_PIDL_TypeInfo_h

#include <string>

namespace Component {
    namespace PIDL {
	class Object_interface;
	class TypeInfo_internal;

	class TypeInfo {
	public:
	    TypeInfo(TypeInfo_internal*);
	    ~TypeInfo();
	    int computeVtableOffset(const TypeInfo*) const;
	    Object_interface* pidl_cast(Object_interface*) const;
	    int isa(const std::string& classname, const std::string& uuid) const;

	    static const int vtable_invalid = -1234;
	    /* 
	     * Do not change this number, unless the corresponding
	     * code generation is changed in the sidl compiler.
	     */
	    static const int vtable_methods_start = 2;
	protected:
	private:
	    static const int vtable_isa_handler = 0;
	    friend class TypeInfo_internal;
	    friend class ServerContext;
	    TypeInfo_internal* d_priv;
	};
    }
}

#endif

//
// $Log$
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
// Revision 1.2  1999/09/17 05:08:10  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.1  1999/08/30 17:39:49  sparker
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
