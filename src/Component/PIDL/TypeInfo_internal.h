
/*
 *  TypeInfo_internal.h: internal representation for a type.
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

#ifndef Component_PIDL_TypeInfo_internal_h
#define Component_PIDL_TypeInfo_internal_h

#include <Component/PIDL/Object.h>
#include <globus_nexus.h>
#include <map>
#include <string>
#include <vector>

namespace Component {
    namespace PIDL {
	class TypeInfo;
	struct TypeInfo_internal {
	public:
	    TypeInfo_internal(const std::string& fullclassname,
			      const std::string& uudi,
			      globus_nexus_handler_t* table, int tableSize,
			      Object_interface* (*create_proxy)(const Reference&));
	    std::string fullclassname;
	    std::string uuid;
	    globus_nexus_handler_t* table;
	    int tableSize;
	    Object_interface* (*create_proxy)(const Reference&);

	    TypeInfo* parentclass;
	    void add_parentclass(const TypeInfo* ti, int vtoffset);

	    std::vector<TypeInfo*> parent_ifaces;
	    void add_parentiface(const TypeInfo* ti, int vtoffset);

	private:
	    friend class TypeInfo;
	    typedef std::map<std::string, std::pair<const TypeInfo_internal*, int> > MapType;
	    MapType classname_map;
	    void add_castable(const TypeInfo_internal* ti, int offset);
	    void add_castables(TypeInfo_internal* ti, int offset);

	};
    }
}

#endif

//
// $Log$
// Revision 1.1  1999/09/17 05:08:11  sparker
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
