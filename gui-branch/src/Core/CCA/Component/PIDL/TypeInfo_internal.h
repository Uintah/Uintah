/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  TypeInfo_internal.h: internal representation for a type.
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

#include <Core/CCA/Component/PIDL/Object.h>
#include <globus_nexus.h>
#include <map>
#include <string>
#include <vector>

namespace PIDL {
/**************************************
 
CLASS
   TypeInfo_internal
   
KEYWORDS
   TypeInfo, Introspection, PIDL
   
DESCRIPTION
   The internal representation of types in PIDL.  These objects are created
   by sidl generated code, and are used by the PIDL type facility and the
   CIA introspection facility.
****************************************/
	struct TypeInfo_internal {
	public:
	    //////////
	    // Create a new TypeInfo_internal. There should be only
	    // one of these per distinct type.  Fullclassname is the
	    // fully qualified C++ class name for the associated
	    // class (i.e. ::Core/CCA/Component::PIDL::TypeInfo_internal).
	    // uuid is a unique identifier for this class.  table and
	    // tableSize are used to create endpoints for objects
	    // of this type.  The create_proxy function will create
	    // a new proxy object for a proxy to this type.
	    TypeInfo_internal(const std::string& fullclassname,
			      const std::string& uuid,
			      globus_nexus_handler_t* table, int tableSize,
			      Object_interface* (*create_proxy)(const Reference&));

	    //////////
	    // The fully qualified classname for this type.
	    std::string fullclassname;

	    //////////
	    // The globally unique identifier associated with this ID.
	    std::string uuid;

	    //////////
	    // The nexus handler table which dispatches the methods
	    // associated with this type.
	    globus_nexus_handler_t* table;

	    //////////
	    // The number of handlers in the nexus handler table.
	    int tableSize;

	    //////////
	    // The function to create a new proxy to this type.
	    Object_interface* (*create_proxy)(const Reference&);

	    //////////
	    // A pointer to the parent class type, if any.  If this
	    // is an interface or the root class (CIA.Object), this
	    // will be 0.
	    TypeInfo* parentclass;

	    //////////
	    // Add the parent class with the given type information
	    // and an offset into the nexus handler table.
	    void add_parentclass(const TypeInfo* ti, int vtoffset);

	    //////////
	    // Pointers to the parent interfaces associated with this
	    // class or interface.
	    std::vector<TypeInfo*> parent_ifaces;

	    //////////
	    // Add a new parent interface, given the offset into the
	    // nexus handler table.
	    void add_parentiface(const TypeInfo* ti, int vtoffset);

	private:

	    //////////
	    // TypeInfo methods need access to these internals.
	    friend class TypeInfo;

	    //////////
	    // The type used for the classname to offset map.
	    typedef std::map<std::string, std::pair<const TypeInfo_internal*, int> > MapType;

	    //////////
	    // A mapping from class names to TypeInfo and vtable offsets.
	    MapType classname_map;

	    //////////
	    // Called by add_parentiface and add_parentclass to create
	    // an entry in the classname_map for the parent and all
	    // ancestors.  This is mutually recurisve with add_castables.
	    void add_castable(const TypeInfo_internal* ti, int offset);

	    //////////
	    // Add the parent classes to the classname_map in ti.
	    void add_castables(TypeInfo_internal* ti, int offset);

	};
} // End namespace PIDL

#endif

