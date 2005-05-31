/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#ifndef CCA_PIDL_TypeInfo_internal_h
#define CCA_PIDL_TypeInfo_internal_h

#include <Core/CCA/PIDL/ProxyBase.h>
#include <Core/CCA/PIDL/Object.h>
#include <sgi_stl_warnings_off.h>
#include <map>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
/**************************************
 
CLASS
   TypeInfo_internal
   
KEYWORDS
   TypeInfo, Introspection, PIDL
   
DESCRIPTION
   The internal representation of types in PIDL.  These objects are created
   by sidl generated code, and are used by the PIDL type facility and the
   SSIDL introspection facility.
****************************************/
	struct TypeInfo_internal {
	public:
	    //////////
	    // Create a new TypeInfo_internal. There should be only
	    // one of these per distinct type.  Fullclassname is the
	    // fully qualified C++ class name for the associated
	    // class (i.e. ::Core/CCA::PIDL::TypeInfo_internal).
	    // table and tableSize are used to create endpoints for
	    // objects of this type.  The create_proxy function will
	    // create a new proxy object for a proxy to this type.
	    TypeInfo_internal(const std::string& fullclassname,
			      void* table, int tableSize,
			      Object* (*create_proxy)(const SCIRun::ReferenceMgr&));

	    //////////
	    // The fully qualified classname for this type.
	    std::string fullclassname;

	    //////////
	    // The nexus handler table which dispatches the methods
	    // associated with this type.
	    void* table;

	    //////////
	    // The number of handlers in the nexus handler table.
	    int tableSize;

	    //////////
	    // The function to create a new proxy to this type.
	    Object* (*create_proxy)(const SCIRun::ReferenceMgr&);

	    //////////
	    // A pointer to the parent class type, if any.  If this
	    // is an interface or the root class (SSIDL.Object), this
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
} // End namespace SCIRun

#endif
