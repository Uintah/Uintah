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
 *  TypeInfo_internal.cc: internal representation for a type.
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/PIDL/TypeInfo_internal.h>
#include <Core/CCA/PIDL/TypeInfo.h>
#include <Core/Exceptions/InternalError.h>
using namespace SCIRun;

TypeInfo_internal::TypeInfo_internal(const std::string& fullclassname,
				     void* table,
				     int tableSize,
				     Object* (*create_proxy)(const ReferenceMgr&))
    : fullclassname(fullclassname), table(table),
	tableSize(tableSize), create_proxy(create_proxy), parentclass(0)
{
  // This is safe because it will not recurse - there are no parents yet.
  add_castable(this, TypeInfo::vtable_methods_start);
}

void TypeInfo_internal::add_parentclass(const TypeInfo* ti, int vtoffset)
{
  parentclass=const_cast<TypeInfo*>(ti);
  add_castable(ti->d_priv, vtoffset);
}

void TypeInfo_internal::add_parentiface(const TypeInfo* ti, int vtoffset)
{
  parent_ifaces.push_back(const_cast<TypeInfo*>(ti));
  add_castable(ti->d_priv, vtoffset);
}

void TypeInfo_internal::add_castable(const TypeInfo_internal* ti, int vtoffset)
{
    MapType::iterator iter_classname = classname_map.find(ti->fullclassname);
    if (iter_classname == classname_map.end()) {
	// Insert this...
	classname_map[ti->fullclassname] = MapType::mapped_type(ti, vtoffset);
	const_cast<TypeInfo_internal*>(ti)->add_castables(this, vtoffset);
    } else {
      // uuid not used to identify class types anymore
      //if(iter_classname->second.first->uuid != ti->uuid){
            //throw SCIRun::InternalError("inconsistent typeinfo");
      //} else {
	  // Ok...
      //}
    }
}

void TypeInfo_internal::add_castables(TypeInfo_internal* parent, int vtoffset)
{
  // This is an already constructed typeinfo.  We are supposed to add
  // our castable list to our parent
  for(MapType::iterator iter=classname_map.begin();
      iter != classname_map.end(); iter++){
    parent->add_castable(iter->second.first, iter->second.second+vtoffset-TypeInfo::vtable_methods_start);
  }
}

