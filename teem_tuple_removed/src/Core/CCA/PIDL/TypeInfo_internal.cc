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
				     const std::string& uuid,
				     void* table,
				     int tableSize,
				     Object* (*create_proxy)(const ReferenceMgr&))
    : fullclassname(fullclassname), uuid(uuid),
      table(table), tableSize(tableSize), create_proxy(create_proxy),
      parentclass(0)
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
  MapType::iterator iter_classname=classname_map.find(ti->fullclassname);
  if(iter_classname == classname_map.end()){
    // Insert this...
    classname_map[ti->fullclassname]=MapType::mapped_type(ti, vtoffset);
    const_cast<TypeInfo_internal*>(ti)->add_castables(this, vtoffset);
  } else {
    if(iter_classname->second.first->uuid != ti->uuid){
      throw SCIRun::InternalError("inconsistent typeinfo");
    } else {
      // Ok...
    }
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

