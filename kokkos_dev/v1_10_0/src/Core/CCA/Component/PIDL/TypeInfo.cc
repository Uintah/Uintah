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
 *  TypeInfo.cc: internal representation for a type.
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include "TypeInfo.h"
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/ProxyBase.h>
#include <Core/CCA/Component/Comm/Message.h>
#include <Core/CCA/Component/Comm/ReplyEP.h>
#include <Core/CCA/Component/PIDL/TypeInfo_internal.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
using namespace SCIRun;
using std::cerr;

TypeInfo::TypeInfo(TypeInfo_internal* priv)
    : d_priv(priv)
{
}

TypeInfo::~TypeInfo()
{
  delete d_priv;
}

int TypeInfo::computeVtableOffset(const TypeInfo* ti) const
{
  TypeInfo_internal::MapType::iterator iter=d_priv->classname_map.find(ti->d_priv->fullclassname);
  if(iter == d_priv->classname_map.end()){
    throw SCIRun::InternalError("computeVtableOffset: "+ti->d_priv->fullclassname+" should be an ancestor of "+d_priv->fullclassname+", but is not!");
  }
  if(iter->second.first->uuid != ti->d_priv->uuid)
    throw SCIRun::InternalError("UUID mismatch in computeVtableOffset");
  return iter->second.second-vtable_methods_start;
}


Object* TypeInfo::pidl_cast(Object* obj) const
{
  // If we aren't a proxy, we don't know what to do...
  ProxyBase* p=dynamic_cast<ProxyBase*>(obj);
  if(!p)
    return 0;

  refList _refL;
  p->_proxyGetReferenceList(_refL,false); 
  Message* message = _refL[0].chan->getMessage();

  // Get a startpoint ready for the reply
  message->createMessage();

  // Size the message
  int classname_size=d_priv->fullclassname.length();
  int uuid_size=d_priv->uuid.length();

  // Pack the message
  message->marshalInt(&classname_size);
  message->marshalChar(const_cast<char*>(d_priv->fullclassname.c_str()),classname_size);
  message->marshalInt(&uuid_size); 
  message->marshalChar(const_cast<char*>(d_priv->uuid.c_str()),uuid_size);
 
  int addRef=1; // Tell the isa handler to increment the ref count on the object
  message->marshalInt(&addRef);

  // Send the message
  int handler=vtable_isa_handler;
  message->sendMessage(handler);
  // Wait for the reply
  message->waitReply();

  // Unpack the reply
  int flag;
  message->unmarshalInt(&flag);
  int vtbase;
  message->unmarshalInt(&vtbase);
  message->destroyMessage();

  if(!flag){
    // isa failed
    return 0;
  } else {
    // isa succeeded, return the correct proxy
    refList new_refL;
    new_refL = _refL;
    for(unsigned int i=0; i < new_refL.size(); i++)
      new_refL[i].d_vtable_base=vtbase;
    return (*d_priv->create_proxy)(new_refL);
  }
}


int TypeInfo::isa(const std::string& classname, const std::string& uuid) const
{
  TypeInfo_internal::MapType::iterator classname_iter=d_priv->classname_map.find(classname);
  if(classname_iter == d_priv->classname_map.end())
    return vtable_invalid;
  const TypeInfo_internal* tip=classname_iter->second.first;
  if(tip->uuid != uuid){
    cerr << "Warning classname is the same, but uuid is different!\n";
    cerr << "class = " << classname << "\n";
    cerr << "remote uuid=" << uuid << "\n";
    cerr << "local uuid=" << tip->uuid << "\n";
  }
  return classname_iter->second.second;
}

