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

#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <Core/CCA/Component/PIDL/GlobusError.h>
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/ProxyBase.h>
#include <Core/CCA/Component/PIDL/ReplyEP.h>
#include <Core/CCA/Component/PIDL/TypeInfo_internal.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>

using std::cerr;

using PIDL::GlobusError;
using PIDL::Object;
using PIDL::TypeInfo;

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
  PIDL::ProxyBase* p=dynamic_cast<PIDL::ProxyBase*>(obj);
  if(!p)
    return 0;

  // Get a startpoint ready for the reply
  ReplyEP* reply=ReplyEP::acquire();
  globus_nexus_startpoint_t sp;
  reply->get_startpoint_copy(&sp);

  // Size the message
  int classname_size=d_priv->fullclassname.length();
  int uuid_size=d_priv->uuid.length();
  int size=globus_nexus_sizeof_startpoint(&sp, 1)+
    globus_nexus_sizeof_int(3)+
    globus_nexus_sizeof_char(classname_size+uuid_size);

  // Pack the message
  globus_nexus_buffer_t buffer;
  if(int gerr=globus_nexus_buffer_init(&buffer, size, 0))
    throw GlobusError("buffer_init", gerr);
  globus_nexus_put_int(&buffer, &classname_size, 1);
  globus_nexus_put_char(&buffer, const_cast<char*>(d_priv->fullclassname.c_str()),
			classname_size);
  globus_nexus_put_int(&buffer, &uuid_size, 1);
  globus_nexus_put_char(&buffer, const_cast<char*>(d_priv->uuid.c_str()),
			uuid_size);
  int addRef=1; // Tell the isa handler to increment the ref count on the object
  globus_nexus_put_int(&buffer, &addRef, 1);
  globus_nexus_put_startpoint_transfer(&buffer, &sp, 1);

    // Send the message
  Reference ref;
  p->_proxyGetReference(ref, false);
  int handler=vtable_isa_handler;
  if(int gerr=globus_nexus_send_rsr(&buffer, &ref.d_sp,
				    handler, GLOBUS_TRUE, GLOBUS_FALSE))
    throw GlobusError("send_rsr", gerr);

    // Wait for the reply
  globus_nexus_buffer_t recvbuff=reply->wait();

  // Unpack the reply
  int flag;
  globus_nexus_get_int(&recvbuff, &flag, 1);
  int vtbase;
  globus_nexus_get_int(&recvbuff, &vtbase, 1);
  ReplyEP::release(reply);
  if(int gerr=globus_nexus_buffer_destroy(&recvbuff))
    throw GlobusError("buffer_destroy", gerr);

  if(!flag){
    // isa failed
    return 0;
  } else {
    // isa succeeded, return the correct proxy
    Reference new_ref;
    if(int gerr=globus_nexus_startpoint_copy(&new_ref.d_sp, &ref.d_sp))
      throw GlobusError("startpoint_copy", gerr);
    new_ref.d_vtable_base=vtbase;

    return (*d_priv->create_proxy)(new_ref);
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

