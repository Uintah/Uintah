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
#include <Core/CCA/Comm/CommError.h>
#include <Core/CCA/PIDL/Object.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/ProxyBase.h>
#include <Core/CCA/Comm/Message.h>
#include <Core/CCA/PIDL/TypeInfo_internal.h>
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
  if (iter == d_priv->classname_map.end()) {
    throw SCIRun::InternalError("computeVtableOffset: " +
	ti->d_priv->fullclassname +
	" should be an ancestor of " +
	d_priv->fullclassname +
	", but is not!");
  }
  return iter->second.second-vtable_methods_start;
}


Object* TypeInfo::pidl_cast(Object* obj) const
{
  // If we aren't a proxy, we don't know what to do...
  ProxyBase* p=dynamic_cast<ProxyBase*>(obj);
  if(!p)
    return 0;

  ReferenceMgr* _rm;
  _rm = p->_proxyGetReferenceMgr(); 
  Message* message = _rm->d_ref[0]->chan->getMessage();

  // Get a startpoint ready for the reply
  message->createMessage();

  // Size the message
  int classname_size=d_priv->fullclassname.length();

  // Pack the message
  message->marshalInt(&classname_size);
  message->marshalChar(const_cast<char*>(d_priv->fullclassname.c_str()),classname_size);

  int addRef=1; //Tell the isa handler to increment the ref count on the object
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
    // isa succeeded 
    // addReference to the other processes in case it is a parallel component
    ::std::vector < ::SCIRun::Message*> save_callnoret_msg;
    for(unsigned int i=1; i < _rm->d_ref.size(); i++) {
      /*CALLNORET*/
      message = _rm->d_ref[i]->chan->getMessage();
      message->createMessage();
      //Marshal flag which informs handler that
      // this message is CALLNORET
      ::SCIRun::callType _flag = ::SCIRun::CALLNORET;
      message->marshalInt((int*)&_flag);
      //Marshal the sessionID and number of actual calls from this proxy
      ProxyID _sessionID = p->getProxyUUID();
      message->marshalInt(&_sessionID.iid);
      message->marshalInt(&_sessionID.pid);
      //CALLNORET always sends (1 call + redis) number of calls per callee proc.
      int _numCalls = 1;
      message->marshalInt(&_numCalls);
      // Send the message
      int _handler= _rm->d_ref[i]->getVtableBase()+0;
      message->sendMessage(_handler);
      save_callnoret_msg.push_back(message);
    }

    for(int i=0; i<save_callnoret_msg.size(); i++){
      int _x_flag;
      save_callnoret_msg[i]->waitReply();
      save_callnoret_msg[i]->unmarshalInt(&_x_flag);
      save_callnoret_msg[i]->destroyMessage();
      if(_x_flag != 0) {
	throw ::SCIRun::InternalError("Unexpected user exception");
      }
    }

    // return the correct proxy
    ReferenceMgr new_rm(*_rm);
    for(unsigned int i=0; i < new_rm.d_ref.size(); i++)
      new_rm.d_ref[i]->d_vtable_base=vtbase;
    return (*d_priv->create_proxy)(new_rm);
  }
}


int TypeInfo::isa(const std::string& classname /*, const std::string& uuid*/) const
{
  TypeInfo_internal::MapType::iterator classname_iter =
    d_priv->classname_map.find(classname);
  if (classname_iter == d_priv->classname_map.end()) {
    return vtable_invalid;
  }
  const TypeInfo_internal* tip=classname_iter->second.first;
  return classname_iter->second.second;
}

