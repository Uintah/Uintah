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
 *  Object_proxy.h
 *
 *  
 */

#include <Core/CCA/PIDL/Object_proxy.h>
#include <Core/CCA/PIDL/TypeInfo.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/Comm/SocketMessage.h>

#include <iostream>
#include <string>

using namespace std;
using namespace SCIRun;

Object_proxy::Object_proxy(const Reference& ref)
    : ProxyBase(ref)
{
  rm.localSize = 1;
  rm.localRank = 0;
}

Object_proxy::Object_proxy(const URL& url)
{
  Reference *ref=new Reference();
  ref->chan->openConnection(url);
  ref->d_vtable_base=TypeInfo::vtable_methods_start;
  rm.insertReference(ref);

  rm.localSize = 1;
  rm.localRank = 0;
  rm.intracomm = NULL;
}

Object_proxy::Object_proxy(const int urlc, const URL urlv[], int mysize, int myrank)
{
  for(int i=0; i < urlc; i++) {
    Reference *ref = new Reference();
    ref->chan->openConnection(urlv[i]);
    ref->d_vtable_base=TypeInfo::vtable_methods_start;
    rm.insertReference(ref);
  }
  rm.localSize = mysize;
  rm.s_lSize = mysize;
  rm.localRank = myrank;

  if(mysize > 1)
    rm.intracomm = PIDL::getIntraComm();
  else
    rm.intracomm = NULL;
}

Object_proxy::Object_proxy(const std::vector<URL>& urlv, int mysize, int myrank)
{
  std::vector<URL>::const_iterator iter = urlv.begin();
  for(unsigned int i=0; i < urlv.size(); i++, iter++) {
    Reference *ref = new Reference();
    ref->chan->openConnection(*iter);
    ref->d_vtable_base=TypeInfo::vtable_methods_start;
    rm.insertReference(ref);
  }
  rm.localSize = mysize;
  rm.s_lSize = mysize;
  rm.localRank = myrank;

  if(mysize > 1)
    rm.intracomm = PIDL::getIntraComm();
  else
    rm.intracomm = NULL;
}

Object_proxy::Object_proxy(const std::vector<Object::pointer>& pxy, int mysize, int myrank)
{
  std::vector<Object::pointer>::const_iterator iter = pxy.begin();
  for(unsigned int i=0; i < pxy.size(); i++, iter++) {
    ProxyBase* pbase = dynamic_cast<ProxyBase* >((*iter).getPointer());
    refList* refL = pbase->_proxyGetReferenceMgr()->getAllReferences();
    refList::const_iterator riter = refL->begin();
    for(unsigned int i=0; i < refL->size(); i++, riter++) {
      Reference *ref = (*riter)->clone();

      Message *message=ref->chan->getMessage();
      message->createMessage();
      message->sendMessage(-101); //addReference;

      rm.insertReference(ref);
    }
  }

  rm.localSize = mysize;
  rm.s_lSize = mysize;
  rm.localRank = myrank;

  if(mysize > 1)
    rm.intracomm = PIDL::getIntraComm();
  else
    rm.intracomm = NULL;
}


Object_proxy::~Object_proxy()
{
}















