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
 *  ProxyBase.cc: Base class for all PIDL proxies
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */
#include <Core/Thread/Thread.h>
#include <Core/CCA/PIDL/ProxyBase.h>
#include <Core/CCA/PIDL/TypeInfo.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <iostream>
#include <Core/CCA/tools/sidl/uuid_wrapper.h>
#include <assert.h>

using namespace SCIRun;

ProxyBase::ProxyBase() 
: proxy_uuid("NONENONENONENONENONENONENONENONENONE") 
{
  xr = new XceptionRelay(this);
}

//remove it later
ProxyBase::ProxyBase(const Reference& ref)
: proxy_uuid("NONENONENONENONENONENONENONENONENONE")
{ 
  xr = new XceptionRelay(this);
  rm.insertReference( ((Reference*)&ref)->clone());
}

ProxyBase::ProxyBase(Reference *ref)
: proxy_uuid("NONENONENONENONENONENONENONENONENONE")
{ 
  xr = new XceptionRelay(this);
  rm.insertReference(ref);
}

ProxyBase::ProxyBase(const ReferenceMgr& refM)
: proxy_uuid("NONENONENONENONENONENONENONENONENONE"),
  rm(refM)
{ 
  xr = new XceptionRelay(this);
}

ProxyBase::~ProxyBase()
{
  /*Close all connections*/
  refList::iterator iter = rm.d_ref.begin();
  for(unsigned int i=0; i < rm.d_ref.size(); i++, iter++) {
    (*iter)->chan->closeConnection();
  }
  /*Delete intercommunicator*/
  if((rm.localSize > 1)&&(rm.intracomm != NULL)) {
    delete (rm.intracomm);
    rm.intracomm = NULL; 
  }  
  /*Delete exception relay*/
  if(xr) {
    delete xr;
  }
}

void ProxyBase::_proxyGetReference(Reference& ref, bool copy) const
{
  if (copy) {
    rm.getIndependentReference()->cloneTo(ref);
  }
  else {
    ref = *(rm.getIndependentReference()); 
  }
}

ReferenceMgr* ProxyBase::_proxyGetReferenceMgr() const
{
  return (ReferenceMgr*)(&rm);
}

::std::string ProxyBase::getProxyUUID()
{
  if(proxy_uuid == "NONENONENONENONENONENONENONENONENONE") {
    if(rm.getRank() == 0) {
      proxy_uuid = getUUID(); 
    }
    if(rm.getSize() > 1) { 
      // Exchange component ID among all parallel processes
      //std::cout << rm.getRank() << "='" << proxy_uuid.c_str() << "'=" << proxy_uuid.size() << "\n";
      (rm.intracomm)->broadcast(0,const_cast<char*>(proxy_uuid.c_str()),36);
    }
  }
  return proxy_uuid;
}

void ProxyBase::_proxycreateSubset(int localsize, int remotesize)
{
  if(proxy_uuid == "NONENONENONENONENONENONENONENONENONE") {
    getProxyUUID();
  }
  rm.createSubset(localsize,remotesize);
}

