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
#include <Core/CCA/Component/PIDL/ProxyBase.h>
#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <iostream>
#include <Core/CCA/tools/sidl/uuid_wrapper.h>

#if HAVE_MPI || HAVE_MPICH
#define TRANSFER_UUID 1
#include <mpi.h>
#endif

using namespace SCIRun;

ProxyBase::ProxyBase() 
: proxy_uuid("NONE") { }

ProxyBase::ProxyBase(const Reference& ref)
: proxy_uuid("NONE")
{ 
  rm.insertReference(ref);
}

ProxyBase::ProxyBase(const ReferenceMgr& refM)
: proxy_uuid("NONE")
{ 
  rm = refM;
}

ProxyBase::~ProxyBase()
{
}

void ProxyBase::_proxyGetReference(Reference& ref, bool copy) const
{
  if (copy) {
    /*Clean up the passed reference just in case*/
    if (ref.chan != NULL) {
      delete (ref.chan);
      ref.chan = NULL;
    }

    ref.chan = (rm.d_ref[0].chan)->SPFactory(true);
  }
  else {
    ref = (rm.d_ref[0]);
  }
}

void ProxyBase::_proxyGetReferenceList(refList& ref, bool copy) const
{
  if (copy) {
    /*Clean up the passed reference just in case*/
    refList::iterator iter = ref.begin();
    for(unsigned int i=0; i < ref.size(); i++, iter++) {      
      if ((*iter).chan != NULL) {
	delete ((*iter).chan);
	(*iter).chan = NULL;
      }
    }

    ref = (rm.d_ref);
    for(unsigned int i=0; i < ref.size(); i++) {      
      ref[i].chan = (rm.d_ref[i].chan)->SPFactory(true);
    }
  }
  else {
    ref = (rm.d_ref); 
  }
}

ReferenceMgr* ProxyBase::_proxyGetReferenceMgr()  
{
  return (&rm);
}

::std::string ProxyBase::getProxyUUID()
{
  if(proxy_uuid == "NONE") {
    if(rm.localRank == 0) {
//    proxy_uuid = getUUID(); 
    }
#ifdef TRANSFER_UUID
//    MPI_Bcast(const_cast<char*>(proxy_uuid.c_str()),64,MPI_CHAR,0,MPI_COMM_WORLD);
#endif
  }
  return proxy_uuid;
}


