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
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Core/CCA/PIDL/ReferenceMgr.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <iostream>
using namespace SCIRun;

ReferenceMgr::ReferenceMgr()
{
  localSize=1; 
  localRank=0;
  s_lSize=1;
  s_refSize=0;
}

ReferenceMgr::ReferenceMgr(int rank, int size)
  :localSize(size), localRank(rank), s_lSize(size), s_refSize(0)
{
}

ReferenceMgr::ReferenceMgr(const ReferenceMgr& copy)
  :localSize(copy.localSize),localRank(copy.localRank),
   s_lSize(copy.s_lSize), s_refSize(copy.s_refSize)
{
  intracomm = PIDL::getIntraComm();
  for(unsigned int i=0; i < copy.d_ref.size(); i++) {
    d_ref.push_back(copy.d_ref[i]->clone());
  }
}

ReferenceMgr::~ReferenceMgr()
{
  refList::iterator iter = d_ref.begin();
  for(unsigned int i=0; i < d_ref.size(); i++, iter++) {
    delete (*iter);
  }
}

Reference* ReferenceMgr::getIndependentReference() const
{
  return d_ref[localRank % s_refSize];
}

::std::vector<Reference*> ReferenceMgr::getCollectiveReference(callType tip)
{
  ::std::vector<Reference*> refPtrList;
  refList::iterator iter;

  /*Sanity check:*/
  if (localRank >= s_lSize) return refPtrList;


  switch(tip) {
  case (CALLONLY):
  case (NOCALLRET):
    refPtrList.insert(refPtrList.begin(), d_ref[localRank % s_refSize]);
    break;
  case (CALLNORET):
    iter = d_ref.begin() + (localRank + s_lSize);
    for(int i=(localRank + s_lSize); i < (int)s_refSize; i+=s_lSize, iter+=s_lSize) 
      refPtrList.insert(refPtrList.begin(), d_ref[i]);
    break;
  case (REDIS):
    ::std::cerr << "ERROR :: getCollectiveReference called for REDIS\n";
  }
  
  return refPtrList;
}

refList* ReferenceMgr::getAllReferences() const
{
  if (localRank >= s_lSize) return(new refList());
  refList *rl = new refList(d_ref);
  rl->resize(s_refSize);
  return (refList*)(rl);
}

void ReferenceMgr::insertReference(Reference *ref)
{
  d_ref.insert(d_ref.begin(),ref);
  s_refSize++;
}

int ReferenceMgr::getRemoteSize()
{
  return s_refSize;
}

int ReferenceMgr::getSize()
{
  return s_lSize;
}

int ReferenceMgr::getRank()
{
  return localRank;
}

void ReferenceMgr::createSubset(int localsize, int remotesize)
{
  /*LOCALSIZE*/
  if(localsize > 0) { 
    if(localsize < this->localSize) s_lSize=localsize;
  }
  else { 
    /*reset subsetting*/
    s_lSize=localSize;
  } 

  /*REMOTESIZE*/
  if(remotesize > 0) { 
    if(remotesize < d_ref.size()) s_refSize=remotesize;
  }
  else { 
    /*reset subsetting*/
    s_refSize=d_ref.size();
  } 


}

