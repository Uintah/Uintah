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

#include "ReferenceMgr.h"
#include <iostream>
using namespace SCIRun;

ReferenceMgr::ReferenceMgr()
{
  localSize=1; 
  localRank=0;
  isSubset=false;
}

ReferenceMgr::ReferenceMgr(int rank, int size)
  :localSize(size), localRank(rank), isSubset(false)
{
}

ReferenceMgr& ReferenceMgr::operator=(const ReferenceMgr& copy)
{
  d_ref = copy.d_ref;
  for(unsigned int i=0; i < d_ref.size(); i++) {
    d_ref[i].chan = (copy.d_ref[i].chan)->SPFactory(true);
  }
  localSize = copy.localSize;
  localRank = copy.localRank;
  isSubset = copy.isSubset;
  if(isSubset) {
    save_lSize = copy.save_lSize;
  } 
  intracomm = copy.intracomm;
  return *this;
}

ReferenceMgr::ReferenceMgr(const ReferenceMgr& copy)
  :localSize(copy.localSize),localRank(copy.localRank),
   isSubset(copy.isSubset), d_ref(copy.d_ref),
   intracomm(copy.intracomm) 
{
  for(unsigned int i=0; i < d_ref.size(); i++) {
    d_ref[i].chan = (copy.d_ref[i].chan)->SPFactory(true);
  }
  if(isSubset) {
    save_lSize = copy.save_lSize;
  }
}

ReferenceMgr::~ReferenceMgr()
{
  refList::iterator iter = d_ref.begin();
  for(unsigned int i=0; i < d_ref.size(); i++, iter++) {
    (*iter).chan->closeConnection();
  }
}

Reference* ReferenceMgr::getIndependentReference() const
{
  return (Reference*)(&(d_ref[localRank % d_ref.size()]));
}

::std::vector<Reference*> ReferenceMgr::getCollectiveReference(callType tip)
{
  ::std::vector<Reference*> refPtrList;
  refList::iterator iter;

  if (localRank >= localSize) return refPtrList;

  switch(tip) {
  case (CALLONLY):
  case (NOCALLRET):
    refPtrList.insert(refPtrList.begin(), &(d_ref[localRank % d_ref.size()]));
    break;
  case (CALLNORET):
    iter = d_ref.begin() + (localRank + localSize);
    for(int i=(localRank + localSize); i < (int)d_ref.size(); i+=localSize, iter+=localSize) 
      refPtrList.insert(refPtrList.begin(), &(d_ref[i]));
    break;
  case (REDIS):
    ::std::cerr << "ERROR :: getCollectiveReference called for REDIS\n";
  }
  
  return refPtrList;
}

refList* ReferenceMgr::getAllReferences() const
{
  if (localRank >= localSize) return(new refList());
  return (refList*)(&d_ref);
}

void ReferenceMgr::insertReference(const Reference& ref)
{
  d_ref.insert(d_ref.begin(),ref);
}

int ReferenceMgr::getRemoteSize()
{
  return (d_ref.size());
}

int ReferenceMgr::getSize()
{
  return localSize;
}

int ReferenceMgr::getRank()
{
  return localRank;
}

void ReferenceMgr::createSubset(int ssize)
{
  if(ssize) { 
    if(ssize >= localSize) return;
    if(!isSubset){
      isSubset=true;
      save_lSize=localSize;
    }
    localSize=ssize; 
  }
  else { 
    if(isSubset) {
      /*reset subsetting*/
      localSize=save_lSize;
      isSubset=false;
    }
  } 

}

