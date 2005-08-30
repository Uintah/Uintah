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
  localSize=PIDL::size; 
  localRank=PIDL::rank;
  s_lSize=localSize;
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
  //  intracomm = PIDL::getIntraComm();
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
    if(remotesize < (int)d_ref.size()) s_refSize=remotesize;
  }
  else { 
    /*reset subsetting*/
    s_refSize=d_ref.size();
  } 
}

void ReferenceMgr::setRankAndSize(int rank, int size)
{
  localRank = rank;
  localSize = size;
  s_lSize = localSize;  
}

void ReferenceMgr::resetRankAndSize()
{
  localSize=PIDL::size; 
  localRank=PIDL::rank;
}
