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
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <iostream>
//#include <Core/CCA/tools/sidl/uuid_wrapper.h>
#include <assert.h>

using namespace SCIRun;

ProxyBase::ProxyBase() 
  : proxy_uuid()
  //: proxy_uuid("NONENONENONENONENONENONENONENONENONE") 
{
    xr = new XceptionRelay(this);
}

//remove it later
ProxyBase::ProxyBase(const Reference& ref)
  : proxy_uuid()
  //: proxy_uuid("NONENONENONENONENONENONENONENONENONE")
{ 
    xr = new XceptionRelay(this);
  rm.insertReference( ((Reference*)&ref)->clone());
}

ProxyBase::ProxyBase(Reference *ref)
  : proxy_uuid()
  //: proxy_uuid("NONENONENONENONENONENONENONENONENONE")
{ 
    xr = new XceptionRelay(this);
  rm.insertReference(ref);
}

ProxyBase::ProxyBase(const ReferenceMgr& refM)
: rm(refM), proxy_uuid()
  //  proxy_uuid("NONENONENONENONENONENONENONENONENONE")
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
//    if((rm.localSize > 1)&&(rm.intracomm != NULL)) {
//      delete (rm.intracomm);
//      rm.intracomm = NULL; 
//    } 
  
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

ProxyID
ProxyBase::getProxyUUID()
{
  if(proxy_uuid.isNull()){
    proxy_uuid= PRMI::getProxyID();
  }
  
  return proxy_uuid;
}

void ProxyBase::_proxycreateSubset(int localsize, int remotesize)
{
  if(proxy_uuid.isNull()){
    getProxyUUID();
  }
  rm.createSubset(localsize,remotesize);
}

void ProxyBase::_proxysetRankAndSize(int rank, int size)
{
  rm.setRankAndSize(rank,size);
}

void ProxyBase::_proxyresetRankAndSize()
{
  rm.resetRankAndSize();
}

int ProxyBase::_proxygetException(int xid, int lineid)
{

  int size = rm.getSize();
  int rank = rm.getRank();
  int recv_xid;
  int recv_lineid;


  typedef enum {
    none,
    winner,
    loser,
    bye
  } role_t;
  role_t role;

  //Rank for loser (calculated by winner) and vice versa
  int rloser;
  int rwinner;

  //Allocate some space in buffers
  char* recvb = (char*)malloc(20);
  char* sendb = (char*)malloc(20);

  for(double round=1; size > (int)pow(2,round-1) ;round++) {
    // distribute roles
    if (((rank % (int)pow(2,round)) == 0)&&
        ((rank+pow(2,round-1)) < size))
      role = winner;
    else if ((rank % (int)pow(2,round)) == pow(2,round-1))
      role = loser;
    else if (((rank % (int)pow(2,round)) == 0)&&
             ((rank+pow(2,round-1)) >= size))
      role = bye;
    else {
      role = none;
    }

    // winners receive, losers send, and byes do nothing
    switch (role) {
    case (winner): {
      rloser = rank + (int)pow(2,round-1);
      //      (rm.intracomm)->receive(rloser,recvb,20);
      recv_xid = *(int *)(recvb);
      recv_lineid = *(int *)(recvb+sizeof(int));
      //Compare exceptions and take preferred one
      if(recv_xid > 0) {
        if(recv_lineid < lineid) {
	  lineid = recv_lineid;
	  xid = recv_xid;
        } else if(recv_lineid == lineid) {
	  if(recv_xid > xid) xid = recv_xid;
        }
      }
      ::std::cout << "Round " << round << ", rank " << rank << " (xid=" << xid << ")(lineID=" << lineid << ") WINS\n";
      break;
    } case (loser): {
      char* p_sendb = sendb;
      memcpy(p_sendb,&xid,sizeof(int));
      p_sendb += sizeof(int);
      memcpy(p_sendb,&lineid,sizeof(int));
      rwinner = rank - (int)pow(2,round-1);
      //      (rm.intracomm)->send(rwinner,sendb,20);
      ::std::cout << "Round " << round << ", rank " << rank << " (xid=" << xid << ")(lineID=" << lineid << ") LOSES\n";
      break;
    } case (bye):
      //::std::cout << "Round " << round << ", rank " << rank << " (xid=" << xid << ") HAS A BYE\n";
      //do nothing
      break;
    case (none):
      //do nothing
      break;
    default:
      ::std::cout << "Unexpected role. I'm out\n";
      exit(1);
    }

  }

  //Clean up
  delete sendb;
  delete recvb;

  //Broadcast definitive exception
  int champ = 0;
  //  (rm.intracomm)->broadcast(champ,(char *)&xid,(int)sizeof(int));
  std::cout << "Rank " << rank << " will throw xid=" << xid << "\n";
  return xid;
}
