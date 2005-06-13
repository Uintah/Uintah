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
 *  PRMI.cc: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2004
 *
 *  Copyright (C) 1999 SCI Group
 */



#include <stdlib.h>
#include <string>
#include <sstream>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <unistd.h>
#include <iostream>
#include <string.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <Core/Thread/Time.h>
#include <sys/time.h>


#include <iostream>
#include <Core/CCA/Comm/CommError.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <Core/CCA/Comm/DT/DTThread.h>
#include <Core/CCA/Comm/DT/DTPoint.h>
#include <Core/CCA/Comm/DT/DTMessage.h>
#include <Core/CCA/Comm/SocketMessage.h>
#include <Core/CCA/Comm/SocketEpChannel.h>
#include <Core/CCA/Comm/SocketSpChannel.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/Comm/PRMI.h>
#include <deque>
#include <sci_defs/mpi_defs.h> // For MPIPP_H
#include <mpi.h>
#include <Core/CCA/Comm/PRMI.h>


using namespace SCIRun;
using namespace std;

int 
PRMI::mpi_size;

int 
PRMI::mpi_rank;

SocketEpChannel
PRMI::orderSvcEp;

SocketEpChannel
PRMI::lockSvcEp;

DTPoint* 
PRMI::orderSvc_ep;

DTAddress 
PRMI::orderSvc_addr;

DTPoint** 
PRMI::lockSvc_ep_list;

DTAddress*     
PRMI::lockSvc_addr_list;

std::deque<PRMI::lockid>
PRMI::lockQ;

Mutex *
PRMI::lock_mutex=new Mutex("lock mutex");

Mutex *
PRMI::order_mutex=new Mutex("order mutex");

std::map<Thread* ,PRMI::states*> 
PRMI::states_map;

std::map<PRMI::lockid ,Semaphore*>
PRMI::lock_sema_map;

std::map<PRMI::lockid ,int> 
PRMI:: lock_req_map;

PRMI::states 
PRMI::fwkstate;

// #ifdef HAVE_MPI
// MPI_Comm
// PRMI::MPI_COMM_WORLD_Dup;
// #endif

void 
PRMI::init(int rank, int size){
  mpi_rank=rank;
  mpi_size=size;
}

void
PRMI::internal_lock(){
  //called by any threads in the same address space. 
  states* stat=getStat();

  //make lock id
  lockid lid;
  lid.invID=stat->invID;
  //only one thread accesses seq at anytime, so it is
  //safe to do so:
  lid.seq=++(stat->lockSeq);
  
  //wrap a message (MPI_LOCK_REQUEST, lockid) and send to order service

  SocketSpChannel spc(orderSvc_ep, orderSvc_addr);
  SocketMessage msg(&spc); 
  //  cerr<<"#### PRMI:lock sending message to orderservice lockid= "<<lid.str()<<" ####\n";
  msg.createMessage();
  msg.marshalInt(&lid.invID.iid);
  msg.marshalInt(&lid.invID.pid);
  msg.marshalInt(&lid.seq);
  msg.sendMessage(SocketEpChannel::MPI_ORDERSERVICE);
  
  //check if lock id is at the head of the lock queue.
  lock_mutex->lock();
  if(lockQ.size()==0 || lid != lockQ[0]){
    if(lock_sema_map.find(lid)==lock_sema_map.end()){
      //create/register a semaphore and wait
      lock_sema_map[lid]=new Semaphore("mpi lock semaphre",0);
    }
    lock_mutex->unlock();
    //    cerr<<"PRMI::lock calls lock_sema_map[lid]->down()\n";
    lock_sema_map[lid]->down();
    //waiting for mpiunlock() or updatlock() to update the semaphore
  }else{
    lock_mutex->unlock();
  }
  //now PRMI_ID should be the head of the lock queue, so grant the lock
}

void
PRMI::internal_unlock(){
  //called by any threads in the same address space. 

  //remove the head of the lock queue
  lock_mutex->lock();
  lockid lid=lockQ[0];
  lockQ.pop_front();
  //destroy the associated semaphore, if there is any.
  if(lock_sema_map.find(lid)!=lock_sema_map.end()){
    lock_sema_map.erase(lid);
  }
  lock_mutex->unlock();

  lid= lockQ[0];

  //if lock queue is not empty
  if(lockQ.size()>0){
    //if no semaphore is associated with the queue head, create one
    lock_mutex->lock();
    if(lock_sema_map.find(lid)==lock_sema_map.end()){
      lock_sema_map[lid]=new Semaphore("mpi lock semaphre",0);
    }
    lock_mutex->unlock();
    //raise the semaphore
    lock_sema_map[lid]->up();
  }
}

void 
PRMI::lock(){
#ifdef HAVE_MPI
#ifndef MPI_IS_THREADSAFE
  internal_lock();
#endif
#endif
}

void 
PRMI::unlock(){
#ifdef HAVE_MPI
#ifndef MPI_IS_THREADSAFE
  internal_unlock();
#endif
#endif
}


// #ifdef HAVE_MPI
// int 
// PRMI::getComm(MPI_Comm *newComm){
//   int retval;
//   internal_lock();
//   retval=MPI_Comm_dup(MPI_COMM_WORLD_Dup, newComm);
//   internal_unlock();
//   return retval;
// }
// #endif

void
PRMI::addStat(states *stat){
  lock_mutex->lock();
  states_map[Thread::self()]=stat;
  lock_mutex->unlock();
}

void
PRMI::delStat(){
  lock_mutex->lock();
  states_map.erase(Thread::self());
  lock_mutex->unlock();
}

PRMI::states *
PRMI::getStat(){
  states *stat;
  lock_mutex->lock();
  stat=states_map[Thread::self()];
  lock_mutex->unlock();
  return stat;
}

void
PRMI::setInvID(ProxyID id){
  if(PIDL::isFramework()){
    //this part should be called exactly once at the beginning of 
    //the framework
    fwkstate.invID=id;
    fwkstate.nextID=id;
  }
  else{
    lock_mutex->lock();
    states_map[Thread::self()]->invID=id;
    states_map[Thread::self()]->lockSeq=0;
    
    lock_mutex->unlock();
  }
}

void
PRMI::setProxyID(ProxyID nextID){
  //framework should ignore this function call
  if(!PIDL::isFramework()){
    lock_mutex->lock();
    states_map[Thread::self()]->nextID=nextID;
    lock_mutex->unlock();
  }
}

ProxyID
PRMI::getProxyID(){
  ProxyID nextID;
  //framework should ignore this function call
  if(PIDL::isFramework()){
    lock_mutex->lock();
    nextID=fwkstate.nextID=fwkstate.nextID.next1st();
    lock_mutex->unlock();
  }
  else{
    lock_mutex->lock();
    nextID=states_map[Thread::self()]->nextID;
    states_map[Thread::self()]->nextID=nextID.next();
    lock_mutex->unlock();
  }
  return nextID;
}

ProxyID
PRMI::peekProxyID(){
  ProxyID nextID;
  //framework should ignore this function call
  if(PIDL::isFramework()){
    lock_mutex->lock();
    nextID=fwkstate.nextID;
    lock_mutex->unlock();
  }
  else{
    lock_mutex->lock();
    nextID=states_map[Thread::self()]->nextID;
    lock_mutex->unlock();
  }
  return nextID;
}


void 
PRMI::order_service(DTMessage *dtmsg){
  //  cerr<<"#### PRMI:order_service called ####\n";
  SocketMessage msg(dtmsg);
  lockid lid;
  msg.unmarshalInt(&lid.invID.iid);
  msg.unmarshalInt(&lid.invID.pid);
  msg.unmarshalInt(&lid.seq);

  order_mutex->lock();
  if(lock_req_map.find(lid)==lock_req_map.end()){
    //new order request
    lock_req_map[lid]=mpi_size-1;

    //broadcast the order to lock services
    for(int i=0; i<mpi_size; i++){
      //      cerr<<"#### PRMI:order_service sending message to lockservice at rank"<<i<<" lockid= "<<lid.str()<<" ####\n";
      SocketSpChannel spc(lockSvc_ep_list[i], lockSvc_addr_list[i]);
      SocketMessage msg(&spc); 
      msg.createMessage();
      msg.marshalInt(&lid.invID.iid);
      msg.marshalInt(&lid.invID.pid);
      msg.marshalInt(&lid.seq);
      msg.sendMessage(SocketEpChannel::MPI_LOCKSERVICE);
    }
  }else{
    lock_req_map[lid]--;
  }
  if(lock_req_map[lid]==0){
    //all requests received, remove it 
    lock_req_map.erase(lid);
  }
  order_mutex->unlock();
}


void 
PRMI::lock_service(DTMessage *dtmsg){
  //  cerr<<"#### PRMI:lock_service called ####\n";
  //first decode lockid
  SocketMessage msg(dtmsg);
  lockid lid;
  msg.unmarshalInt(&lid.invID.iid);
  msg.unmarshalInt(&lid.invID.pid);
  msg.unmarshalInt(&lid.seq);

  lock_mutex->lock(); 
  //append lockid to lock queue
  lockQ.push_back(lid);

  //if lockid is the only element
  if(lockQ.size()==1){
    //if no semaphore is associated with the queue head, create one
    if(lock_sema_map.find(lid)==lock_sema_map.end()){
      lock_sema_map[lid]=new Semaphore("mpi lock semaphre",0);
    }
    //raise the semaphore
    lock_sema_map[lid]->up();
    //    cerr<<"PRMI::lock_service calls lock_sema_map[lid]->up()\n";
  }
  lock_mutex->unlock();
}



