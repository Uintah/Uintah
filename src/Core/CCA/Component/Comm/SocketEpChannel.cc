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
 *  SocketEpChannel.cc: Socket implemenation of Ep Channel
 *
 *  Written by:
 *   Kosta Damevski and Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */



#include <iostream>
#include <sstream>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>


#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/Comm/SocketThread.h>
#include <Core/CCA/Component/Comm/DT/DTPoint.h>
#include <Core/CCA/Component/Comm/DT/DTMessage.h>
#include <Core/CCA/Component/Comm/DT/DataTransmitter.h>
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/ServerContext.h>

#include <Core/Thread/Thread.h>

using namespace std;
using namespace SCIRun;

SocketEpChannel::SocketEpChannel(){ 
  ep=new DTPoint;
  handler_table=NULL;
}

SocketEpChannel::~SocketEpChannel(){ 
  if(handler_table!=NULL) delete []handler_table;
  delete ep;
}

void SocketEpChannel::openConnection() {
  //...do nothing
}

void SocketEpChannel::closeConnection() {
  //...do nothing 
}

string SocketEpChannel::getUrl() {
  return PIDL::getDT()->getUrl(); 
}

void SocketEpChannel::activateConnection(void* obj){
  ep->object=obj;
  ServerContext* sc=(ServerContext*)(obj);
  sc->d_objid=(int)ep;
  Thread *service_thread = new Thread(new SocketThread(this, NULL,  -1), "SocketServiceThread", 0, Thread::Activated);
  service_thread->detach();
}

Message* SocketEpChannel::getMessage() {
  //this should never be called!
  throw CommError("SocketEpChannel::getMessage", -1);
}

void SocketEpChannel::allocateHandlerTable(int size){
  handler_table = new HPF[size];
  table_size = size;
}

void 
SocketEpChannel::registerHandler(int num, void* handle){
  handler_table[num-1] = (HPF) handle;
}

void 
SocketEpChannel::bind(SpChannel* spchan){
  SocketSpChannel *chan=dynamic_cast<SocketSpChannel*>(spchan);
  chan->ep=ep;
  chan->ep_addr=PIDL::getDT()->getAddress();
}

void 
SocketEpChannel::runService(){
  //cerr<<"ServiceThread starts\n";
  bool alive=true;
  while(alive){
    DTMessage *msg=ep->getMessage();
    if(msg==NULL){
      cerr<<"runService get NULL msg\n";
      break;
    }
    int id=*((int *)msg->buf);
    //cerr<<"Servicing id="<<id<<endl;
    //filter internal messages
    if(id<=-100){
      ServerContext* sc=(ServerContext*)(ep->object);
      switch(id){
      case -101:
	sc->d_objptr->_addReference();
	break;
      case -102:
	sc->d_objptr->_deleteReference();
	if(sc->d_objptr->getRefCount()==0) alive=false;
	break;
      }
    }
    else{
      SocketMessage* new_msg=new SocketMessage(this, msg);
      //The SocketHandlerThread is responsible to free the buf.   

      Thread* t = new Thread(new SocketThread(this, new_msg, id), "SocketHandlerThread", 0, Thread::Activated);
      t->detach();
    }
  }  
  //cerr<<"ServiceThread stops\n";
}


