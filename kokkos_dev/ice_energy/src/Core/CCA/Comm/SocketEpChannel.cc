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


#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/Comm/SocketEpChannel.h>
#include <Core/CCA/Comm/SocketSpChannel.h>
#include <Core/CCA/Comm/CommError.h>
#include <Core/CCA/Comm/SocketMessage.h>
#include <Core/CCA/Comm/SocketThread.h>
#include <Core/CCA/Comm/DT/DTPoint.h>
#include <Core/CCA/Comm/DT/DTMessage.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <Core/CCA/PIDL/Object.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/ServerContext.h>

#include <Core/Thread/Thread.h>
#include <Core/CCA/Comm/PRMI.h>

using namespace std;
using namespace SCIRun;

static void service(DTMessage *dtmsg){
  int id=*((int *)dtmsg->buf);
  
  if(id==SocketEpChannel::MPI_LOCKSERVICE){
    PRMI::lock_service(dtmsg);
    return;
  }
  if(id==SocketEpChannel::MPI_ORDERSERVICE){
    PRMI::order_service(dtmsg);
    return;
  }
  
  void* v=dtmsg->recver->object;
  ServerContext* sc=static_cast< ServerContext*>(v); 
  SocketEpChannel *chan = dynamic_cast<SocketEpChannel*>(sc->chan);
  if (chan) {
    if(id==SocketEpChannel::ADD_REFERENCE){
      sc->d_objptr->addReference();
      delete dtmsg;
    }
    else if(id==SocketEpChannel::DEL_REFERENCE){
      sc->d_objptr->deleteReference();
      delete dtmsg;
    }
    else{
      if (id >= chan->getTableSize())
	throw CommError("Handler function does not exist",1101);
      SocketMessage* msg=new SocketMessage(dtmsg);
      Thread *t = new Thread(new SocketThread(chan, msg, id), "HANDLER_THREAD");
      t->detach(); 
    }
  }
}


SocketEpChannel::SocketEpChannel(){ 
  ep=new DTPoint(PIDL::getDT());
  ep->service=::service;
  handler_table=NULL;
}

SocketEpChannel::~SocketEpChannel(){ 
  if(handler_table!=NULL) delete []handler_table;
  delete ep;
}


void SocketEpChannel::openConnection() {
  //do nothing
}

void SocketEpChannel::closeConnection() {
  //do nothing
}

string SocketEpChannel::getUrl() {
  return PIDL::getDT()->getUrl(); 
}

void SocketEpChannel::activateConnection(void* obj){
  ep->object=obj;
  ServerContext* sc=(ServerContext*)(obj);
  sc->d_objid=(int)ep;
}

Message* SocketEpChannel::getMessage() {
  throw CommError("SocketEpChannel::getMessage should never be called", -1);
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


int 
SocketEpChannel::getTableSize(){
  return table_size;
}

DTPoint*
SocketEpChannel::getEP(){
  return ep;
}
