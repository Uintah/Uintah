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
 *  SocketMessage.cc: Socket implemenation of Message
 *
 *  Written by:
 *   Kosta Damevski and Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
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
#include <unistd.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>


#include <iostream>
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>
#include <Core/CCA/Component/Comm/DT/DTMessage.h>
#include <Core/CCA/Component/Comm/DT/DataTransmitter.h>
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/Warehouse.h>
#include <Core/CCA/Component/PIDL/URL.h>

using namespace std;
using namespace SCIRun;


SocketMessage::SocketMessage(DTMessage *dtmsg)
{
  this->msg=dtmsg->buf;
  if(dtmsg->autofree) dtmsg->buf=NULL;
  else{
    cerr<<"TODO: shoudl copy memory here"<<endl;
  }
  this->dtmsg=dtmsg;
  msg_size=sizeof(int); //skip handler_id
  spchan=NULL;
}

SocketMessage::SocketMessage(SocketSpChannel *spchan)
{
  this->dtmsg=NULL;
  this->msg=NULL;
  msg_size=sizeof(int); //skip handler_id
  this->spchan=spchan;
}

SocketMessage::~SocketMessage() {
  if(dtmsg!=NULL)  delete dtmsg;
  if(msg!=NULL) free(msg);
}

void 
SocketMessage::createMessage()  { 
  msg=realloc(msg, INIT_SIZE);
  capacity=INIT_SIZE;
  msg_size=sizeof(int); //reserve for handler_id
}

void 
SocketMessage::marshalInt(const int *buf, int size){
  marshalBuf(buf, size*sizeof(int));
  //  cerr<<"Marshal Int("<<size<<")="<<*buf<<endl;
}

void 
SocketMessage::marshalByte(const char *buf, int size){
  marshalBuf(buf, size*sizeof(char));
}

void 
SocketMessage::marshalChar(const char *buf, int size){
  marshalBuf(buf, size*sizeof(char));
  //cerr<<"Marshal Char("<<size<<")=";
  //for(int i=0; i<size; i++) cerr<<*(buf+i);
  //cerr<<endl;
}

void 
SocketMessage::marshalFloat(const float *buf, int size){
  marshalBuf(buf, size*sizeof(float));
}
void 
SocketMessage::marshalDouble(const double *buf, int size){
  marshalBuf(buf, size*sizeof(double));
}

void 
SocketMessage::marshalLong(const long *buf, int size){
  marshalBuf(buf, size*sizeof(long));
}

void 
SocketMessage::marshalSpChannel(SpChannel* channel){
  SocketSpChannel * chan = dynamic_cast<SocketSpChannel*>(channel);
  marshalBuf(&(chan->ep_addr), sizeof(DTAddress));
  marshalBuf(&(chan->ep), sizeof(void *));
}

void 
SocketMessage::sendMessage(int handler){
  //cerr<<"SocketMessage::sendMessage handler id="<<handler<<endl;
  memcpy(msg, &handler, sizeof(int));
  DTMessage *wmsg=new DTMessage;
  wmsg->buf=(char*)msg;
  wmsg->length=msg_size;
  wmsg->autofree=true;
  msg=NULL;   //DT is responsible to delete it.
  if(dtmsg!=NULL){
    wmsg->recver=dtmsg->sender;
    wmsg->to_addr=dtmsg->fr_addr;
    dtmsg->recver->putMessage(wmsg);
  }
  else{
    wmsg->recver=spchan->ep;
    wmsg->to_addr= spchan->ep_addr;
    spchan->sp->putMessage(wmsg);
  }
}

void 
SocketMessage::waitReply(){
  DTMessage *wmsg=spchan->sp->getMessage();
  wmsg->autofree=false;
  if(msg!=NULL) free(msg);
  msg=wmsg->buf;
  delete wmsg;
  msg_size=sizeof(int);//skip handler_id
}

void 
SocketMessage::unmarshalReply(){
  //do nothing
}

void 
SocketMessage::unmarshalInt(int *buf, int size){
  unmarshalBuf(buf, size*sizeof(int));
}

void 
SocketMessage::unmarshalByte(char *buf, int size){
  unmarshalBuf(buf, size*sizeof(char));
}
void 
SocketMessage::unmarshalChar(char *buf, int size){
  unmarshalBuf(buf, size*sizeof(char));
}

void 
SocketMessage::unmarshalFloat(float *buf, int size){
  unmarshalBuf(buf, size*sizeof(float));
}

void 
SocketMessage::unmarshalDouble(double *buf, int size){
  unmarshalBuf(buf, size*sizeof(double));
}

void 
SocketMessage::unmarshalLong(long *buf, int size){
  unmarshalBuf(buf, size*sizeof(long));
}

void 
SocketMessage::unmarshalSpChannel(SpChannel* channel){
  SocketSpChannel * chan = dynamic_cast<SocketSpChannel*>(channel);

  unmarshalBuf(&(chan->ep_addr), sizeof(DTAddress));
  unmarshalBuf(&(chan->ep), sizeof(void *));
}

void* 
SocketMessage::getLocalObj(){
  if(dtmsg!=NULL){
    return dtmsg->recver->object;
  }
  else{
    if(!PIDL::getDT()->isLocal(spchan->ep_addr)) return NULL;
    return spchan->ep->object;
  }
}

void SocketMessage::destroyMessage() {
  if(dtmsg!=NULL)  delete dtmsg;
  if(msg!=NULL) free(msg);
  dtmsg=NULL;
  msg=NULL;
  delete this;
}


//private methods
void 
SocketMessage::marshalBuf(const void *buf, int fullsize){
  msg_size+=fullsize;
  if(msg_size>capacity){
    capacity=msg_size+INIT_SIZE;
    msg=realloc(msg, capacity);
  }
  memcpy((char*)msg+msg_size-fullsize, buf, fullsize); 
}

void 
SocketMessage::unmarshalBuf(void *buf, int fullsize){
  memcpy(buf, (char*)msg+msg_size, fullsize); 
  msg_size+=fullsize;
}

