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
#include <Core/CCA/Comm/CommError.h>
#include <Core/CCA/Comm/SocketMessage.h>
#include <Core/CCA/Comm/SocketEpChannel.h>
#include <Core/CCA/Comm/SocketSpChannel.h>
#include <Core/CCA/Comm/DT/DTMessage.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/Warehouse.h>
#include <Core/CCA/PIDL/URL.h>

using namespace std;
using namespace SCIRun;

//TODO: handle network byte order and local byte order compitablility.


SocketMessage::SocketMessage(DTMessage *dtmsg)
{
  msg=dtmsg->buf;
  if(dtmsg->autofree) dtmsg->buf=NULL;
  else{
    msg=malloc(dtmsg->length);
    memcpy(msg, dtmsg->buf, dtmsg->length);
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
}

void 
SocketMessage::marshalByte(const char *buf, int size){
  marshalBuf(buf, size*sizeof(char));
}

void 
SocketMessage::marshalChar(const char *buf, int size){
  marshalBuf(buf, size*sizeof(char));
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
SocketMessage::marshalOpaque(void **buf, int size){
  DTAddress addr=PIDL::getDT()->getAddress();
  marshalBuf(&addr, sizeof(DTAddress));
  marshalBuf(buf, size*sizeof(void*));
}


void 
SocketMessage::sendMessage(int handler){
  memcpy(msg, &handler, sizeof(int));
  DTMessage *wmsg=new DTMessage;
  wmsg->buf=(char*)msg;
  wmsg->length=msg_size;
  wmsg->autofree=true;
  msg=NULL;   //DT is responsible to delete it.
  if(dtmsg!=NULL){
    wmsg->tag=dtmsg->tag;  //reply
    wmsg->recver=dtmsg->sender;
    wmsg->to_addr=dtmsg->fr_addr;
    dtmsg->recver->putReplyMessage(wmsg);
  }
  else{
    wmsg->recver=spchan->ep;
    wmsg->to_addr= spchan->ep_addr;
    tag=spchan->sp->putInitialMessage(wmsg);//initial message, save the tag
  }
}

void 
SocketMessage::waitReply(){
  DTMessage *wmsg=spchan->sp->getMessage(tag);
  msg_length=wmsg->length;
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


void 
SocketMessage::unmarshalOpaque(void **buf, int size){
  DTAddress dtaddr;
  unmarshalBuf(&dtaddr, sizeof(DTAddress));
  unmarshalBuf(buf, size*sizeof(void*));
  if(! (dtaddr==PIDL::getDT()->getAddress()) ){
    for(int i=0; i<size; i++){
      buf[i]=NULL;
    }
  }
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

int SocketMessage::getRecvBufferCopy(void* buf)
{
  memcpy(buf, msg, msg_length);
  return msg_length;
}

int SocketMessage::getSendBufferCopy(void* buf)
{
  memcpy(buf, msg, msg_size);
  return msg_size;
}

void SocketMessage::setRecvBuffer(void* buf, int len)
{
  if(msg!=NULL) free(msg);
  msg=buf;
  msg_length=len;
  msg_size=sizeof(int); //skip the handler id.
}

void SocketMessage::setSendBuffer(void* buf, int len)
{
  if(msg!=NULL) free(msg);
  msg=buf;
  msg_size=len;
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

