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
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/URL.h>

using namespace std;
using namespace SCIRun;


SocketMessage::SocketMessage(int sockfd, void *msg)
{
  this->msg=msg;
  this->sockfd=sockfd;
  this->object=NULL;
  msg_size=0;
}

SocketMessage::~SocketMessage() {
  if(msg!=NULL) free(msg);
}

void SocketMessage::setLocalObject(void *obj){
  object=obj;
}

void 
SocketMessage::createMessage()  { 
  msg=realloc(msg, INIT_SIZE);
  capacity=INIT_SIZE;
  msg_size=sizeof(long)+sizeof(int);
  object=NULL;
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
  SocketStartPoint *sp=chan->sp;

  marshalBuf(&(sp->ip), sizeof(long));
  marshalBuf(&(sp->port), sizeof(short));
  marshalBuf(&(sp->pid), sizeof(int));
  marshalBuf(&(sp->object), sizeof(int));
}

void 
SocketMessage::sendMessage(int handler){
  memcpy((char*)msg, &msg_size, sizeof(long));
  memcpy((char*)msg+sizeof(long), &handler, sizeof(int));
  
  if(sockfd==-1) throw CommError("SocketMessage::sendMessage", -1);

  //cerr<<"SEND MSG (sid="<<sockfd<<"):"<<msg_size<<" bytes, handlerID="<<handler<<endl;
  if(sendall(sockfd, msg, msg_size) == -1){ 
      throw CommError("send", errno);
  }
}

void 
SocketMessage::waitReply(){
  if(sockfd==-1) throw CommError("waitReply", -1);
  int headerSize=sizeof(long)+sizeof(int);
  void *buf=malloc(headerSize);
  if (recvall(sockfd, buf, headerSize) == -1) {
    throw CommError("recv", errno);
  }
  int id;
  memcpy(&msg_size, buf, sizeof(long));
  memcpy(&id, (char*)buf+sizeof(long), sizeof(int));

  //cerr<<"WAITREPLY RECV MSG:"<<msg_size<<" bytes, handlerID="<<id<<endl;

  msg=realloc(msg, msg_size-headerSize);
  if (recvall(sockfd, msg, msg_size-headerSize) == -1) {
    throw CommError("recv", errno);
  }
  
  free(buf);
  msg_size=0;
}

void 
SocketMessage::unmarshalReply(){
  //don't have to do anything
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
  SocketStartPoint *sp=chan->sp;

  unmarshalBuf(&(sp->ip), sizeof(long));
  unmarshalBuf(&(sp->port), sizeof(short));
  unmarshalBuf(&(sp->pid), sizeof(int));
  unmarshalBuf(&(sp->object), sizeof(int));

  sp->refcnt=1;
  //TODO: check local object 
  if(sp->ip==SocketEpChannel::getIP() && sp->pid==PIDL::getPID() && sp->object!=NULL){
    sp->sockfd=-1; 
  }
  else{
    sp->object=NULL;
    //cerr<<"sockfd opened\n";
    struct sockaddr_in their_addr; // connector's address information 
    
    if( (sp->sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
      throw CommError("socket", errno);
    }

    their_addr.sin_family = AF_INET;                   // host byte order 
    their_addr.sin_port = htons(sp->port);  // short, network byte order 
    their_addr.sin_addr = *(struct in_addr*)(&(sp->ip));
    memset(&(their_addr.sin_zero), '\0', 8);  // zero the rest of the struct 
    
    if(connect(sp->sockfd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1) {
      perror("connect");
      throw CommError("connect", errno);
    }

    Message *msg=chan->getMessage();
    msg->createMessage();
    msg->sendMessage(-101);  //call deleteReference
    msg->destroyMessage(); 
  }
}

void* 
SocketMessage::getLocalObj(){
  return object;
}

void SocketMessage::destroyMessage() {
  if(msg!=NULL) free(msg);
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


int 
SocketMessage::sendall(int sockfd, void *buf, int len)
{
  int total = 0;        // how many bytes we've sent
  int n;
  
  while(total < len) {
    n = send(sockfd, (char*)buf+total, len, 0);
    if (n == -1) { break; }
    total += n;
    len -= n;
  }
  return n==-1?-1:0; // return -1 on failure, 0 on success
} 


int 
SocketMessage::recvall(int sockfd, void *buf, int len)
{
  int total = 0;        // how many bytes we've recved
  int n;
  
  while(total < len) {
    n = recv(sockfd, (char*)buf+total, len, 0);
    if (n == -1) { break; }
    total += n;
    len -= n;
  }
  if(n==-1) perror("recv");
  return n==-1?-1:0; // return -1 on failure, 0 on success
} 

