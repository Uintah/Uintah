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

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
//#include <stdio.h>

#include <iostream>
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>

using namespace std;
using namespace SCIRun;

SocketMessage::SocketMessage(int sockfd, void *msg)
{
  this->ep=NULL;
  this->sp=NULL;
  this->msg=msg;
  this->sockfd=sockfd;
  msg_size=0;
  //if(this->msg==NULL) createMessage();
}

void
SocketMessage::setSocketEp(SocketEpChannel* ep)
{
  this->ep=ep;
}

void
SocketMessage::setSocketSp(SocketSpChannel* sp)
{
  this->sp=sp;
}

SocketMessage::~SocketMessage() {
  if(msg!=NULL) free(msg);
}

void 
SocketMessage::createMessage()  { 
  msg=realloc(msg, INIT_SIZE);
  capacity=INIT_SIZE;
  msg_size=sizeof(long)+sizeof(int);
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
  throw CommError("SocketMessage::marshalSpChannel not inmplemented!",-1);
}

void 
SocketMessage::sendMessage(int handler){
  memcpy((char*)msg, &msg_size, sizeof(long));
  memcpy((char*)msg+sizeof(long), &handler, sizeof(int));
  
  if(sockfd==-1) sockfd=sp->sockfd;
  if(sockfd==-1) throw CommError("SocketMessage::sendMessage", -1);
  /*
  //print the msg
  printf("\nMsg=");
  for(int i=0; i<msg_size; i++){
    printf("%4u",((unsigned char*)msg)[i]);
  }
  printf("\n\n");
  */

  if(sendall(sockfd, msg, &msg_size) == -1){ 
      throw CommError("sendall", errno);
  }
}

void 
SocketMessage::waitReply(){
  if(sockfd==-1) sockfd=sp->sockfd;
  if(sockfd==-1) throw CommError("waitReply", -1);
  int headerSize=sizeof(long)+sizeof(int);
  void *buf=malloc(headerSize);
  int numbytes;
  if ((numbytes=recv(sockfd, buf, headerSize, MSG_WAITALL)) == -1) {
    throw CommError("recv", errno);
  }
  int id;
  memcpy(&msg_size, buf, sizeof(long));
  memcpy(&id, (char*)buf+sizeof(long), sizeof(int));

  msg=realloc(msg, msg_size-headerSize);
  if ((numbytes=recv(sockfd, msg, msg_size-headerSize, MSG_WAITALL)) == -1) {
    throw CommError("recv", errno);
  }

  if(numbytes!=msg_size-headerSize) throw CommError("waitReply: numbytes!=msg_size-headerSize", -1);
  
  /*
  //print the msg
  printf("\nMsg=");
  for(int i=0; i<headerSize; i++){
    printf("%4u",((unsigned char*)buf)[i]);
  }
  
  for(int i=0; i<msg_size-headerSize; i++){
    printf("%4u",((unsigned char*)msg)[i]);
  }
  printf("\n\n");
  */
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
  throw CommError("SocketMessage::unmarshalSpChannel not inmplemented!",-1);
}

void* 
SocketMessage::getLocalObj(){
  if(ep==NULL) throw CommError("SocketMessagegetLocalObj", -1);
  return ep->object; 
}

void SocketMessage::destroyMessage() {
  if(msg!=NULL) free(msg);
  msg=NULL;
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


int SocketMessage::sendall(int sockfd, void *buf, int *len)
{
  int total = 0;        // how many bytes we've sent
  int bytesleft = *len; // how many we have left to send
  int n;
  
  while(total < *len) {
    n = send(sockfd, (char*)buf+total, bytesleft, 0);
    if (n == -1) { break; }
    total += n;
    bytesleft -= n;
  }
  
  *len = total; // return number actually sent here
  
  return n==-1?-1:0; // return -1 on failure, 0 on success
} 

