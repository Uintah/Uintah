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
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>
#include <Core/CCA/Component/PIDL/URL.h>

using namespace std;
using namespace SCIRun;


string SocketMessage::sitetag;
string SocketMessage::hostname;

SocketMessage::SocketMessage(int sockfd, void *msg)
{
  this->ep=NULL;
  this->sp=NULL;
  this->msg=msg;
  this->sockfd=sockfd;
  this->object=NULL;
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

  //cerr<<"MarshaSp:"<<chan->ep_url<<endl;

  //marshalBuf(&(chan->sockfd), sizeof(int));
  int urlSize=chan->ep_url.size(); 
  marshalBuf(&urlSize, sizeof(int));
  marshalBuf(chan->ep_url.c_str(), urlSize);

  int tagSize=sitetag.size();
  marshalBuf(&tagSize, sizeof(int));
  marshalBuf(sitetag.c_str(), tagSize);

  marshalBuf(&(chan->object), sizeof(int));
}

void 
SocketMessage::sendMessage(int handler){
  memcpy((char*)msg, &msg_size, sizeof(long));
  memcpy((char*)msg+sizeof(long), &handler, sizeof(int));
  
  if(sockfd==-1) sockfd=sp->sockfd;
  if(sockfd==-1) throw CommError("SocketMessage::sendMessage", -1);

  //cerr<<"SEND MSG (sid="<<sockfd<<"):"<<msg_size<<" bytes, handlerID="<<handler<<endl;
  if(sendall(sockfd, msg, msg_size) == -1){ 
      throw CommError("send", errno);
  }
}

void 
SocketMessage::waitReply(){
  if(sockfd==-1) sockfd=sp->sockfd;
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
  SocketSpChannel * chan = dynamic_cast<SocketSpChannel*>(channel);
  //unmarshalBuf(&(chan->sockfd), sizeof(int));
  int urlSize;
  unmarshalBuf(&urlSize, sizeof(int));
  char *buf=new char[urlSize+1];
  unmarshalBuf(buf, urlSize);
  buf[urlSize]='\0';
  chan->ep_url=buf;
  delete []buf;
  //cerr<<"UnmarshaSp:"<<chan->ep_url<<endl;


  int tagSize;
  unmarshalBuf(&tagSize, sizeof(int));
  buf=new char[tagSize+1];
  unmarshalBuf(buf, tagSize);
  buf[tagSize]='\0';
  string sp_sitetag=buf;
  delete []buf;
  //cerr<<"UnmarshaSp:"<<sp_sitetag<<endl;

  marshalBuf(&object, sizeof(int));  

  //if(!isLocal(sp_sitetag))

  chan->openConnection(URL(chan->ep_url));

  //This is only a temporary solution
  Message *msg=chan->getMessage();
  msg->createMessage();
  msg->sendMessage(-100);  //call deleteReference
  msg->destroyMessage();
  
  ////////////////////////////////
}

void* 
SocketMessage::getLocalObj(){
  if(object!=NULL) return object;
  if(ep==NULL) return NULL; //throw CommError("SocketMessagegetLocalObj", -1);
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



bool 
SocketMessage::isLocal(const string& tag){
  return tag==sitetag;
}


string
SocketMessage::getHostname(){
  return hostname;
}

void
SocketMessage::setSiteTag(){
  char *name=new char[128];
  if(gethostname(name, 127)==-1){
    throw CommError("gethostname", errno);
  }
  hostname=name;
  delete []name;

  pid_t pid=getpid();

  ostringstream o;
  o << hostname << ":" << pid;

  sitetag=o.str();

  //cerr<<"sitetag="<<sitetag<<endl;
}

string
SocketMessage::getSiteTag(){
  return sitetag;
}


void 
SocketMessage::desplayMessage(){
  //print the msg
  printf("\nMsg=");
  for(int i=0; i<msg_size; i++){
    printf("%02x",((unsigned char*)msg)[i]);
  }
  printf("\n\n");
}


