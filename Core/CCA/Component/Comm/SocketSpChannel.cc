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


#include <iostream>
#include <sstream>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/Thread/Thread.h>


using namespace std;
using namespace SCIRun;

SocketSpChannel::SocketSpChannel() { 
  sockfd = -1;
  msg = NULL;
}

SocketSpChannel::~SocketSpChannel(){
  if(sockfd!=-1) close(sockfd);
  if(msg!=NULL) delete msg;
}

void SocketSpChannel::openConnection(const URL& url) {
  ep_url=url.getString();
  struct hostent *he;
  struct sockaddr_in their_addr; // connector's address information 

  // get the host info 
  if((he=gethostbyname(url.getHostname().c_str())) == NULL){
    throw CommError("gethostbyname", errno);
  }
     
  if( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
    throw CommError("socket", errno);
  }

  their_addr.sin_family = AF_INET;                   // host byte order 
  their_addr.sin_port = htons(url.getPortNumber());  // short, network byte order 
  their_addr.sin_addr = *((struct in_addr *)he->h_addr);
  memset(&(their_addr.sin_zero), '\0', 8);  // zero the rest of the struct 

  if(connect(sockfd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1) {
    throw CommError("connect", errno);
  }
}

SpChannel* SocketSpChannel::SPFactory(bool deep) {
  //I am not sure about this yet.
  SocketSpChannel *new_sp=new SocketSpChannel(); 
  if(deep){
    new_sp->openConnection(ep_url);
  }
  else{
    new_sp->ep_url=ep_url;
    new_sp->sockfd=sockfd;
    msg=NULL; // should I copy msg too?
  }
  return new_sp;
}

void SocketSpChannel::closeConnection() {
  if(msg==NULL)  msg = new SocketMessage(sockfd);
  msg->createMessage();
  msg->sendMessage(1);  //delete the reference
  close(sockfd);
}

Message* SocketSpChannel::getMessage() {
  if (msg == NULL){
    msg = new SocketMessage(sockfd);
    msg->setSocketSp(this);
  }
  return msg;
}















