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
 *  SocketSpChannel.cc: Socket implemenation of Sp Channel
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
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/ServerContext.h>
#include <Core/Thread/Thread.h>


using namespace std;
using namespace SCIRun;

int
SocketSpChannel::cnt_c(0);

//#define SHOW_SP_REF_COUNT

SocketSpChannel::SocketSpChannel() { 
  sp=new (struct SocketStartPoint);
  sp->sockfd=-1; 
  sp->refcnt=1;
#ifdef SHOW_SP_REF_COUNT
  cerr<<"SocketSpChannel cnt_c (NEW SP)="<<++cnt_c<<" sp="<<sp->refcnt<<endl;
#endif
}

SocketSpChannel::SocketSpChannel(struct SocketStartPoint *sp) { 
  this->sp=sp;
  sp->refcnt++;
#ifdef SHOW_SP_REF_COUNT
  cerr<<"SocketSpChannel cnt_c="<<++cnt_c<<" sp="<<sp->refcnt<<endl;
#endif
}

SocketSpChannel::~SocketSpChannel(){
  //TODO: need synchronization for sp->refcnt
  
  --(sp->refcnt);
#ifdef SHOW_SP_REF_COUNT
  cerr<<"SocketSpChannel cnt_c="<<--cnt_c<<" sp="<<sp->refcnt<<endl;
#endif
  if(sp->refcnt ==0){
    if(sp->sockfd!=-1){
      //cerr<<"Shutdown Service thread\n";
      Message *msg=getMessage();
      msg->createMessage();
      msg->sendMessage(-100); //quit service 
      msg->destroyMessage(); 
    
      close(sp->sockfd);
      sp->sockfd=-1;
      //cerr<<"sockfd closed\n";
    }
    else{
    }
    delete sp;
  }
}

void SocketSpChannel::openConnection(const URL& url) {
  if(sp->sockfd!=-1) return;
  //cerr<<"sockfd opened\n";

  struct hostent *he;
  struct sockaddr_in their_addr; // connector's address information 

  // get the host info 
  if((he=gethostbyname(url.getHostname().c_str())) == NULL){
    throw CommError("gethostbyname", errno);
  }
     
  if( (sp->sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
    throw CommError("socket", errno);
  }

  their_addr.sin_family = AF_INET;                   // host byte order 
  their_addr.sin_port = htons(url.getPortNumber());  // short, network byte order 
  their_addr.sin_addr = *((struct in_addr *)he->h_addr);
  memset(&(their_addr.sin_zero), '\0', 8);  // zero the rest of the struct 

  if(connect(sp->sockfd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1) {
    throw CommError("connect", errno);
  }

  if(sp->sockfd!=-1){
    Message *msg=getMessage();
    msg->createMessage();
    msg->sendMessage(-102); //ask for object 
    msg->waitReply();

    msg->unmarshalInt(&(sp->pid));
    msg->unmarshalInt((int*)(&(sp->object)));
    msg->destroyMessage(); 
  }

  sp->ip=their_addr.sin_addr.s_addr;
  sp->port=url.getPortNumber();
  //sp->spec
}

SpChannel* SocketSpChannel::SPFactory(bool deep) {
  SocketSpChannel *new_sp=new SocketSpChannel(sp); 
  return new_sp;
}

void SocketSpChannel::closeConnection() {
  if(sp->sockfd!=-1){
    Message *msg=getMessage();
    msg->createMessage();
    msg->sendMessage(1); 
    msg->destroyMessage(); 
  }
  else{
    cerr<<" trying local closeConnection ###\n";
    if(sp->ip==SocketEpChannel::getIP() && sp->pid==PIDL::getPID() && sp->object!=NULL){
	::SCIRun::ServerContext* _sc=static_cast< ::SCIRun::ServerContext*>(sp->object);
	_sc->d_objptr->deleteReference();
	cerr<<" local closeConnection ###\n";
    }
  }
}

//new message is created and user should call destroyMessage to delete it.
Message* SocketSpChannel::getMessage() {
  //if(sp->sockfd==-1) throw CommError("SocketSpChannel::getMessage", -1);
  SocketMessage *msg=new SocketMessage(sp->sockfd);
  msg->setLocalObject(sp->object);
  return msg;
}















