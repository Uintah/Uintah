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
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
//#include <sys/wait.h>

#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/Comm/SocketThread.h>
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/ServerContext.h>

#include <Core/Thread/Thread.h>

using namespace std;
using namespace SCIRun;

long SocketEpChannel::ip(0);

SocketEpChannel::SocketEpChannel(){ 
  struct sockaddr_in my_addr;    // my address information
  
  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    throw CommError("socket", errno);
  }
  
  my_addr.sin_family = AF_INET;         // host byte order
  my_addr.sin_port = 0;                 // automatically select an unused port
  my_addr.sin_addr.s_addr = htonl(INADDR_ANY); // automatically fill with my IP
  memset(&(my_addr.sin_zero), '\0', 8); // zero the rest of the struct
  
  if (::bind(sockfd, (struct sockaddr *)&my_addr, sizeof(struct sockaddr)) == -1) {
    throw CommError("bind", errno);
  }

  handler_table=NULL;
  object=NULL;
  accept_thread=NULL;
  dead=false;

  hostname=new char[128];
  if(gethostname(hostname, 127)==-1){
    throw CommError("gethostname", errno);
  } 

  struct hostent *he;
  if((he=gethostbyname(hostname)) == NULL){
    throw CommError("gethostbyname", errno);
  }

  ip=*((long*)he->h_addr);

  socklen_t namelen=sizeof(struct sockaddr);
  if(getsockname(sockfd, (struct sockaddr*)&my_addr, &namelen )==-1){
    throw CommError("getsockname", errno);
  }  
  port=ntohs(my_addr.sin_port);
}

SocketEpChannel::~SocketEpChannel(){ 
  if(handler_table!=NULL) delete []handler_table;
  delete []hostname;
}


void SocketEpChannel::openConnection() {
  //at most 1024 waiting clients
  if (listen(sockfd, 10) == -1){ 
    throw CommError("listen", errno);
  }
}

void SocketEpChannel::closeConnection() {
  //close(sockfd);
  //sockfd=-1;
}

string SocketEpChannel::getUrl() {

  std::ostringstream o;
  o << "socket://" << hostname << ":" << port << "/";
  return o.str();
}

void SocketEpChannel::activateConnection(void* obj){
  object=obj;
  accept_thread = new Thread(new SocketThread(this, NULL,  -1), "SocketAcceptThread", 0, Thread::Activated);
  accept_thread->detach();
}

Message* SocketEpChannel::getMessage() {
  //this should not be never called!
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
  SocketStartPoint *sp=chan->sp;
  sp->ip=ip;
  sp->port=port;
  sp->pid=PIDL::getPID();
  sp->object=object;

  /*
  struct sockaddr_in their_addr; // connector's address information 

  if( (sp->sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
    throw CommError("socket", errno);
  }

  their_addr.sin_family = AF_INET;                   // host byte order 
  their_addr.sin_port = htons(sp->port);  // short, network byte order 
  their_addr.sin_addr = *(struct in_addr*)(&(sp->ip));
  memset(&(their_addr.sin_zero), '\0', 8);  // zero the rest of the struct 

  if(connect(sp->sockfd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1) {
    perror("connect #2");
    throw CommError("connect", errno);
  }

  Message *msg=chan->getMessage();
  msg->createMessage();
  msg->sendMessage(-101);  //call deleteReference
  msg->destroyMessage(); 

  sp->object=NULL;
  */
}


void 
SocketEpChannel::runAccept(){
  fd_set read_fds; // temp file descriptor list for select()
  struct timeval timeout;
  // add the listener to the master set

  while(!dead && sockfd!=-1){
    timeout.tv_sec=0;
    timeout.tv_usec=0;
    FD_ZERO(&read_fds);
    FD_SET(sockfd, &read_fds);
    if (select(sockfd+1, &read_fds, NULL, NULL, &timeout) == -1) {
      throw CommError("select", errno);
    }
    // run through the existing connections looking for data to read
    if(FD_ISSET(sockfd, &read_fds)){
      int new_fd;
      socklen_t sin_size = sizeof(struct sockaddr_in);
      sockaddr_in their_addr;
      //Waiting for socket connections ...;
      if ((new_fd = accept(sockfd, (struct sockaddr *)&their_addr,
			   &sin_size)) == -1) {
	throw CommError("accept", errno);
      }
      
      //printf("server: got connection from %s\n", inet_ntoa(their_addr.sin_addr));
      if(object==NULL) throw CommError("Access Null object", -1);
      ::SCIRun::ServerContext* _sc=static_cast< ::SCIRun::ServerContext*>(object);
      _sc->d_objptr->addReference();
      
      Thread *t= new Thread(new SocketThread(this, NULL, -2, new_fd), "SocketServiceThread", 0, Thread::Activated);
      t->detach();
    }
    // for test only, should not keep, because if object is already deleted, segmentation fault occurs.
    ::SCIRun::ServerContext* _sc=static_cast< ::SCIRun::ServerContext*>(object);
    if(_sc->d_objptr->getRefCount()==0) break;
   }
  close(sockfd);
  sockfd=-1;
}


void 
SocketEpChannel::runService(int new_fd){
  //cerr<<"ServiceThread starts\n";
  while(true){
    int headerSize=sizeof(long)+sizeof(int);
    void *buf=malloc(headerSize);
    int numbytes=0;
    if (SocketMessage::recvall(new_fd, buf, headerSize) == -1) {
	throw CommError("recv", errno);
    }

    int msg_size;
    int id;
    memcpy(&msg_size, buf, sizeof(long));
    memcpy(&id, (char*)buf+sizeof(long), sizeof(int));

    //cerr<<"SERVICE RECV MSG:"<<msg_size<<" bytes, handlerID="<<id<<endl;

    buf=realloc(buf, msg_size-headerSize);
    numbytes=0;
    if((numbytes=SocketMessage::recvall(new_fd, buf, msg_size-headerSize)) == -1) {
      throw CommError("recv", errno);
    }



    //filter internal messages
    if(id<=-100){
      if(id==-100){ //quit service
	close(new_fd);
	break;
      }
      else if(id==-101){ //delete Reference
	::SCIRun::ServerContext* _sc=static_cast< ::SCIRun::ServerContext*>(object);
	_sc->d_objptr->deleteReference();
      
      }
      else if(id==-102){ //return object, pid
	SocketMessage* msg=new SocketMessage(new_fd);
	msg->createMessage();
	int pid=PIDL::getPID();
	msg->marshalInt(&pid);
	msg->marshalInt((int*)(&object));
	msg->sendMessage(0);
	msg->destroyMessage();
      }
    }
    else{
      SocketMessage* new_msg=new SocketMessage(new_fd, buf);
      new_msg->setLocalObject(object);
      //The SocketHandlerThread is responsible to free the buf.   

      Thread* t = new Thread(new SocketThread(this, new_msg, id, new_fd), "SocketHandlerThread", 0, Thread::Activated);
      t->detach();
    }
  }  
  //cerr<<"ServiceThread stops\n";
}


