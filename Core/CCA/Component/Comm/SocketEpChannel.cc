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
//#include <sys/wait.h>

#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/Comm/SocketThread.h>
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/ServerContext.h>

#include <Core/Thread/Thread.h>

using namespace std;
using namespace SCIRun;

SocketEpChannel::SocketEpChannel(){ 
  int new_fd;  // listen on sock_fd, new connection on new_fd
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
  msg = NULL;
  object=NULL;
}

SocketEpChannel::~SocketEpChannel(){ 
  if(handler_table!=NULL) delete []handler_table;
  if(msg!=NULL) delete msg;
}


void SocketEpChannel::openConnection() {
  //at most 1024 waiting clients
  if (listen(sockfd, 10) == -1){ 
    throw CommError("listen", errno);
  }
}

void SocketEpChannel::closeConnection() {
  close(sockfd);
  sockfd=0;
}

string SocketEpChannel::getUrl() {
  struct sockaddr_in my_addr;
  my_addr.sin_port=11111;
  socklen_t namelen=sizeof(struct sockaddr);
  if(getsockname(sockfd, (struct sockaddr*)&my_addr, &namelen )==-1){
    throw CommError("getsockname", errno);
  }
  char *hostname=new char[128];
  //strcpy(hostname,inet_ntoa(my_addr.sin_addr));
  std::ostringstream o;
  if(gethostname(hostname, 127)==-1){
    throw CommError("gethostname", errno);
  } 
  o << "socket://" << hostname << ":" << ntohs(my_addr.sin_port) << "/";
  delete []hostname;
  return o.str();
}

void SocketEpChannel::activateConnection(void* obj){
  object=obj;
  Thread* t = new Thread(new SocketThread(this, -1), "SocketAcceptThread", 0, Thread::Activated);
  t->detach();
}

Message* SocketEpChannel::getMessage() {
  if (sockfd == 0)
    return NULL;
  if (msg == NULL)
    throw CommError("SocketEpChannel::getMessage", -1);
  return msg;
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
  cerr<<"****bind is called *****"<<endl;
  spchan->openConnection(URL(getUrl()));
  //might save spchan for reference
}


void 
SocketEpChannel::runAccept(){
  //cerr<<"SocketAcceptThread is running\n";
  while(sockfd!=0){
    int new_fd;
    socklen_t sin_size = sizeof(struct sockaddr_in);
    sockaddr_in their_addr;
    //cerr<<"Waiting for socket connections...\n";
    if ((new_fd = accept(sockfd, (struct sockaddr *)&their_addr,
			 &sin_size)) == -1) {
      throw CommError("accept", errno);
    }

    //printf("server: got connection from %s\n", inet_ntoa(their_addr.sin_addr));
    if(object==NULL) throw CommError("Access Null object", -1);
    ::SCIRun::ServerContext* _sc=static_cast< ::SCIRun::ServerContext*>(object);
    _sc->d_objptr->addReference();
    
    Thread* t = new Thread(new SocketThread(this, -2, new_fd), "SocketServiceThread", 0, Thread::Activated);
    t->detach();

    //add some break condition

  }  
}


void 
SocketEpChannel::runService(int new_fd){
  //cerr<<"SocketServiceThread is running\n";
  while(true){
    int headerSize=sizeof(long)+sizeof(int);
    void *buf=malloc(headerSize);
    int numbytes=0;
    while(numbytes==0){
      if ((numbytes=recv(new_fd, buf, headerSize, MSG_WAITALL)) == -1) {
	throw CommError("recv", errno);
      }
    }

    if(numbytes!=headerSize){
      //cerr<<"????? numbytes=, headerSize="<<numbytes<<", "<<headerSize<<endl;
      CommError("numbytes!=headerSize",-1);
    }

    int msg_size;
    int id;
    memcpy(&msg_size, buf, sizeof(long));
    memcpy(&id, (char*)buf+sizeof(long), sizeof(int));

    //cerr<<"=======RECV(runService)   msg_size, id="<<msg_size<<", "<<id<<endl;

    buf=realloc(buf, msg_size-headerSize);
    numbytes=0;
    while(numbytes==0 && msg_size-headerSize!=0){
      if ((numbytes=recv(new_fd, buf, msg_size-headerSize, MSG_WAITALL)) == -1) {
	throw CommError("recv", errno);
      }
    }
    if(numbytes!=msg_size-headerSize) throw CommError("numbytes!=msg_size-headerSize",-1);

    if(msg!=NULL) delete msg;
    msg=new SocketMessage(new_fd, buf);
    msg->setSocketEp(this);

    Thread* t = new Thread(new SocketThread(this, id, new_fd), "SocketHandlerThread", 0, Thread::Activated);
    t->detach();

    //sleep(2); //avoid the synchornization

    //The SocketHandlerThread is responsible to free the buf.   

    if(id==1){
      close(new_fd);
      break;
    }
  }  
}


