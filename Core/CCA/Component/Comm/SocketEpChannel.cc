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

#include "SocketEpChannel.h"
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>


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
  sockfd=0;
  /*  
  while(1) {  // main accept() loop
    sin_size = sizeof(struct sockaddr_in);
    if ((new_fd = accept(sockfd, (struct sockaddr *)&their_addr,
			 &sin_size)) == -1) {
      perror("accept");
      continue;
    }
    printf("server: got connection from %s\n",
                                               inet_ntoa(their_addr.sin_addr));
    if (!fork()) { // this is the child process
      close(sockfd); // child doesn't need the listener
      if (send(new_fd, "Hello, world!\n", 14, 0) == -1)
	perror("send");
      close(new_fd);
      exit(0);
    }
    close(new_fd);  // parent doesn't need this
        }
  
  return 0;
  */
  msg = NULL;
}

SocketEpChannel::~SocketEpChannel(){ 
  if(handler_table!=NULL) delete []handler_table;
  if(msg!=NULL) delete msg;
}


void SocketEpChannel::openConnection() {
  //at most 1024 waiting clients
  if (listen(sockfd, 1024) == -1){ 
    throw CommError("socket", errno);
  }
}

void SocketEpChannel::closeConnection() {
  close(sockfd);
  sockfd=0;
}

string SocketEpChannel::getUrl() {
  struct sockaddr_in my_addr;
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
  o << "socket://" << hostname << ":" << my_addr.sin_port << "/";
  delete []hostname;
  return o.str();
}

void SocketEpChannel::activateConnection(void* obj){
  //need create one thread to do this
  /*
  while(true) {  // main accept() loop
    sin_size = sizeof(struct sockaddr_in);
    if ((new_fd = accept(sockfd, (struct sockaddr *)&their_addr,
			 &sin_size)) == -1) {
      perror("accept");
      continue;
    }
    printf("server: got connection from %s\n",
                                               inet_ntoa(their_addr.sin_addr));
    if (!fork()) { // this is the child process
      close(sockfd); // child doesn't need the listener
      if (send(new_fd, "Hello, world!\n", 14, 0) == -1)
	perror("send");
      close(new_fd);
      exit(0);
    }
    close(new_fd);  // parent doesn't need this
  */
}

Message* SocketEpChannel::getMessage() {
  if (sockfd == 0)
    return NULL;
  if (msg == NULL)
    msg = new SocketMessage();
  return msg;
}

void SocketEpChannel::allocateHandlerTable(int size){
  handler_table = new HPF[size];
  table_size = size;
}

void SocketEpChannel::registerHandler(int num, void* handle){
  handler_table[num-1] = (HPF) handle;
}

void SocketEpChannel::bind(SpChannel* /*spchan*/){
  //does nothing, at this time
}










