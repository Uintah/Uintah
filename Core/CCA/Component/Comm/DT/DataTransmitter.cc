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
 *  DataTransmitter.cc: 
 *
 *  Written by:
 *   Keming Zhang
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
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>


#include <iostream>
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/CCA/Component/Comm/DT/DataTransmitter.h>
#include <Core/CCA/Component/Comm/DT/DTThread.h>
#include <Core/CCA/Component/Comm/DT/DTPoint.h>
#include <Core/CCA/Component/Comm/DT/DTMessage.h>

using namespace SCIRun;
using namespace std;


DataTransmitter::DataTransmitter(){
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

  hostname=new char[128];
  if(gethostname(hostname, 127)==-1){
    throw CommError("gethostname", errno);
  } 

  struct hostent *he;
  if((he=gethostbyname(hostname)) == NULL){
    throw CommError("gethostbyname", errno);
  }

  addr.ip=*((long*)he->h_addr);

  socklen_t namelen=sizeof(struct sockaddr);
  if(getsockname(sockfd, (struct sockaddr*)&my_addr, &namelen )==-1){
    throw CommError("getsockname", errno);
  }  
  addr.port=ntohs(my_addr.sin_port);

  sendQ_mutex=new Mutex("sendQ_mutex");
  recvQ_mutex=new Mutex("recvQ_mutex");
  sendQ_cond=new ConditionVariable("sendQ_cond");
  recvQ_cond=new ConditionVariable("recvQ_cond");

  quit=false;
}

DataTransmitter::~DataTransmitter(){
  delete sendQ_mutex;
  delete recvQ_mutex;
  delete sendQ_cond;
  delete recvQ_cond;
}

void 
DataTransmitter::putMessage(DTMessage *msg){
  //cerr<<"putMessage to point="<<(long)msg->recver<<endl;
  msg->fr_addr=addr;
  //need synchronization
  sendQ_mutex->lock();
  send_msgQ.push_back(msg);
  sendQ_mutex->unlock();
  sendQ_cond->conditionSignal();
}

DTMessage *
DataTransmitter::getMessage(DTPoint *pt){
  //cerr<<"getMessage for point="<<(long)pt<<endl;
  //assuming only one or zero thread can access getMessage
  recvQ_mutex->lock();
  while(recv_msgQ.size()==0){
      recvQ_cond->wait(*recvQ_mutex);
  }
  DTMessage *msg=NULL;
  deque<DTMessage*>::iterator iter=recv_msgQ.begin();
  for(deque<DTMessage*>::iterator iter=recv_msgQ.begin(); iter!=recv_msgQ.end(); iter++){
    if( (*iter)->recver==pt){
      msg=*iter;
      recv_msgQ.erase(iter);
      break;
    }
  }
  //  recv_msgQ.pop_front();
  recvQ_mutex->unlock();
  return msg;
}

void 
DataTransmitter::run(){
  Thread *listening_thread = new Thread(new DTThread(this, 0), "Data Transmitter Listening Thread", 0, Thread::Activated);
  listening_thread->detach();

  Thread *sending_thread = new Thread(new DTThread(this, 1), "Data Transmitter Sending Thread", 0, Thread::Activated);
  sending_thread->detach();

  Thread *recving_thread = new Thread(new DTThread(this, 2), "Data Transmitter Recving Thread", 0, Thread::Activated);
  recving_thread->detach();
}

void 
DataTransmitter::runListeningThread(){
  //at most 1024 waiting clients
  if (listen(sockfd, 10) == -1){ 
    throw CommError("listen", errno);
  }

  cerr<<"DataTransmitter is Listening: URL="<<getUrl()<<endl;

  fd_set read_fds; // temp file descriptor list for select()
  struct timeval timeout;
  // add the listener to the master set

  while(!quit){
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
      //immediately register the new process address
      DTAddress newAddr;
      newAddr.ip=their_addr.sin_addr.s_addr;
      short listPort;
      newAddr.port=ntohs(their_addr.sin_port);
      recvall(new_fd, &(newAddr.port), sizeof(short));
      sockmap[newAddr]=new_fd;
      cerr<<"Accept connection from port"<<newAddr.port<<"  sockmap.size="<<sockmap.size()<<endl;
    }
  }
  close(sockfd);
  sockfd=-1;
  for(SocketMap::iterator iter=sockmap.begin(); iter!=sockmap.end(); iter++){
    //should wait sending and receving threads to finish
    //close(iter->second);
    //shutdown(iter->second, SHUT_RDWR);
  }
}


void 
DataTransmitter::runSendingThread(){
  //cerr<<"DataTransmitter is Sending"<<endl;
  while(true){ //!quit){
	
    //use while is safer
    sendQ_mutex->lock();
    while(send_msgQ.empty()){
      if(quit) return;
      sendQ_cond->wait(*sendQ_mutex);
      if(send_msgQ.empty() && quit) return;
    }
    sendQ_mutex->unlock();
    //cerr<<"trying to send a message"<<endl;
    DTMessage *msg=send_msgQ.front();
    SocketMap::iterator iter=sockmap.find(msg->to_addr);
    int new_fd;
    if(iter==sockmap.end()){
      new_fd=socket(AF_INET, SOCK_STREAM, 0);
      if( new_fd  == -1){
	throw CommError("socket", errno);
      }
      struct sockaddr_in their_addr; // connector's address information 
      their_addr.sin_family = AF_INET;                   // host byte order 
      their_addr.sin_port = htons(msg->to_addr.port);  // short, network byte order 
      their_addr.sin_addr = *(struct in_addr*)(&(msg->to_addr.ip));
      memset(&(their_addr.sin_zero), '\0', 8);  // zero the rest of the struct 
      
      if(connect(new_fd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1) {
	perror("connect");
	throw CommError("connect", errno);
      }
      
      //immediate register the listening port

      sendall(new_fd, &addr.port, sizeof(short));

      sockmap[msg->to_addr]=new_fd;
      //cerr<<"add an entry to sockmap, new size="<<sockmap.size()<<endl;
    }
    else{
      new_fd=iter->second;
    }

    sendall(new_fd, &(msg->recver), sizeof(void *));
    sendall(new_fd, &(msg->sender), sizeof(void *));
    sendall(new_fd, &(msg->fr_addr), sizeof(DTAddress));
    sendall(new_fd, &(msg->length), sizeof(long));
    sendall(new_fd, msg->buf, msg->length);
    cerr<<"Message is sent to point="<<(long)msg->recver<<endl;
    msg->display();
    delete msg;

    sendQ_mutex->lock();
    send_msgQ.pop_front();
    sendQ_mutex->unlock();
  }
}

void 
DataTransmitter::runRecvingThread(){
  //cerr<<"DataTransmitter is Recving"<<endl;

  fd_set read_fds; // temp file descriptor list for select()
  struct timeval timeout;
  while(!quit){
    timeout.tv_sec=0;
    timeout.tv_usec=0;
    FD_ZERO(&read_fds);
    // add the listener to the master set
    int maxfd=0;
    for(SocketMap::iterator iter=sockmap.begin(); iter!=sockmap.end(); iter++){
      FD_SET(iter->second, &read_fds);
      if(maxfd<iter->second) maxfd=iter->second;
    }
    if (select(maxfd+1, &read_fds, NULL, NULL, &timeout) == -1) {
      throw CommError("select", errno);
    }
    // run through the existing connections looking for data to read

    for(SocketMap::iterator iter=sockmap.begin(); iter!=sockmap.end(); iter++){
      if(FD_ISSET(iter->second, &read_fds)){
	DTMessage *msg=new DTMessage;

	if(recvall(iter->second, &(msg->recver), sizeof(void *))!=0){
	  recvall(iter->second, &(msg->sender), sizeof(void *));
	  recvall(iter->second, &(msg->fr_addr), sizeof(DTAddress));
	  recvall(iter->second, &(msg->length), sizeof(long));
	  msg->buf=(char *)malloc(msg->length);
	  recvall(iter->second, msg->buf, msg->length);
	  
	  msg->to_addr=addr;
	  msg->autofree=true;
	  msg->fr_addr=iter->first;
	  
	  msg->display();
	
	  recvQ_mutex->lock();
	  recv_msgQ.push_back(msg);
	  recvQ_mutex->unlock();
	  recvQ_cond->conditionSignal();
	  
	  SemaphoreMap::iterator found=semamap.find(msg->recver);
	  if(found!=semamap.end()){
	    found->second->up();
	  }
	  else{
	    //discard the message
	    cerr<<"warning: message discarded!\n";
	  }
	}
	else{//catch(DTException){
	  cerr<<"receive 0 bytes"<<endl;
	  close(iter->second);
	  sockmap.erase(iter);
	}
      }
    }
  }
  cerr<<"RecvThread quits when recvQ size="<<recv_msgQ.size()<<endl;
}


string 
DataTransmitter::getUrl() {
  std::ostringstream o;
  o << "socket2://" << hostname << ":" << addr.port << "/";
  return o.str();
}


void
DataTransmitter::sendall(int sockfd, void *buf, int len)
{
  int total = 0;        // how many bytes we've sent
  int n;
  
  while(total < len) {
    n = send(sockfd, (char*)buf+total, len, 0);
    if (n == -1) throw CommError("recv", errno);
    total += n;
    len -= n;
  }
} 

int
DataTransmitter::recvall(int sockfd, void *buf, int len)
{
  int total = 0;        // how many bytes we've recved
  int n;
  
  while(total < len) {
    n = recv(sockfd, (char*)buf+total, len, 0);
    if (n == -1) throw CommError("recv", errno);
    if(n==0) return 0;//throw DTException();
    total += n;
    len -= n;
  }
  return total;
} 

void DataTransmitter::registerPoint(DTPoint * pt){
  //cerr<<"registerPoint: "<<(long)pt<<endl;
  semamap[pt]=pt->sema;
}

void DataTransmitter::unregisterPoint(DTPoint * pt){
  SemaphoreMap::iterator iter=semamap.find(pt);
  if(iter!=semamap.end()) semamap.erase(iter);
}

DTAddress
DataTransmitter::getAddress(){
  return addr;
}


bool 
DataTransmitter::isLocal(DTAddress& addr)
{
  return this->addr==addr;
}

void 
DataTransmitter::exit(){
  quit=true;
  sendQ_cond->conditionSignal(); //awake the sendingThread
}
