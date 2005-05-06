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
 *  DataTransmitter.cc: 
 *
 *  Written by:
 *   Keming Zhang / heavily modified by McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#else
#include <winsock2.h>
#define socklen_t int
#endif

#include <iostream>
#include <string>
#include <sstream>

#include <Core/Comm/CommError.h>
#include <Core/Comm/StringSocket.h>
#include <Core/Comm/StringSocketThread.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>

using namespace SCIRun;
using namespace std;

StringSocket::StringSocket(int port)  
{
  struct sockaddr_in my_addr;    // my address information
  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    throw CommError("socket", errno);
  }
  my_addr.sin_family = AF_INET;         // host byte order
  my_addr.sin_port = htons(port);       // automatically select an unused port
  my_addr.sin_addr.s_addr = htonl(INADDR_ANY); // automatically fill with my IP
  memset(&(my_addr.sin_zero), '\0', 8); // zero the rest of the struct
  
  if (::bind(sockfd, (struct sockaddr *)&my_addr, sizeof(struct sockaddr)) == -1) {
    if (errno == 98) 
      cerr << "Port " << port << " already in use. Exiting...\n";
    else
      cerr << "Error: " << errno << " opening socket.  Exiting...\n";
    Thread::exitAll(1);
    
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
  recv_sema = new Semaphore("Receiver",0);

  send_sockmap_mutex=new Mutex("send_sockmap_mutex");
  recv_sockmap_mutex=new Mutex("recv_sockmap_mutex");

  sendQ_cond=new ConditionVariable("sendQ_cond");

  quit=false;
}

StringSocket::~StringSocket(){
  delete sendQ_mutex;
  delete recvQ_mutex;
  delete send_sockmap_mutex;
  delete recv_sockmap_mutex;
  delete sendQ_cond;
  delete hostname;
}


void 
StringSocket::putMessage(const string &str){
  sendQ_mutex->lock();
  send_queue.push(str);
  sendQ_mutex->unlock();
  sendQ_cond->conditionSignal();
}

string
StringSocket::getMessage()
{
  recv_sema->down();
  string ret_val;
  recvQ_mutex->lock();
  ret_val = recv_queue.front();
  recv_queue.pop();
  recvQ_mutex->unlock();
  return ret_val;
}

void 
StringSocket::run()
{
  //at most 1 client
  if (listen(sockfd, 10) == -1){ 
    throw CommError("listen", errno);
  }
  
  Thread *sending_thread = new Thread(new StringSocketThread(this, 1), "Data Transmitter Sending Thread", 0, Thread::NotActivated);
  sending_thread->setStackSize(1024*256);
  sending_thread->activate(false);
  sending_thread->detach();

  Thread *recving_thread = new Thread(new StringSocketThread(this, 2), "Data Transmitter Recving Thread", 0, Thread::NotActivated);
  recving_thread->setStackSize(1024*256);
  recving_thread->activate(false);
  recving_thread->detach();
}


void 
StringSocket::runSendingThread()
{
  while(true)
  {
    if(send_queue.empty())
    {
      sendQ_mutex->lock();
      while(send_queue.empty())
      {
	if(quit)
	{
	  sendQ_mutex->unlock();
	  for(SocketMap::iterator iter=send_sockmap.begin(); 
	      iter!=send_sockmap.end(); iter++){
	    close(iter->second);
	  }
	  sendQ_mutex->unlock();
	  return;
	}
	sendQ_cond->wait(*sendQ_mutex);
      }
      sendQ_mutex->unlock();
    } else {
      sendQ_mutex->lock();
      sendPacket(send_queue.front());
      send_queue.pop();
      sendQ_mutex->unlock();
    }
  }
}

void 
StringSocket::runRecvingThread()
{
  fd_set read_fds; // temp file descriptor list for select()
  struct timeval timeout;
  while(!quit || !recv_queue.empty()){
    timeout.tv_sec=0;
    timeout.tv_usec=20000;
    FD_ZERO(&read_fds);
    // add the listener to the master set
    int maxfd=sockfd;
    FD_SET(sockfd, &read_fds);

    // add all other sockets into read_fds
    for(SocketMap::iterator iter=recv_sockmap.begin(); 
	iter!=recv_sockmap.end(); iter++){
      FD_SET(iter->second, &read_fds);
      if(maxfd<iter->second) maxfd=iter->second;
    }
    if (select(maxfd+1, &read_fds, NULL, NULL, &timeout) == -1) {
      throw CommError("select", errno);
    }

    // run through the existing connections looking for data to read
    for(SocketMap::iterator iter=recv_sockmap.begin(); 
	iter!=recv_sockmap.end(); iter++)
    {
      const int socket_fd = (*iter).second;
      if(FD_ISSET(socket_fd, &read_fds)){
	char data[2];
	if(recvall(socket_fd, data, 1)!=0){
	  data[1] = 0;
	  map<int, string>::iterator buff = recv_buff.find(socket_fd);
	  if (buff == recv_buff.end()) {
	    buff = (recv_buff.insert(make_pair(socket_fd,string()))).first;
	    (*buff).second.reserve(80);
	  }
	  switch (data[0]) {
	  case 0x0d:(*buff).second.append("\n"); break;
	    //	  case '"': (*buff).second.append("\\\""); break;
	    //	  case '$': (*buff).second.append("\\$"); break;
	    //case '{': (*buff).second.append("\\{"); break;
	    //case '}': (*buff).second.append("\\}"); break;

	  default:  (*buff).second.append(data); break;
	  }

	  if (data[0] ==  0x0d) {
	    recvQ_mutex->lock();
	    recv_queue.push((*buff).second);
	    recvQ_mutex->unlock();
	    recv_sema->up();
	    recv_buff.erase(buff);
	    char dummy;
	    recvall(socket_fd, &dummy, 1);
	  }
	} else {
	  cerr << "Closing TCL socket connection.";
	  close(socket_fd);
	  recv_sockmap_mutex->lock();
	  recv_sockmap.erase(iter);
	  recv_sockmap_mutex->unlock();
	}
      }
    }
    
    
    // check the new connection requests
    if(FD_ISSET(sockfd, &read_fds))
    {
      socklen_t sin_size = sizeof(struct sockaddr_in);
      sockaddr_in their_addr;
      //Waiting for socket connections ...;

      int new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);
      if (new_fd == -1) {
	throw CommError("accept", errno);
      }
      if (!recv_sockmap.empty()) {
	close (new_fd);
	continue;
      }
          
      static int protocol_id = -1;
#if defined(__sgi) || defined(__APPLE__)
      // SGI does not have SOL_TCP defined.  To the best of my knowledge
      // (ie: reading the man page) this is what you are supposed to do. (Dd)
      if(protocol_id == -1)
      {
	struct protoent * p = getprotobyname("TCP");
	if(p == NULL)
	{
	  cout << "StringSocket.cc Error: Lookup of protocol TCP failed!\n";
	  exit();
	  return;
	}
	protocol_id = p->p_proto;
      }
      int yes = 1;
#else
#ifdef _WIN32
      protocol_id = IPPROTO_TCP;
      char yes = 1; // the windows version of setsockopt takes a char*
#else
      protocol_id = SOL_TCP;
      int yes = 1;
#endif
#endif

      if(setsockopt(new_fd, protocol_id, TCP_NODELAY, &yes, sizeof(int))==-1) {
	perror("setsockopt");
      }
      
      //immediately register the new process address
      //there is no way to get the remote listening port number
      //so it has to be sent explcitly.
      CommAddress newAddr;
      newAddr.ip=ntohl(their_addr.sin_addr.s_addr);
      newAddr.port=ntohs(their_addr.sin_port);
      recv_sockmap_mutex->lock();
      recv_sockmap[newAddr]=new_fd;
      recv_sockmap_mutex->unlock();
      putMessage("scirun> ");
      cerr << std::endl << "TCL socket connection from: " 
	   << int((newAddr.ip>>24)&255) << "." 
	   << int((newAddr.ip>>16)&255) << "." 
	   << int((newAddr.ip>> 8)&255) << "."
	   << int(newAddr.ip&255) << ":" << (newAddr.port) << std::endl;
    }
    
  }

  close(sockfd);
  sockfd=-1;

  //sending sockets and recving sockets are different, so we can close the 
  //recving sockets before the recving thread quits.
  recv_sockmap_mutex->lock();
  for(SocketMap::iterator iter=recv_sockmap.begin(); iter!=recv_sockmap.end(); iter++){
    close(iter->second);
  }
  recv_sockmap_mutex->unlock();
}

string 
StringSocket::getUrl() {
  std::ostringstream o;
  o << "socket://" << hostname << ":" << addr.port << "/";
  return o.str();
}


void 
StringSocket::sendPacket(const string &str) {
  recv_sockmap_mutex->lock();
  sendall((*recv_sockmap.begin()).second, str.c_str(), str.length());
  recv_sockmap_mutex->unlock();
#if 0

  SocketMap::iterator iter=send_sockmap.find(addr);
  int new_fd;
  if(0 && iter==send_sockmap.end()){
    new_fd=socket(AF_INET, SOCK_STREAM, 0);
    if( new_fd  == -1){
      throw CommError("socket", errno);
    }
    
    static int protocol_id = -1;
#if defined(__sgi) || defined(__APPLE__)
    // SGI does not have SOL_TCP defined.  To the best of my knowledge
    // (ie: reading the man page) this is what you are supposed to do. (Dd)
    if( protocol_id == -1 ){
      struct protoent * p = getprotobyname("TCP");
      if( p == NULL ){
	cout << "StringSocket.cc: Error.  TCP protocol lookup failed!\n";
	exit();
	return;
      }
      protocol_id = p->p_proto;
    }
#else
    protocol_id = SOL_TCP;
#endif
    int yes=1;
    if( setsockopt( new_fd, protocol_id, TCP_NODELAY, &yes, sizeof(int) ) == -1 ) {
      perror("setsockopt");
    }
    struct sockaddr_in their_addr; // connector's address information 
    their_addr.sin_family = AF_INET;                   // host byte order 
    their_addr.sin_port = htons(addr.port);  // short, network byte order 
    their_addr.sin_addr = *(struct in_addr*)(&(addr.ip));
    memset(&(their_addr.sin_zero), '\0', 8);  // zero the rest of the struct 
    if(connect(new_fd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1) {
      perror("connect");
      throw CommError("connect", errno);
    }
    //immediate register the listening port
    sendall(new_fd, &addr.port, sizeof(short));
    send_sockmap_mutex->lock();
    send_sockmap[addr]=new_fd;
    send_sockmap_mutex->unlock();
  }
  else{
    new_fd=iter->second;
  }
#endif

}

void
StringSocket::sendall(int sockfd, const void *buf, int len)
{
  int left=len;
  int total = 0;        // how many bytes we've sent
  while(total < len) {
    int n = send(sockfd, (char*)buf+total, left, 0);
    if (n == -1) throw CommError("send", h_errno);
    total += n;
    left -= n;
  }
} 


int
StringSocket::recvall(int sockfd, void *buf, int len)
{
  int left=len;
  int total = 0;        // how many bytes we've recved
  while(total < len) {
  int flags;
#ifdef _WIN32
    flags = 0;
#else
    flags = MSG_WAITALL;
#endif
    int n = recv(sockfd, (char*)buf+total, left, flags);
#if 0
    cerr.setf(ios::hex, ios::basefield);
    unsigned char *blah = (unsigned char *)buf;
    cerr << "got: " << int(blah[0]) << std::endl;
    cerr.setf(ios::dec, ios::basefield);
#endif
    if (n == -1) throw CommError("recv", errno);
    if(n==0) return 0;
    total += n;
    left -= n;
  }
  return total;
} 

void 
StringSocket::exit(){
  quit=true;
  sendQ_cond->conditionSignal(); //wake up the sendingThread
}
