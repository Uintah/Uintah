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
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <Core/Thread/Time.h>
#include <sys/time.h>


#include <iostream>
#include <Core/CCA/Comm/CommError.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <Core/CCA/Comm/DT/DTThread.h>
#include <Core/CCA/Comm/DT/DTPoint.h>
#include <Core/CCA/Comm/DT/DTMessage.h>

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

  send_sockmap_mutex=new Mutex("send_sockmap_mutex");
  recv_sockmap_mutex=new Mutex("recv_sockmap_mutex");

  sendQ_cond=new ConditionVariable("sendQ_cond");
  newMsgCnt=0;
  quit=false;
}

DataTransmitter::~DataTransmitter(){
  delete sendQ_mutex;
  delete recvQ_mutex;
  delete send_sockmap_mutex;
  delete recv_sockmap_mutex;
  delete sendQ_cond;
  delete hostname;
}


void 
DataTransmitter::putMessage(DTMessage *msg){
  msg->fr_addr=addr;

  /////////////////////////////////
  // if the msg is sent to the same process
  // process it right away.
  if(msg->to_addr==addr){
    msg->autofree=true;
    DTPoint *pt=msg->recver;
    SemaphoreMap::iterator found=semamap.find(pt);
    //cerr<<"Send & Recv Message:\n";
    //msg->display();
    
    if(found!=semamap.end()){
      if(pt->service!=NULL){
	pt->service(msg);
      }
      else{
	recvQ_mutex->lock();
	recv_msgQ.push_back(msg);
	recvQ_mutex->unlock();
	found->second->up();
      }
    }
    else{
      //discard the message
      cerr<<"warning: message discarded!\n";
    }
  }
  else{
    msg->offset=0;
    sendQ_mutex->lock();
    //cerr<<"####BEFORE got new message and send_msgQ.size="<<send_msgQ.size()<<endl;
    send_msgQ.push_back(msg);
    newMsgCnt++;
    //cerr<<"got new message and send_msgQ.size="<<send_msgQ.size()<<endl;
    sendQ_mutex->unlock();
    sendQ_cond->conditionSignal();
  }
}

DTMessage *
DataTransmitter::getMessage(DTPoint *pt, int tag){
  recvQ_mutex->lock();
  DTMessage *msg=NULL;
  for(vector<DTMessage*>::iterator iter=recv_msgQ.begin(); iter!=recv_msgQ.end(); iter++){
    if( (*iter)->recver==pt && (*iter)->tag==tag){
      msg=*iter;
      recv_msgQ.erase(iter);
      break;
    }
  }
  recvQ_mutex->unlock();
  return msg;
}

void 
DataTransmitter::run(){
  //at most 10 waiting clients
  if (listen(sockfd, 10) == -1){ 
    throw CommError("listen", errno);
  }
  //cerr<<"DataTransmitter is Listening: URL="<<getUrl()<<endl;

  Thread *sending_thread = new Thread(new DTThread(this, 1), "Data Transmitter Sending Thread", 0, Thread::NotActivated);
  sending_thread->setStackSize(1024*256);
  sending_thread->activate(false);
  sending_thread->detach();

  Thread *recving_thread = new Thread(new DTThread(this, 2), "Data Transmitter Recving Thread", 0, Thread::NotActivated);
  recving_thread->setStackSize(1024*256);
  recving_thread->activate(false);
  recving_thread->detach();
}


void 
DataTransmitter::runSendingThread(){
  //cerr<<"DataTransmitter is Sending"<<endl;
  DTDestination sentRecver;
  sentRecver.unset();
  //iterator of Round Robin queue, indicating next message.
  RRMap::iterator rrIter;
  
  while(true){
    sendQ_mutex->lock();
    if(sentRecver.isSet()){
      //one complete message has been sent
      //sendQ_mutex->lock();
      //cerr<<"sentRecver!=NULL and send_msgQ.size="<<send_msgQ.size()<<endl;
      for(unsigned int i=0; i<send_msgQ.size(); i++){
	if(send_msgQ[i]->getDestination()==sentRecver){
	  send_msgMap[sentRecver]=send_msgQ[i];
	  rrIter=send_msgMap.begin();
	  if(i>=send_msgQ.size()-newMsgCnt){
	    //if it happens that one new message is processed here,
	    //we should update newMsgCnt
	    newMsgCnt--;
	  }
	  send_msgQ.erase(send_msgQ.begin()+i);
	  //cerr<<"retrive one sentRecver from sendQ send_msgQ.size="<<send_msgQ.size()<<endl;
	  break;
	}
      }
      //cerr<<"when quiting sentRecver!=NULL send_msgQ.size="<<send_msgQ.size()<<endl;
      //sendQ_mutex->unlock();
      sentRecver.unset();
    }
    
    //sendQ_mutex->lock();
    while(newMsgCnt>0){
      //cerr<<"newMsgCnt="<<newMsgCnt<<endl;
      //some new messages have entered the send queue    
      std::vector<DTMessage*>::iterator newMsg=send_msgQ.end()-newMsgCnt;
      if(send_msgMap.find( (*newMsg)->getDestination())==send_msgMap.end()){
	//the new message is sent to a new receiver, too.
	send_msgMap[(*newMsg)->getDestination()]=*newMsg;
	rrIter=send_msgMap.begin();
	//cerr<<"BEFORE move one newMsg into rrQ send_msgQ.size="<<send_msgQ.size()<<endl;
	send_msgQ.erase(newMsg);
	//cerr<<"move one newMsg into rrQ send_msgQ.size="<<send_msgQ.size()<<endl;
      }
      //else     cerr<<"keep one newMsg in sendQ"<<endl;
      newMsgCnt--;
      //break;
    }
    sendQ_mutex->unlock();

    if(send_msgMap.empty()){
      sendQ_mutex->lock();
      while(send_msgQ.empty()){
	//cerr<<"send_msgQ is empty()"<<endl;
	if(quit){
	  sendQ_mutex->unlock();
	  //sending sockets and recving sockets are different, so we can close the 
	  //sending sockets before the sending thread quits.
	  for(SocketMap::iterator iter=send_sockmap.begin(); iter!=send_sockmap.end(); iter++){
	    close(iter->second);
	  }
	  sendQ_mutex->unlock();
	  return;
	}
	sendQ_cond->wait(*sendQ_mutex);
      }
      sendQ_mutex->unlock();
    }
    else{
      DTMessage *msg= rrIter->second;
      int packetLen=msg->length-msg->offset;
      if(packetLen>PACKET_SIZE){
	packetLen=PACKET_SIZE;
	msg->offset+=packetLen;
	sendPacket(msg, packetLen);
	rrIter++;
	if(rrIter == send_msgMap.end()) rrIter=send_msgMap.begin();
      }
      else{
	//done with this message
	msg->offset+=packetLen;
	sendPacket(msg, packetLen);
	sentRecver=msg->getDestination();
	send_msgMap.erase(rrIter);
	rrIter=send_msgMap.begin();
	//cerr<<"Send Message:";
	//msg->display();
	delete msg;
      }
    }
  }

}

void 
DataTransmitter::runRecvingThread(){
  fd_set read_fds; // temp file descriptor list for select()
  struct timeval timeout;
  while(!quit || !recv_msgMap.empty()){
    timeout.tv_sec=0;
    timeout.tv_usec=20000;
    FD_ZERO(&read_fds);
    // add the listener to the master set
    int maxfd=sockfd;
    FD_SET(sockfd, &read_fds);

    // add all other sockets into read_fds
    for(SocketMap::iterator iter=recv_sockmap.begin(); iter!=recv_sockmap.end(); iter++){
      FD_SET(iter->second, &read_fds);
      if(maxfd<iter->second) maxfd=iter->second;
    }
    if (select(maxfd+1, &read_fds, NULL, NULL, &timeout) == -1) {
      throw CommError("select", errno);
    }

    // run through the existing connections looking for data to read
    for(SocketMap::iterator iter=recv_sockmap.begin(); iter!=recv_sockmap.end(); iter++){
      if(FD_ISSET(iter->second, &read_fds)){
	DTMessage *msg=new DTMessage;
	if(recvall(iter->second, msg, sizeof(DTMessage))!=0){

	  RVMap::iterator rrIter=recv_msgMap.find(msg->getPacketID());
	  if(rrIter==recv_msgMap.end()){
	    //this is the first packet
	    msg->buf=(char *)malloc(msg->length);	    
	    recv_msgMap[msg->getPacketID()]=msg;
	    recvall(iter->second, msg->buf, msg->offset);
	    msg->autofree=true;
	  }
	  else{
	    //this is the one packet after the first
	    int packetLen=msg->offset-rrIter->second->offset;
	    msg->autofree=false;
	    delete msg;
	    msg=rrIter->second;
	    recvall(iter->second, msg->buf+msg->offset, packetLen);
	    msg->offset+=packetLen;
	  }

	  //check if a complete  message received.
	  if(msg->offset==msg->length){
	    DTPoint *pt=msg->recver;
	    recv_msgMap.erase(msg->getPacketID());

	    SemaphoreMap::iterator found=semamap.find(pt);
	    //cerr<<"Recv Message:";
	    //msg->display();

	    if(found!=semamap.end()){
	      if(pt->service!=NULL){
		pt->service(msg);
	      }
	      else{
		recvQ_mutex->lock();
		recv_msgQ.push_back(msg);
		recvQ_mutex->unlock();
		found->second->up();
	      }
	    }
	    else{
	      //discard the message
	      cerr<<"warning: message discarded!\n";
	    }
	  }
	}
	else{
	  //remote connection is closed, if receive 0 bytes
	  close(iter->second);
	  recv_sockmap_mutex->lock();
	  recv_sockmap.erase(iter);
	  recv_sockmap_mutex->unlock();
	  msg->autofree=false;
	  delete msg;
	}
      }
    }
    
    
    // check the new connection requests
    if(FD_ISSET(sockfd, &read_fds)){
      int new_fd;
      socklen_t sin_size = sizeof(struct sockaddr_in);
      sockaddr_in their_addr;
      //Waiting for socket connections ...;
      if ((new_fd = accept(sockfd, (struct sockaddr *)&their_addr,
			   &sin_size)) == -1) {
	throw CommError("accept", errno);
      }
      
      static int protocol_id = -1;
#if defined(__sgi)
      // SGI does not have SOL_TCP defined.  To the best of my knowledge
      // (ie: reading the man page) this is what you are supposed to do. (Dd)
      if( protocol_id == -1 )
	{
	  struct protoent * p = getprotobyname("TCP");
	  if( p == NULL )
	    {
	      cout << "DataTransmitter.cc: Error.  Lookup of protocol TCP failed!\n";
	      exit();
	      return;
	    }
	  protocol_id = p->p_proto;
	}
#else
      protocol_id = SOL_TCP;
#endif
      
      int yes = 1;
      if( setsockopt( new_fd, protocol_id, TCP_NODELAY, &yes, sizeof(int) ) == -1 ) {
	perror("setsockopt");
      }
      
      //immediately register the new process address
      //there is no way to get the remote listening port number,
      //so it has to be sent explcitly.
      DTAddress newAddr;
      newAddr.ip=their_addr.sin_addr.s_addr;
      newAddr.port=ntohs(their_addr.sin_port);
      recvall(new_fd, &(newAddr.port), sizeof(short));
      //cerr<<"Done register port "<<newAddr.port<<endl;
      recv_sockmap_mutex->lock();
      recv_sockmap[newAddr]=new_fd;
      recv_sockmap_mutex->unlock();
    }
  }

  close(sockfd);
  sockfd=-1;

  //sending sockets and recving sockets are different, so we can close the 
  //recving sockets before the recving thread quits.
  for(SocketMap::iterator iter=recv_sockmap.begin(); iter!=recv_sockmap.end(); iter++){
    close(iter->second);
  }
}

string 
DataTransmitter::getUrl() {
  std::ostringstream o;
  o << "socket://" << hostname << ":" << addr.port << "/";
  return o.str();
}


void 
DataTransmitter::sendPacket(DTMessage *msg, int packetLen){
  SocketMap::iterator iter=send_sockmap.find(msg->to_addr);
  int new_fd;
  if(iter==send_sockmap.end()){
    new_fd=socket(AF_INET, SOCK_STREAM, 0);
    if( new_fd  == -1){
      throw CommError("socket", errno);
    }
    
    static int protocol_id = -1;
#if defined(__sgi)
    // SGI does not have SOL_TCP defined.  To the best of my knowledge
    // (ie: reading the man page) this is what you are supposed to do. (Dd)
    if( protocol_id == -1 ){
      struct protoent * p = getprotobyname("TCP");
      if( p == NULL ){
	cout << "DataTransmitter.cc: Error.  TCP protocol lookup failed!\n";
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
    their_addr.sin_port = htons(msg->to_addr.port);  // short, network byte order 
    their_addr.sin_addr = *(struct in_addr*)(&(msg->to_addr.ip));
    memset(&(their_addr.sin_zero), '\0', 8);  // zero the rest of the struct 
    
    if(connect(new_fd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1) {
      perror("connect");
      throw CommError("connect", errno);
    }
    //immediate register the listening port
    //cerr<<"register port "<<addr.port<<endl;
    sendall(new_fd, &addr.port, sizeof(short));
    send_sockmap_mutex->lock();
    send_sockmap[msg->to_addr]=new_fd;
    send_sockmap_mutex->unlock();
  }
  else{
    new_fd=iter->second;
  }
  sendall(new_fd, msg, sizeof(DTMessage));
  sendall(new_fd, msg->buf+msg->offset-packetLen, packetLen);
}

void
DataTransmitter::sendall(int sockfd, void *buf, int len)
{
  int left=len;
  int total = 0;        // how many bytes we've sent
  while(total < len) {
    int n = send(sockfd, (char*)buf+total, left, 0);
    if (n == -1) throw CommError("recv", errno);
    total += n;
    left -= n;
  }
} 


int
DataTransmitter::recvall(int sockfd, void *buf, int len)
{
  int left=len;
  int total = 0;        // how many bytes we've recved
  while(total < len) {
    int n = recv(sockfd, (char*)buf+total, left, MSG_WAITALL);
    if (n == -1) throw CommError("recv", errno);
    if(n==0) return 0;
    total += n;
    left -= n;
  }
  return total;
} 

void DataTransmitter::registerPoint(DTPoint * pt){
  semamap[pt]=pt->sema;
}

void DataTransmitter::unregisterPoint(DTPoint * pt){
  semamap.erase(pt);
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
  sendQ_cond->conditionSignal(); //wake up the sendingThread
}
