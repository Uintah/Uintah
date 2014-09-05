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
#include <sys/time.h>


#include <iostream>
#include <Core/CCA/Exceptions/CommError.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/CCA/DT/DataTransmitter.h>
#include <Core/CCA/DT/DTThread.h>
#include <Core/CCA/DT/DTPoint.h>
#include <Core/CCA/DT/DTMessage.h>
#include <deque>

using namespace SCIRun;

DataTransmitter::DataTransmitter() : newMsgCnt(0), sending_thread(0), recving_thread(0), quit(false)
{
  struct sockaddr_in my_addr;    // my address information
  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    throw CommError("socket", __FILE__, __LINE__, errno);
  }

  my_addr.sin_family = AF_INET;         // host byte order
  my_addr.sin_port = 0;                 // automatically select an unused port
  my_addr.sin_addr.s_addr = htonl(INADDR_ANY); // automatically fill with my IP
  memset(&(my_addr.sin_zero), '\0', 8); // zero the rest of the struct

  if (::bind(sockfd, (struct sockaddr *)&my_addr, sizeof(struct sockaddr)) == -1) {
    throw CommError("bind", __FILE__, __LINE__, errno);
  }

  hostname = new char[128];
  if (gethostname(hostname, 127) == -1) {
    throw CommError("gethostname", __FILE__, __LINE__, errno);
  }

  struct hostent *he;
  if ((he = gethostbyname(hostname)) == 0) {
    int saveErrno = errno;
    char* buf =  strerror(saveErrno);
    throw CommError(std::string("\'gethostbyname\' ") + buf, __FILE__, __LINE__, saveErrno);
  }

  addr.setIP(*((long*)he->h_addr));

  socklen_t namelen = sizeof(struct sockaddr);
  if (getsockname(sockfd, (struct sockaddr*)&my_addr, &namelen ) == -1) {
    throw CommError("getsockname", __FILE__, __LINE__, errno);
  }
  addr.setPort(ntohs(my_addr.sin_port));

  sendQ_mutex = new Mutex("sendQ_mutex");
  recvQ_mutex = new Mutex("recvQ_mutex");

  send_sockmap_mutex = new Mutex("send_sockmap_mutex");
  recv_sockmap_mutex = new Mutex("recv_sockmap_mutex");

  sendQ_cond = new ConditionVariable("sendQ_cond");
  defaultSema = new Semaphore("semaphore for default message tag", 0);
}

DataTransmitter::~DataTransmitter()
{
  delete sendQ_mutex;
  delete recvQ_mutex;
  delete send_sockmap_mutex;
  delete recv_sockmap_mutex;
  delete sendQ_cond;
  delete hostname;
}


DTMessageTag
DataTransmitter::putMessage(DTMessage *msg)
{
  msg->fr_addr = addr;

  DTMessageTag newTag = currentTag.nextTag();

  //  cout<<"===DT putMessage: "<<newTag.lo<<" Destination="; msg->getDestination().display();

  msg->tag = newTag;
  /////////////////////////////////
  // if the msg is sent to a local
  // address, process it right away.
  if (isLocal(msg->to_addr)) {
    msg->autofree = true;
    DTPoint *pt = msg->recver;

    //generate a new tag for this message
    //this is already thread-safe
    if (pt->service != 0) {
      pt->service(msg);
    } else {
      throw CommError("Client trying to send a message to a client.", __FILE__, __LINE__, -1);
    }
  } else {
    msg->offset = 0;
    sendQ_mutex->lock();
    send_msgQ.push_back(msg);
    newMsgCnt++;
    sendQ_mutex->unlock();
    sendQ_cond->conditionSignal();
  }
  return newTag;
}



void
DataTransmitter::putMsg(DTMessage *msg)
{
  //cout<<"===DT putMsg: "<<msg->tag.lo<<" Destination="; msg->getDestination().display();
  msg->fr_addr = addr;

  DTMessageTag newTag = currentTag.defaultTag();
  msg->tag = newTag;
  /////////////////////////////////
  // if the msg is sent to a local
  // address, process it right away.
  if (isLocal(msg->to_addr)) {
    msg->autofree = true;
    recvQ_mutex->lock();
    recv_msgQ.push_back(msg);
    defaultSema->up();
    recvQ_mutex->unlock();
  } else {
    msg->offset = 0;
    sendQ_mutex->lock();
    send_msgQ.push_back(msg);
    newMsgCnt++;
    sendQ_mutex->unlock();
    sendQ_cond->conditionSignal();
  }
}


void
DataTransmitter::putReplyMessage(DTMessage *msg)
{
  //  cout<<"===DT putReplyMessage: "<<msg->tag.lo<<" Destination="; msg->getDestination().display();
  msg->fr_addr = addr;

  /////////////////////////////////
  // if the msg is sent to a local
  // address, process it right away.
  if (isLocal(msg->to_addr)) {
    msg->autofree = true;
    DTPoint *pt = msg->recver;

    if (pt->service != 0) {
      throw CommError("Server trying to send a message to a server.", __FILE__, __LINE__, -1);
    } else {
      recvQ_mutex->lock();
      recv_msgQ.push_back(msg);
      SemaphoreMap::iterator found = semamap.find(msg->tag);
      if (found != semamap.end())  found->second->up();
      recvQ_mutex->unlock();
    }
  } else {
    msg->offset = 0;
    sendQ_mutex->lock();
    send_msgQ.push_back(msg);
    newMsgCnt++;
    sendQ_mutex->unlock();
    sendQ_cond->conditionSignal();
  }
}

DTMessage *
DataTransmitter::getMessage(const DTMessageTag &tag)
{
  recvQ_mutex->lock();
  DTMessage *msg = 0;
  for (std::vector<DTMessage*>::iterator iter = recv_msgQ.begin(); iter != recv_msgQ.end(); iter++) {
    if ( (*iter)->tag == tag) {
      msg = *iter;
      recv_msgQ.erase(iter);
      recvQ_mutex->unlock();
      return msg;
    }
  }

  //now, the expected message has not arrived yet
  //create a semaphore for it and then wait
  semamap[tag] = new Semaphore("reply message semaphore", 0);
  recvQ_mutex->unlock();
  semamap[tag]->down();

  recvQ_mutex->lock();
  //after the message arrives, fetch it and delete the semaphore
  delete semamap[tag];
  semamap.erase(tag);
  for (std::vector<DTMessage*>::iterator iter = recv_msgQ.begin(); iter != recv_msgQ.end(); iter++) {
    if ( (*iter)->tag == tag) {
      msg = *iter;
      recv_msgQ.erase(iter);
      recvQ_mutex->unlock();
      return msg;
    }
  }
  throw CommError("Semaphore up but not message received", __FILE__, __LINE__, -1);
  return 0;
}



DTMessage *
DataTransmitter::getMsg()
{
  recvQ_mutex->lock();
  DTMessage *msg = 0;
  defaultSema->down();
  for (std::vector<DTMessage*>::iterator iter = recv_msgQ.begin(); iter != recv_msgQ.end(); iter++) {
    if ( (*iter)->tag == currentTag.defaultTag()) {
      msg = *iter;
      recv_msgQ.erase(iter);
      recvQ_mutex->unlock();
      return msg;
    }
  }
  throw CommError("Semaphore up but not message received", __FILE__, __LINE__, -1);
  return 0;
}

void
DataTransmitter::run()
{
  //at most 10 waiting clients
  if (listen(sockfd, 10) == -1) {
    throw CommError("listen", __FILE__, __LINE__, errno);
  }
  std::cout << "DataTransmitter is listening: URL=" << getUrl() << std::endl;

  sending_thread = new Thread(new DTThread(this, 1), "Data Transmitter Sending Thread", 0, Thread::NotActivated);
  sending_thread->setStackSize(STACK_SIZE);
  sending_thread->activate(false);
  //sending_thread->detach();

  recving_thread = new Thread(new DTThread(this, 2), "Data Transmitter Recving Thread", 0, Thread::NotActivated);
  recving_thread->setStackSize(STACK_SIZE);
  recving_thread->activate(false);
  //recving_thread->detach();
}


void
DataTransmitter::runSendingThread()
{
  //cout<<"DataTransmitter is Sending"<<endl;
  DTDestination sentRecver;
  sentRecver.unset();
  //iterator of Round Robin queue, indicating next message.
  RRMap::iterator rrIter;

  while (true) {
    sendQ_mutex->lock();
    if (sentRecver.isSet()) {
      //one complete message has been sent
      //sendQ_mutex->lock();
      //cout<<"sentRecver != NULL and send_msgQ.size="<<send_msgQ.size()<<endl;
      for (unsigned int i = 0; i < send_msgQ.size(); i++) {
        if (send_msgQ[i]->getDestination() == sentRecver) {
          send_msgMap[sentRecver] = send_msgQ[i];
          rrIter = send_msgMap.begin();
          if (i >= send_msgQ.size() - newMsgCnt) {
            //if it happens that one new message is processed here,
            //we should update newMsgCnt
            newMsgCnt--;
          }
          send_msgQ.erase(send_msgQ.begin()+i);
          //cout<<"retrive one sentRecver from sendQ send_msgQ.size="<<send_msgQ.size()<<endl;
          break;
        }
      }
      //cout<<"when quiting sentRecver != NULL send_msgQ.size="<<send_msgQ.size()<<endl;
      //sendQ_mutex->unlock();
      sentRecver.unset();
    }

    //sendQ_mutex->lock();
    while (newMsgCnt > 0) {
      //cout<<"newMsgCnt="<<newMsgCnt<<endl;
      //some new messages have entered the send queue
      std::vector<DTMessage*>::iterator newMsg = send_msgQ.end()-newMsgCnt;
      if (send_msgMap.find( (*newMsg)->getDestination()) == send_msgMap.end()) {
        //the new message is sent to a new receiver, too.
        send_msgMap[(*newMsg)->getDestination()] = *newMsg;
        rrIter = send_msgMap.begin();
        //cout<<"BEFORE move one newMsg into rrQ send_msgQ.size = "<<send_msgQ.size()<<endl;
        send_msgQ.erase(newMsg);
        //cout<<"move one newMsg into rrQ send_msgQ.size="<<send_msgQ.size()<<endl;
      }
      //else     cout<<"keep one newMsg in sendQ"<<endl;
      newMsgCnt--;
      //break;
    }
    sendQ_mutex->unlock();

    if (send_msgMap.empty()) {
      sendQ_mutex->lock();
      while (send_msgQ.empty()) {
        //cout<<"send_msgQ is empty()"<<endl;
        if (quit) {
          sendQ_mutex->unlock();
          //sending sockets and recving sockets are different, so we can close the
          //sending sockets before the sending thread quits.
          for (SocketMap::iterator iter = send_sockmap.begin(); iter != send_sockmap.end(); iter++) {
            close(iter->second);
          }
          sendQ_mutex->unlock();
          return;
        }
        sendQ_cond->wait(*sendQ_mutex);
      }
      sendQ_mutex->unlock();
    } else {
      DTMessage *msg= rrIter->second;
      int packetLen = msg->length-msg->offset;
      if (packetLen > PACKET_SIZE) {
        packetLen = PACKET_SIZE;
        msg->offset += packetLen;
        sendPacket(msg, packetLen);
        rrIter++;
        if (rrIter == send_msgMap.end()) rrIter = send_msgMap.begin();
      } else {
        //done with this message
        msg->offset += packetLen;
        sendPacket(msg, packetLen);
        sentRecver = msg->getDestination();
        send_msgMap.erase(rrIter);
        rrIter = send_msgMap.begin();
        std::string id;
        if ((int)(msg->buf[0]) == -101) id = " ADD ";
        if ((int)(msg->buf[0]) == -102) id = " DEL ";
        if ((int)(msg->buf[0]) == 0) id = " ADD_ISA1 ";
        if ((int)(msg->buf[0]) == 1) id = " DEL_HAN2 ";
        if ((int)(msg->buf[0]) == 3) id = " ADD_HAN4 ";
        if ((int)(msg->buf[0]) == 4) id = " DEL_HAN5 ";
        //if (id != "")
        //cout<<"===DT Send Message: "<<msg->tag.lo<<id<<" Destination="; msg->getDestination().display();
        //  cout<<"Send Message:";
        //msg->display();
        delete msg;
      }
    }
  }

}

void
DataTransmitter::runRecvingThread()
{
  fd_set read_fds; // temp file descriptor list for select()
  struct timeval timeout;
  while (!quit || !recv_msgMap.empty()) {
    timeout.tv_sec = 0;
    timeout.tv_usec = 20000;
    FD_ZERO(&read_fds);
    // add the listener to the master set
    int maxfd = sockfd;
    FD_SET(sockfd, &read_fds);

    // add all other sockets into read_fds
    for (SocketMap::iterator iter = recv_sockmap.begin(); iter != recv_sockmap.end(); iter++) {
      FD_SET(iter->second, &read_fds);
      if (maxfd < iter->second) maxfd = iter->second;
    }
    if (select(maxfd+1, &read_fds, 0, 0, &timeout) == -1) {
      int saveErrno = errno;
      // lock?
      char* buf =  strerror(saveErrno);
      throw CommError(std::string("select error: ") + buf, __FILE__, __LINE__, saveErrno);
    }

    // run through the existing connections looking for data to read
    for (SocketMap::iterator iter = recv_sockmap.begin(); iter != recv_sockmap.end(); iter++) {
      if (FD_ISSET(iter->second, &read_fds)) {
        DTMessage *msg = new DTMessage;
        if (recvall(iter->second, msg, sizeof(DTMessage)) != 0) {

          RVMap::iterator rrIter = recv_msgMap.find(msg->tag);
          if (rrIter == recv_msgMap.end()) {
            //this is the first packet
            msg->buf = (char *)malloc(msg->length);
            recv_msgMap[msg->tag] = msg;
            recvall(iter->second, msg->buf, msg->offset);
            msg->autofree = true;
          } else {
            //this is the one packet after the first
            int packetLen = msg->offset-rrIter->second->offset;
            msg->autofree = false;
            delete msg;
            msg = rrIter->second;
            recvall(iter->second, msg->buf+msg->offset, packetLen);
            msg->offset += packetLen;
          }

          //check if a complete  message received.
          if (msg->offset == msg->length) {
            DTPoint *pt = msg->recver;
            recv_msgMap.erase(msg->tag);

            std::string id;
            if (!(msg->tag == currentTag.defaultTag())) {
              if ((int)(msg->buf[0]) == -101) id = " ADD ";
              if ((int)(msg->buf[0]) == -102) id = " DEL ";
              if ((int)(msg->buf[0]) == 0) id = " ADD_ISA1 ";
              if ((int)(msg->buf[0]) == 1) id = " DEL_HAN2 ";
              if ((int)(msg->buf[0]) == 3) id = " ADD_HAN4 ";
              if ((int)(msg->buf[0]) == 4) id = " DEL_HAN5 ";
            }
            //if (id != "")
            // cout<<"===DT Recv Message: "<<msg->tag.lo<<id<<" Destination="; msg->getDestination().display();

            if (pt->service != 0) {
              pt->service(msg);
            } else {
              if (msg->tag == currentTag.defaultTag()) {
                defaultSema->up();
              } else {
                recvQ_mutex->lock();
                recv_msgQ.push_back(msg);
                SemaphoreMap::iterator found = semamap.find(msg->tag);
                if (found != semamap.end()) found->second->up();
                recvQ_mutex->unlock();
              }
            }
          }
        } else {
          //remote connection is closed, if receive 0 bytes
          close(iter->second);
          recv_sockmap_mutex->lock();
          recv_sockmap.erase(iter);
          recv_sockmap_mutex->unlock();
          msg->autofree = false;
          delete msg;
        }
      }
    }


    // check the new connection requests
    if (FD_ISSET(sockfd, &read_fds)) {
      int new_fd;
      socklen_t sin_size = sizeof(struct sockaddr_in);
      sockaddr_in their_addr;
      //Waiting for socket connections ...;
      if ((new_fd = accept(sockfd, (struct sockaddr *)&their_addr,
                           &sin_size)) == -1) {
        throw CommError("accept", __FILE__, __LINE__, errno);
      }

      static int protocol_id = -1;
#if defined(__sgi) || defined(__APPLE__)
      // SGI does not have SOL_TCP defined.  To the best of my knowledge
      // (ie: reading the man page) this is what you are supposed to do. (Dd)
      if(protocol_id == -1) {
        struct protoent * p = getprotobyname("TCP");
        if(p == 0) {
          std::cout << "DataTransmitter.cc: Error.  Lookup of protocol TCP failed!\n";
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

      if ( setsockopt( new_fd, protocol_id, TCP_NODELAY, &yes, sizeof(int) ) == -1 ) {
        perror("setsockopt");
      }

      //immediately register the new process address
      //there is no way to get the remote listening port number,
      //so it has to be sent explcitly.
      DTAddress newAddr;
      newAddr.setIP(their_addr.sin_addr.s_addr);
      newAddr.setPort(ntohs(their_addr.sin_port));
      int _port = newAddr.getPort();
      recvall(new_fd, &_port, sizeof(short));
      //cout<<"Done register port "<<newAddr.port<<endl;
      recv_sockmap_mutex->lock();
      recv_sockmap[newAddr] = new_fd;
      recv_sockmap_mutex->unlock();
    }
  }

  close(sockfd);
  sockfd =- 1;

  //sending sockets and recving sockets are different, so we can close the
  //recving sockets before the recving thread quits.
  for (SocketMap::iterator iter = recv_sockmap.begin(); iter != recv_sockmap.end(); iter++) {
    close(iter->second);
  }
}

std::string
DataTransmitter::getUrl()
{
  std::ostringstream o;
  o << "socket://" << hostname << ":" << addr.getPort() << "/";
  return o.str();
}


void
DataTransmitter::sendPacket(DTMessage *msg, int packetLen)
{
  SocketMap::iterator iter = send_sockmap.find(msg->to_addr);
  int new_fd;
  if (iter == send_sockmap.end()) {
    new_fd = socket(AF_INET, SOCK_STREAM, 0);
    if ( new_fd  == -1) {
      throw CommError("socket", __FILE__, __LINE__, errno);
    }

    static int protocol_id = -1;
#if defined(__sgi) || defined(__APPLE__)
    // SGI does not have SOL_TCP defined.  To the best of my knowledge
    // (ie: reading the man page) this is what you are supposed to do. (Dd)
    if(protocol_id == -1) {
      struct protoent * p = getprotobyname("TCP");
      if(p == 0) {
        std::cout << "DataTransmitter.cc: Error.  Lookup of protocol TCP failed!\n";
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

    if ( setsockopt( new_fd, protocol_id, TCP_NODELAY, &yes, sizeof(int) ) == -1 ) {
      perror("setsockopt");
    }

    struct sockaddr_in their_addr; // connector's address information
    their_addr.sin_family = AF_INET;                   // host byte order
    their_addr.sin_port = htons(msg->to_addr.getPort());  // short, network byte order
    int _ip = msg->to_addr.getIP();
    their_addr.sin_addr = *(struct in_addr*)(&_ip);
    memset(&(their_addr.sin_zero), '\0', 8);  // zero the rest of the struct

    if (connect(new_fd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1) {
      perror("connect");
      throw CommError("connect", __FILE__, __LINE__, errno);
    }
    //immediate register the listening port
    //cout<<"register port "<<addr.port<<endl;
    int _port = addr.getPort();
    sendall(new_fd, &_port, sizeof(short));
    send_sockmap_mutex->lock();
    send_sockmap[msg->to_addr] = new_fd;
    send_sockmap_mutex->unlock();
  } else {
    new_fd = iter->second;
  }
  sendall(new_fd, msg, sizeof(DTMessage));
  sendall(new_fd, msg->buf+msg->offset-packetLen, packetLen);
}

void
DataTransmitter::sendall(int sockfd, void *buf, int len)
{
  int left = len;
  int total = 0;        // how many bytes we've sent
  while (total < len) {
    int n = send(sockfd, (char*)buf+total, left, 0);
    if (n == -1) throw CommError("recv", __FILE__, __LINE__, errno);
    total += n;
    left -= n;
  }
}


int
DataTransmitter::recvall(int sockfd, void *buf, int len)
{
  int left = len;
  int total = 0;        // how many bytes we've recved
  while (total < len) {
    int n = recv(sockfd, (char*)buf+total, left, MSG_WAITALL);
    if (n == -1) throw CommError("recv", __FILE__, __LINE__, errno);
    if (n == 0) return 0;
    total += n;
    left -= n;
  }
  return total;
}

#if 0
//////////////////////////////////////////////////////////////////////////
//   void DataTransmitter::registerPoint(DTPoint * pt){
//   semamap[pt]=pt->sema;
//   }
//
//   void DataTransmitter::unregisterPoint(DTPoint * pt){
//   semamap.erase(pt);
//   }
//////////////////////////////////////////////////////////////////////////
#endif

DTAddress
DataTransmitter::getAddress()
{
  return addr;
}

bool
DataTransmitter::isLocal(DTAddress& addr)
{
  return this->addr == addr;
}

void
DataTransmitter::exit()
{
  quit = true;
  sendQ_cond->conditionSignal(); //wake up the sendingThread

  recving_thread->join();
  recving_thread = 0;
  sending_thread->join();
  sending_thread = 0;
}
