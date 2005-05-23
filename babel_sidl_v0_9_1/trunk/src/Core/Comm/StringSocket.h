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
 *  StringSocket.h
 *
 *  Written by:
 *   Keming Zhang / heavily modified by McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#ifndef CORE_COMM_STRINGSOCKET_H
#define CORE_COMM_STRINGSOCKET_H


#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <queue>
#include <Core/Comm/CommAddress.h>

namespace SCIRun {

  class ConditionVariable;
  class Semaphore;
  class Mutex;

  class StringSocket {
  public:
    //send buf to address paddr, if no connection made for paddr,
    //build a new connection and add the entry into the socket address
    //table, sockmap.
    
    StringSocket(int port = 0);
    ~StringSocket();

    //interfaces:
    void putMessage(const std::string &);
    std::string getMessage();
    std::string getUrl();

    //start the threads of listening, recieving, and sending
    void run();
    void runSendingThread();
    void runRecvingThread();
    void exit();
    
  private:
    void sendall(int sockfd, const void *buf, int len);
    int  recvall(int sockfd, void *buf, int len);
    void sendPacket(const std::string &);


    std::queue<std::string>	send_queue;
    std::queue<std::string>	recv_queue;
    std::map<int, std::string>	recv_buff;
    Semaphore*			recv_sema;

    typedef std::map<CommAddress, int> SocketMap;

    SocketMap send_sockmap;
    SocketMap recv_sockmap;

    int sockfd; //listening socket

    char *hostname;

    //This Data Tansmitter's address
    CommAddress addr;

    Mutex *sendQ_mutex;
    Mutex *recvQ_mutex;

    Mutex *send_sockmap_mutex;
    Mutex *recv_sockmap_mutex;

    ConditionVariable *sendQ_cond;

    bool quit;
  };

}//namespace SCIRun

#endif

