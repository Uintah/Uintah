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
 *  DataTransmitter.h
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#ifndef CORE_CCA_COMM_DT_DATATRANSMITTER_H
#define CORE_CCA_COMM_DT_DATATRANSMITTER_H


#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <Core/CCA/Comm/DT/DTAddress.h>
#include <Core/CCA/Comm/DT/DTMessageTag.h>

namespace SCIRun {

  class DTPoint;
  class DTMessage;
  class ConditionVariable;
  class Semaphore;
  class Mutex;
  class Thread;
  class DTException{
  };


  class DataTransmitter{
  public:
    //send buf to address paddr, if no connection made for paddr,
    //build a new connection and add the entry into the socket address
    //table, sockmap.
    
    DataTransmitter();
    ~DataTransmitter();

    ///////////////////////////////////////////////////////
    // This is for sending initial message (caller message)
    // a message tag is generated and returned.
    DTMessageTag putMessage(DTMessage *msg);

    ///////////////////////////////////////////////////////
    // This is for sending general message with no tag,
    // for backward compatibility
    void putMsg(DTMessage *msg);

    ////////////////////////////////////////////
    // This is for reply message (callee message)
    void putReplyMessage(DTMessage *msg);


    /////////////////////////////////////////////
    // This method fetch a message with the given
    // message tag. 
    DTMessage *getMessage(const DTMessageTag &tag);

    /////////////////////////////////////////////
    // This method fetch a message with no tag for
    // backward compatibility
    DTMessage *getMsg();


    //deprecated!
    //void registerPoint(DTPoint *pt);

    //deprecated!
    //void unregisterPoint(DTPoint *pt);




    std::string getUrl();

    //start the threads of listening, recieving, and sending
    void run();
    void runSendingThread();
    void runRecvingThread();

    DTAddress getAddress();

    bool isLocal(DTAddress& addr);

    static void mpi_lock();
    static void mpi_unlock();

    void exit();
  private:
    void sendall(int sockfd, void *buf, int len);
    int recvall(int sockfd, void *buf, int len);
    void sendPacket(DTMessage *msg, int packetLen);    

    std::vector<DTMessage*> send_msgQ;
    std::vector<DTMessage*> recv_msgQ;

    typedef std::map<DTDestination, DTMessage*> RRMap;
    RRMap send_msgMap;

    typedef std::map<DTMessageTag, DTMessage*> RVMap;
    RVMap recv_msgMap;


    typedef std::map<DTMessageTag, Semaphore *> SemaphoreMap;
    SemaphoreMap semamap;

    typedef std::map<DTAddress, int> SocketMap;

    Semaphore *defaultSema; //used for default message passing between any two DTs using default message Tag

    SocketMap send_sockmap;
    SocketMap recv_sockmap;

    DTMessageTag currentTag;

    int sockfd; //listening socket

    char *hostname;

    //This Data Tansmitter's address
    DTAddress addr;

    Mutex *sendQ_mutex;
    Mutex *recvQ_mutex;

    Mutex *send_sockmap_mutex;
    Mutex *recv_sockmap_mutex;

    int newMsgCnt;

    ConditionVariable *sendQ_cond;

    const static int PACKET_SIZE=1024*32;

    bool quit;
    
  };

}//namespace SCIRun

#endif

