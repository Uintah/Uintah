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
#include <Core/CCA/Comm/DT/DTAddress.h>

namespace SCIRun {

  class DTPoint;
  class DTMessage;
  class ConditionVariable;
  class Semaphore;
  class Mutex;

  class DTException{
  };

  class DataTransmitter{
  public:
    //send buf to address paddr, if no connection made for paddr,
    //build a new connection and add the entry into the socket address
    //table, sockmap.
    
    DataTransmitter();
    ~DataTransmitter();

    //interfaces:
    void putMessage(DTMessage *msg);
    DTMessage *getMessage(DTPoint *pt);

    void registerPoint(DTPoint *pt);
    void unregisterPoint(DTPoint *pt);

    std::string getUrl();

    //start the threads of listening, recieving, and sending
    void run();
    void runSendingThread();
    void runRecvingThread();
    DTAddress getAddress();
    bool isLocal(DTAddress& addr);
    void exit();
    
  private:
    void sendall(int sockfd, void *buf, int len);
    int recvall(int sockfd, void *buf, int len);
    void sendPacket(DTMessage *msg, int packetLen);    

    std::vector<DTMessage*> send_msgQ;
    std::vector<DTMessage*> recv_msgQ;

    typedef std::map<DTDestination, DTMessage*> RRMap;
    RRMap send_msgMap;

    typedef std::map<DTPacketID, DTMessage*> RVMap;
    RVMap recv_msgMap;


    typedef std::map<DTPoint *, Semaphore *> SemaphoreMap;
    SemaphoreMap semamap;

    typedef std::map<DTAddress, int> SocketMap;

    SocketMap send_sockmap;
    SocketMap recv_sockmap;

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

