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


#ifndef CORE_CCA_COMPONENT_COMM_DT_DATATRANSMITTER_H
#define CORE_CCA_COMPONENT_COMM_DT_DATATRANSMITTER_H


#include <iostream>
#include <deque>
#include <map>
#include <string>
#include <Core/CCA/Component/Comm/DT/DTAddress.h>

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
    

    std::deque<DTMessage*> send_msgQ;
    std::deque<DTMessage*> recv_msgQ;
    typedef std::map<DTPoint *, Semaphore *> SemaphoreMap;
    SemaphoreMap semamap;

    static const int SEGMENT_LEN=1024*32;
    typedef std::map<DTAddress, int> SocketMap;
    SocketMap sockmap;


    int sockfd; //listening socket

    char *hostname;
    DTAddress addr;

    Mutex *sendQ_mutex;
    Mutex *recvQ_mutex;
    Mutex *sockmap_mutex;

    ConditionVariable *sendQ_cond;
    ConditionVariable *recvQ_cond;

    //number of TD threads have quit
    //0: all are running, 3: all quit
    //int nquit; 

    bool quit;
    bool quitSending;
  };

}//namespace SCIRun

#endif

