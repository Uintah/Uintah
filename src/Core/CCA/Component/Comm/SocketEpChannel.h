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


#ifndef SOCKET_EP_CHANNEL_H
#define SOCKET_EP_CHANNEL_H 

#include <Core/CCA/Component/Comm/EpChannel.h>

namespace SCIRun {
  class Message;
  class SpChannel;
  class SocketEpChannel : public EpChannel {
  public:

    SocketEpChannel();
    virtual ~SocketEpChannel();
    void openConnection();
    void activateConnection(void* obj);
    void closeConnection();
    std::string getUrl(); 
    Message* getMessage();
    void allocateHandlerTable(int size);
    void registerHandler(int num, void* handle);
    void bind(SpChannel* spchan);

  private:
  
    /////////////
    // File descriptor for the socket
    int sockfd;  

    ////////////
    // The table of handlers from the sidl generated file
    // used to relay a message to them
    HPF* handler_table;

    /////////////
    // Handler table size
    int table_size;

    Message* msg; 
  };
}

#endif
