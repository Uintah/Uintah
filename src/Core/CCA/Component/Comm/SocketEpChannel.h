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

using namespace std;
#include <unistd.h>
#include <string>
#include <Core/CCA/Component/PIDL/URL.h> 
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/EpChannel.h>
#include <Core/CCA/Component/Comm/listener.h>
#include <Core/CCA/Component/Comm/Message.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/Comm/Communication.h>

#define PORT 22222

class SocketEpChannel : public EpChannel {
public:

  SocketEpChannel();
  virtual ~SocketEpChannel();
  void openConnection();
  void activateConnection(void* obj);
  void closeConnection();
  string getUrl(); 
  Message* getMessage();
  void allocateHandlerTable(int size);
  void registerHandler(int num, void* handle);
  void bind(SpChannel* spchan);

private:
  
  /////////////
  // Hostname of this computer
  string hostname;

  /////////////
  // File descriptor for the socket
  int connfd;  

  Message* msg; 
};



#endif





















