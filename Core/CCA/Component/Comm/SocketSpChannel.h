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


#ifndef SOCKET_SP_CHANNEL_H
#define SOCKET_SP_CHANNEL_H 

using namespace std;
#include <Core/CCA/Component/PIDL/URL.h> 
#include <Core/CCA/Component/Comm/SpChannel.h>
#include <Core/CCA/Component/Comm/connector.h>
#include <Core/CCA/Component/Comm/Message.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/Comm/Communication.h>

class SocketSpChannel : public SpChannel {
public:

  SocketSpChannel();
  virtual ~SocketSpChannel();
  void openConnection(const PIDL::URL& url);
  void closeConnection();
  Message* getMessage();
  SpChannel* SPFactory(bool deep);

private:
  
  /////////////
  // File descriptor for the socket
  int connfd;  

  Message* msg; 

};



#endif





















