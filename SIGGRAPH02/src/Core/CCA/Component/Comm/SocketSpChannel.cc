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


#include "SocketSpChannel.h"
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/PIDL/URL.h>
using namespace SCIRun;

SocketSpChannel::SocketSpChannel() { 
  connfd = 0;
  msg = 0;
}

SocketSpChannel::~SocketSpChannel() { }

void SocketSpChannel::openConnection(const URL& /*url*/) {
  //Call connector and pass it the args
  //connfd = open_connection(url.getHostname().c_str(),url.getPortNumber());
}

SpChannel* SocketSpChannel::SPFactory(bool /*deep*/) {
  SocketSpChannel* new_sp = new SocketSpChannel(); 
  return new_sp;
}

void SocketSpChannel::closeConnection() {
  //close_connection(connfd);

}

Message* SocketSpChannel::getMessage() {
  if (connfd == 0)
    return NULL;
  if (msg == 0)
    msg = new SocketMessage();
  return msg;
}




















