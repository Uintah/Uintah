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


#include "SocketEpChannel.h"
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <sstream>
#include <unistd.h>
using namespace std;
using namespace SCIRun;
#define PORT 7675

SocketEpChannel::SocketEpChannel() { 
  char name [255];
  if (gethostname(name, 255) != 0) {
    throw CommError("Can't resolve machine hostname. SocketEpChannel::SocketEpChannel()",11);
  }
  else {    
    hostname = name;
  }
  msg = NULL;
}

SocketEpChannel::~SocketEpChannel() { }

void SocketEpChannel::openConnection() {
  //Call listener and pass it the args
  //connfd = openListener(PORT);

}

void SocketEpChannel::closeConnection() {
  //closeListener(connfd);

}

string SocketEpChannel::getUrl() {

  std::ostringstream o;
  o << "socket://" << hostname << ":" << PORT << "/";
  return o.str();
}

void SocketEpChannel::activateConnection(void* /*obj*/) {  }

Message* SocketEpChannel::getMessage() {
  if (connfd == 0)
    return NULL;
  if (msg == NULL)
    msg = new SocketMessage();
  return msg;
}

void SocketEpChannel::allocateHandlerTable(int /*size*/) { }

void SocketEpChannel::registerHandler(int /*num*/, void* /*handle*/) { }

void SocketEpChannel::bind(SpChannel* /*spchan*/) { }










