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
 *  SocketSpChannel.cc: Socket implemenation of Sp Channel
 *
 *  Written by:
 *   Kosta Damevski and Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */



#include <iostream>
#include <sstream>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <Core/CCA/Component/Comm/CommError.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <Core/CCA/Component/Comm/SocketMessage.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/ServerContext.h>
#include <Core/Thread/Thread.h>


using namespace std;
using namespace SCIRun;

SocketSpChannel::SocketSpChannel() { 
  sp=new DTPoint(PIDL::getDT());
  ep=NULL;
}

SocketSpChannel::SocketSpChannel(SocketSpChannel &spchan) { 
  sp=new DTPoint(PIDL::getDT());
  ep=spchan.ep;
  ep_addr=spchan.ep_addr;
}

SocketSpChannel::~SocketSpChannel(){
  delete sp;
}

void SocketSpChannel::openConnection(const URL& url) {
  struct hostent *he;
  // get the host info 
  if((he=gethostbyname(url.getHostname().c_str())) == NULL){
    throw CommError("gethostbyname", errno);
  }
  ep_addr.ip=((struct in_addr *)he->h_addr)->s_addr;
  ep_addr.port=url.getPortNumber();

  ep=(DTPoint*)(atol(  url.getSpec().c_str() ) );

  //addReference upon openning connection
  Message *message=getMessage();
  message->createMessage();
  message->sendMessage(-101); //addReference;
  //message->destroyMessage();
}

SpChannel* SocketSpChannel::SPFactory(bool deep) {
  SocketSpChannel *new_sp=new SocketSpChannel(*this); 
  return new_sp;
}

void SocketSpChannel::closeConnection() {
  //addReference upon openning connection
  Message *message=getMessage();
  message->createMessage();
  message->sendMessage(-102); //deleteReference;  
  //message->destroyMessage();
}

//new message is created and user should call destroyMessage to delete it.
Message* SocketSpChannel::getMessage() {
  SocketMessage *msg=new SocketMessage(this);
  return msg;
}















