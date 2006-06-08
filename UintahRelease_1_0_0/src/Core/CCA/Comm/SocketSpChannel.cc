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
#include <Core/CCA/Comm/CommError.h>
#include <Core/CCA/Comm/SocketSpChannel.h>
#include <Core/CCA/Comm/SocketEpChannel.h>
#include <Core/CCA/Comm/SocketMessage.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/Object.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/ServerContext.h>
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

SocketSpChannel::SocketSpChannel(DTPoint *ep, DTAddress ep_addr) { 
  sp=new DTPoint(PIDL::getDT());
  this->ep=ep;
  this->ep_addr=ep_addr;
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

  //TODO: what if the url is not associated with a exsiting server?

  //addReference upon openning connection
  Message *message=getMessage();
  message->createMessage();
  message->sendMessage(SocketEpChannel::ADD_REFERENCE);
  message->destroyMessage();
}

SpChannel* SocketSpChannel::SPFactory(bool deep) {
  SocketSpChannel *new_sp=new SocketSpChannel(*this); 
  return new_sp;
}

void SocketSpChannel::closeConnection() {
  //delete reference upon closing connection
  Message *message=getMessage();
  message->createMessage();
  message->sendMessage(SocketEpChannel::DEL_REFERENCE); 
  message->destroyMessage();
}

//new message is created and user should call destroyMessage to delete it.
Message* SocketSpChannel::getMessage() {
  SocketMessage *msg=new SocketMessage(this);
  return msg;
}















