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
 *  SocketThread.cc: Threads used by Socket communication channels
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#include <iostream>
#include <Core/Thread/Thread.h>
#include <Core/CCA/Comm/SocketEpChannel.h>
#include <Core/CCA/Comm/SocketThread.h>
#include <Core/CCA/Comm/Message.h>
#include <Core/CCA/PIDL/ServerContext.h>

using namespace SCIRun;
using namespace std;

  
SocketThread::SocketThread(SocketEpChannel *ep, Message* msg, int id){
  this->ep=ep;
  this->msg=msg;
  this->id=id;
}

void 
SocketThread::run()
{
  //cerr<<"calling handler #"<<id<<"\n";
  ep->handler_table[id](msg);
  //handler will call destroyMessage
}
