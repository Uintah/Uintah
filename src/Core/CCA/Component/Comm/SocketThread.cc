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
 *  QtUtils.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 */

#include <iostream>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <Core/CCA/Component/Comm/SocketSpChannel.h>
#include <Core/CCA/Component/Comm/SocketThread.h>
#include <Core/CCA/Component/Comm/Message.h>
#include <Core/CCA/Component/PIDL/ServerContext.h>

using namespace SCIRun;
using namespace std;

static Semaphore* startup;
  
SocketThread::SocketThread(SocketEpChannel *ep, int id, int new_fd){
  this->ep=ep;
  this->id=id;
  this->new_fd=new_fd;
}


void 
SocketThread::run()
{
  if(id==-1) ep->runAccept();
  if(id==-2) ep->runService(new_fd);
  else{
    //cerr<<"calling handler #"<<id<<"\n";
    Message *msg=ep->getMessage();
    ep->handler_table[id](msg);
    
    if(id==1){
      ::SCIRun::ServerContext* _sc=static_cast< ::SCIRun::ServerContext*>(ep->object);
      if(_sc->d_objptr->getRefCount()==0){
	cerr<<"calling accept_thread->exit()...";
	ep->accept_thread->stop();
	ep->accept_thread->exit();
	cerr<<"Done";
      }
    }
  }
}
