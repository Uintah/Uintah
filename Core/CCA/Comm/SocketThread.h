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
 *  SocketThread.h: Threads used by Socket communication channels
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CORE_CCA_COMM_SOCKETTHREAD_H
#define CORE_CCA_COMM_SOCKETTHREAD_H

#include <Core/Thread/Runnable.h>

namespace SCIRun{
  class SocketEpChannel;
  class SocketSpChannel;
  class Message;

  class SocketThread : public Runnable{
  public:
    SocketThread(SocketEpChannel *ep, Message *msg, int id);
    //id>=0, handlers
    //id==-1, service
    
    ~SocketThread() {}
    void run();
  private:
    SocketEpChannel *ep;
    Message *msg;
    int id;
  };
} // namespace SCIRun
  
#endif  

