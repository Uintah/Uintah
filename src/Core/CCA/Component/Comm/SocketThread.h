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


#ifndef SOCKET_THREAD_H
#define SOCKET_THREAD_H 

#include <Core/Thread/Runnable.h>

namespace SCIRun{
  class SocketEpChannel;
  class SocketSpChannel;
  class Message;

  class SocketThread : public Runnable{
  public:
    SocketThread(SocketEpChannel *ep, Message *msg, int id, int new_fd=-1);
    //id>=0, handlers
    //id==-1, accept
    //id==-2, service...
    
    ~SocketThread() {}
    void run();
  private:
    SocketEpChannel *ep;
    Message *msg;
    int id;
    int new_fd;
  };
} // namespace SCIRun
  
#endif  

