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
 *  SocketSpChannel.h: Socket implemenation of Sp Channel
 *
 *  Written by:
 *   Kosta Damevski and Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#ifndef CORE_CCA_COMPONENT_COMM_SOCKETSPCHANNEL_H
#define CORE_CCA_COMPONENT_COMM_SOCKETSPCHANNEL_H

#include <Core/CCA/Component/Comm/SpChannel.h>

namespace SCIRun {
  using namespace std;

  struct SocketStartPoint{
    int sockfd;
    long ip;
    unsigned short port;
    int spec; //not used
    int pid;
    void *object;
    int refcnt;
  };



  class SocketSpChannel : public SpChannel {
    friend class SocketMessage;
    friend class SocketEpChannel;
  public:

    SocketSpChannel();
    ~SocketSpChannel();
    void openConnection(const URL& url);
    void closeConnection();
    Message* getMessage();
    SpChannel* SPFactory(bool deep);

  private:
    SocketSpChannel(struct SocketStartPoint *sp);
    struct SocketStartPoint *sp;
    static int cnt_c;
  };
}


#endif
