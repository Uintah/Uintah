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
 *  Scheduler.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCIRun_Dataflow_Network_Scheduler_h
#define SCIRun_Dataflow_Network_Scheduler_h

#include <Dataflow/share/share.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/Runnable.h>
#include <string>

namespace SCIRun {
  using std::string;

  class Connection;
  class MessageBase;
  class Module;
  class Network;
  class OPort;

  class PSECORESHARE Scheduler : public Runnable  {
    Network* net;
    void multisend(OPort*);
    void do_scheduling(Module*);
    bool first_schedule;
    bool schedule;
  public:
    Mailbox<MessageBase*> mailbox;

    Scheduler(Network*);
    ~Scheduler();

    void do_scheduling();
    void request_multisend(OPort*);
  private:
    virtual void run();
    void main_loop();

  };

  class PSECORESHARE Scheduler_Module_Message : public MessageBase {
  public:
    Connection* conn;
    Scheduler_Module_Message();
    Scheduler_Module_Message(Connection* conn);
    virtual ~Scheduler_Module_Message();
  };

  class PSECORESHARE Module_Scheduler_Message : public MessageBase {
  public:
    OPort* p1;
    Module_Scheduler_Message();
    Module_Scheduler_Message(OPort*);
    virtual ~Module_Scheduler_Message();
  };

} // End namespace SCIRun

#endif

