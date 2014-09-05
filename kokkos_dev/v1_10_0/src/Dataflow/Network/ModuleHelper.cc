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
 *  ModuleHelper.cc:  Thread to execute modules..
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/ModuleHelper.h>

#include <Dataflow/Comm/MessageBase.h>
#include <Dataflow/Comm/MessageTypes.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Network/Port.h>

#include <iostream>
#include <unistd.h>
using namespace std;

#define DEFAULT_MODULE_PRIORITY 90

namespace SCIRun {

ModuleHelper::ModuleHelper(Module* module)
: module(module)
{
}

ModuleHelper::~ModuleHelper()
{
}

void ModuleHelper::run()
{
  module->setPid(getpid());
  if(module->have_own_dispatch)
    module->do_execute();
  else {
    for(;;){
      MessageBase* msg=module->mailbox.receive();
      switch(msg->type){
      case MessageTypes::GoAway:
	return;
      case MessageTypes::ExecuteModule:
	module->do_execute();
	break;
      case MessageTypes::TriggerPort:
	{
	  Scheduler_Module_Message* smsg=(Scheduler_Module_Message*)msg;
	  smsg->conn->oport->resend(smsg->conn);
	}
	break;
      default:
	cerr << "Illegal Message type: " << msg->type << std::endl;
	break;
      }
      delete msg;
    }
  }
}

} // End namespace SCIRun

