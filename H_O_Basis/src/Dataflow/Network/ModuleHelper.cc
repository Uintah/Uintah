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


void
ModuleHelper::run()
{
  module->setPid(getpid());
  if(module->have_own_dispatch)
  {
    module->do_execute();
  }
  else
  {
    for(;;)
    {
      MessageBase* msg = module->mailbox.receive();
      switch(msg->type) {
      case MessageTypes::GoAway:
	delete msg;
	return;

      case MessageTypes::GoAwayWarn:
	break;

      case MessageTypes::ExecuteModule:
        module->do_execute();
        module->do_synchronize();
        module->sched->report_execution_finished(msg);
	break;

      case MessageTypes::SynchronizeModule:
        module->do_synchronize();
        module->sched->report_execution_finished(msg);
        break;

      case MessageTypes::TriggerPort:
	{
	  Scheduler_Module_Message *smsg = (Scheduler_Module_Message*)msg;
	  smsg->conn->oport->resend(smsg->conn);
	}
	break;

      default:
	cerr << "(ModuleHelper.cc) Illegal Message type: " << msg->type
             << std::endl;
	break;
      }

      delete msg;
    }
  }
}


} // End namespace SCIRun

