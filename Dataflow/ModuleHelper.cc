
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

#include <Dataflow/ModuleHelper.h>

#include <Comm/MessageBase.h>
#include <Comm/MessageTypes.h>
#include <Dataflow/Connection.h>
#include <Dataflow/Module.h>
#include <Dataflow/NetworkEditor.h>
#include <Dataflow/Port.h>

#include <iostream.h>

#define DEFAULT_MODULE_PRIORITY 90
ModuleHelper::ModuleHelper(Module* module)
: Task(module->name(), 1, DEFAULT_MODULE_PRIORITY),
  module(module)
{
}

ModuleHelper::~ModuleHelper()
{
}

int ModuleHelper::body(int)
{
    if(module->have_own_dispatch){
	module->do_execute();
    } else {
	for(;;){
	    MessageBase* msg=module->mailbox.receive();
	    switch(msg->type){
	    case MessageTypes::ExecuteModule:
		module->do_execute();
		break;
	    case MessageTypes::TriggerPort:
		{
		    Scheduler_Module_Message* smsg=(Scheduler_Module_Message*)msg;
		    smsg->conn->oport->resend(smsg->conn);
		}
		break;
	    case MessageTypes::Demand:
  	        {
#if 0
		    Demand_Message* dmsg=(Demand_Message*)msg;
		    if(dmsg->conn->oport->have_data()){
		        dmsg->conn->oport->resend(dmsg->conn);
		    } else {
		        dmsg->conn->demand++;
			while(dmsg->conn->demand)
			    module->do_execute();
		    }
#endif
		}
		break;
	    default:
		cerr << "Illegal Message type: " << msg->type << endl;
		break;
	    }
	    delete msg;
	}
    }
    return 0;
}
