//static char *id="@(#) $Id$";

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

#include <PSECore/Dataflow/ModuleHelper.h>

#include <PSECore/Comm/MessageBase.h>
#include <PSECore/Comm/MessageTypes.h>
#include <PSECore/Dataflow/Connection.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Dataflow/NetworkEditor.h>
#include <PSECore/Dataflow/Port.h>

#include <iostream.h>

#define DEFAULT_MODULE_PRIORITY 90

namespace PSECore {
namespace Dataflow {

ModuleHelper::ModuleHelper(Module* module)
: module(module)
{
}

ModuleHelper::~ModuleHelper()
{
}

void ModuleHelper::run()
{
  using PSECore::Comm::MessageTypes;

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
}

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/28 17:54:29  sparker
// Integrated new Thread library
//
// Revision 1.2  1999/08/17 06:38:23  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:58  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//
