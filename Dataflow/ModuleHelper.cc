
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
#include <Dataflow/Module.h>

#include <iostream.h>

#define DEFAULT_MODULE_PRIORITY 90
ModuleHelper::ModuleHelper(Module* module)
: Task(module->name, 1, DEFAULT_MODULE_PRIORITY),
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
	while(1){
	    MessageBase* msg=module->mailbox.receive();
	    switch(msg->type){
	    case MessageTypes::ExecuteModule:
		module->do_execute();
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
