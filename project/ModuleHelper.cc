
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

#include <ModuleHelper.h>
#include <Module.h>
#include <MessageTypes.h>
#include <MUI.h>
#include <UserModule.h>
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
    while(1){
	MessageBase* msg=module->mailbox.receive();
	switch(msg->type){
	case MessageTypes::ExecuteModule:
	    module->do_execute();
	    break;
	case MessageTypes::MUIDispatch:
	    {
		MUI_Module_Message* dmsg=(MUI_Module_Message*)msg;
		dmsg->do_it();
		dmsg->module->mui_callback(dmsg->cbdata, dmsg->flags);
	    }
	    break;
	default:
	    cerr << "Illegal Message type: " << msg->type << endl;
	    break;
	}
	delete msg;
    }
    return 0;
}
