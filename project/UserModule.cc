
/*
 *  UserModule.cc: Base class for defined modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <UserModule.h>
#include <Data.h>
#include <MUI.h>
#include <NotFinished.h>
#include <Port.h>

#define DEFAULT_MODULE_PRIORITY 90

UserModule::UserModule(const clString& name)
: Module(name), Task(name, 1, DEFAULT_MODULE_PRIORITY),
  window(0)
{
}

UserModule::~UserModule()
{
    if(window)delete window;
}

UserModule::UserModule(const UserModule& copy, int deep)
: Module(copy, deep), Task(copy)
{
    NOT_FINISHED("UserModule::UserModule");
}

void UserModule::activate()
{
    Task::activate(0);
}

extern "C" int abs(int);
int UserModule::body(int)
{
    while(1){
	ModuleMsg* msg=mailbox.receive();
	switch(msg->type){
	case ModuleMsg::Connect:
	    // Tell the data items about the connections...
	    cerr << "Connect: " << get_name() << ", output=" << msg->output << "port=" << msg->port << endl;
	    if(msg->output){
		oports[msg->port]->data->connection=msg->connection;
	    } else {
		iports[msg->port]->data->connection=msg->connection;
	    }
	    // Do the connection callback...
	    connection(msg->mode, msg->port, msg->output);
	    break;
	}
	// See if the module should be executed...
	int exec=0;
	switch(ec){
	case Always:
	    // Only if at least one output port is connected...
	    exec=1;
	    break;
	case NewDataOnAllConnectedPorts:
	    exec=1;
	    break;
	case OnOffSwitch:
	    exec=1;
	    break;
	}
	// This is a hack...
	if(oports.size() > 0 && oports[0]->data->connection==0)exec=0;
	if(iports.size() > 0 && iports[0]->data->connection==0)exec=0;
	if(exec){
	    // We should only do this for ports with new data...
	    state=Executing;
	    for(int i=0;i<iports.size();i++)
		iports[i]->data->reset();
	    for(i=0;i<oports.size();i++)
		oports[i]->data->reset();
		
	    execute();
	    for(i=0;i<iports.size();i++)
		iports[i]->data->finish();
	    for(i=0;i<oports.size();i++)
		oports[i]->data->finish();
	    state=Completed;
	    progress=1.0;
	    need_update=1;
	}
    }
    return 0;
}


// User interface stuff...
void UserModule::add_ui(MUI_widget* widget)
{
    if(!window)
	window=new MUI_window(this);
    window->attach(widget);
}

void UserModule::remove_ui(MUI_widget* widget)
{
    ASSERT(window != 0);
    window->detach(widget);
}

void UserModule::reconfigure_ui()
{
    ASSERT(window != 0);
    window->reconfigure();
}


// Port stuff
void UserModule::add_iport(InData* data, const clString& name,
			   int protocols)
{
    iports.add(new IPort(this, iports.size(), data, name));

    // Send the update message to the user interface...
    NOT_FINISHED("UserModule::add_iport");
}

void UserModule::add_oport(OutData* data, const clString& name,
			   int protocols)
{
    oports.add(new OPort(this, oports.size(), data, name));

    // Send an update message to the user interface...
    NOT_FINISHED("UserModule::add_oport");
}

void UserModule::remove_iport(int)
{
    NOT_FINISHED("UserModule::remove_iport");
}

void UserModule::remove_oport(int)
{
    NOT_FINISHED("UserModule::remove_oport");
}

void UserModule::rename_iport(int, const clString&)
{
    NOT_FINISHED("UserModule::rename_iport");
}

// Error conditions
void UserModule::error(const clString& string)
{
    cerr << string << endl;
}


// Execute conditions
void UserModule::execute_condition(CommonEC _ec)
{
    ec=_ec;
}

void UserModule::execute_condition(MUI_onoff_switch* _sw, int _swval)
{
    ec=OnOffSwitch;
    sw=_sw;
    swval=_swval;
}

void UserModule::connection(ConnectionMode, int, int)
{
    // Default - do nothing...
}
