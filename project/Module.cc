
/*
 *  Module.cc: Basic implementation of modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Module.h>
#include <Connection.h>
#include <ModuleHelper.h>
#include <MotifCallbackBase.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <stdlib.h>

Module::Module(const clString& name, SchedClass sched_class)
: name(name), sched_class(sched_class), state(NeedData), mailbox(5),
  xpos(10), ypos(10)
{
    helper=new ModuleHelper(this);
    helper->activate(0);
}

Module::Module(const Module& copy, int)
: name(copy.name), state(NeedData), mailbox(5),
  xpos(10), ypos(10)
{
    helper=new ModuleHelper(this);
    helper->activate(0);
    NOT_FINISHED("Module copy CTOR");
}

Module::~Module()
{
}

void Module::update_progress(double p)
{
    progress=p;
    need_update=1;
}

void Module::update_progress(int n, int max)
{
    progress=double(n)/double(max);
    need_update=1;
}

// Port stuff
void Module::add_iport(IPort* port)
{
    iports.add(port);

    // Send the update message to the user interface...
    NOT_FINISHED("Module::add_iport");
}

void Module::add_oport(OPort* port)
{
    oports.add(port);

    // Send an update message to the user interface...
    NOT_FINISHED("Module::add_oport");
}

void Module::remove_iport(int)
{
    NOT_FINISHED("Module::remove_iport");
}

void Module::remove_oport(int)
{
    NOT_FINISHED("Module::remove_oport");
}

void Module::rename_iport(int, const clString&)
{
    NOT_FINISHED("Module::rename_iport");
}


void Module::connection(ConnectionMode, int, int)
{
    // Default - do nothing...
}

void Module::set_context(NetworkEditor* _netedit, Network* _network)
{
    netedit=_netedit;
    network=_network;
    create_interface();
}

OPort* Module::oport(int i)
{
    return oports[i];
}

IPort* Module::iport(int i)
{
    return iports[i];
}

void Module::want_to_execute()
{
    sched_state=SchedNewData;
    state=NeedData;
    netedit->mailbox.send(new Module_Scheduler_Message);
}

