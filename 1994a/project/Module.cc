
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
#include <ModuleShape.h>
#include <MotifCallbackBase.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <Port.h>
#include <stdlib.h>

Module::Module(const clString& name, SchedClass sched_class)
: name(name), sched_class(sched_class), state(NeedData), mailbox(100),
  xpos(10), ypos(10), width(100), height(100), helper(0),
  sched_state(SchedDormant)
{
}

Module::Module(const Module& copy, int)
: name(copy.name), state(NeedData), mailbox(5),
  xpos(10), ypos(10), width(100), height(100), sched_state(SchedDormant)
{
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
    port->set_which_port(iports.size());
    iports.add(port);
    reconfigure_iports();
}

void Module::add_oport(OPort* port)
{
    port->set_which_port(oports.size());
    oports.add(port);
    reconfigure_oports();
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

int Module::noports()
{
    return oports.size();
}

int Module::niports()
{
    return iports.size();
}

void Module::want_to_execute()
{
    sched_state=SchedNewData;
    state=NeedData;
    netedit->mailbox.send(new Module_Scheduler_Message);
}

void Module::get_iport_coords(int which, int& x, int& y)
{
    int port_spacing=MODULE_PORTPAD_WIDTH+MODULE_PORTPAD_SPACE;
    int p2=(MODULE_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
    x=xpos+which*port_spacing+MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER+p2;
    y=ypos;
}

void Module::get_oport_coords(int which, int& x, int& y)
{
    get_iport_coords(which, x, y);
    y+=height;
}

void Module::geom_pick(void*)
{
    cerr << "Caught stray pick event!\n";
}

void Module::geom_release(void*)
{
    cerr << "Caught stray release event!\n";
}

void Module::geom_moved(int, double, const Vector&, void*)
{
    cerr << "Caught stray moved event!\n";
}

