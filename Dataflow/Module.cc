
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

#include <Dataflow/Module.h>

#include <Classlib/NotFinished.h>
#include <Dataflow/Connection.h>
#include <Dataflow/ModuleHelper.h>
#include <Dataflow/Network.h>
#include <Dataflow/NetworkEditor.h>
#include <Dataflow/Port.h>
#include <TCL/TCL.h>

#include <stdlib.h>

Module::Module(const clString& name, const clString& id,
	       SchedClass sched_class)
: name(name), id(id), sched_class(sched_class), state(NeedData), mailbox(100),
  helper(0), sched_state(SchedDormant), have_own_dispatch(0), abort_flag(0)
{
}

Module::Module(const Module& copy, int)
: name(copy.name), id(copy.id), state(NeedData), mailbox(5),
 sched_state(SchedDormant), have_own_dispatch(0), abort_flag(0)
{
    NOT_FINISHED("Module copy CTOR");
}

Module::~Module()
{
}

void Module::update_state(State st)
{
    state=st;
    char* s="unknown";
    switch(st){
    case NeedData:
	s="NeedData";
	break;
    case Executing:
	s="Executing";
	break;
    case Completed:
	s="Completed";
	break;
    }
    TCL::execute("updateState "+id+" "+s);
}

void Module::update_progress(double p)
{
    TCL::execute("updateProgress "+id+" "+to_string(p));
    progress=p;
}

void Module::update_progress(int n, int max)
{
    update_progress(double(n)/double(max));
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

    // Start up the event loop
    helper=new ModuleHelper(this);
    helper->activate(0);
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

void Module::tcl_command(TCLArgs& args, void*)
{ 
    if(args.count() < 2){
	args.error("netedit needs a minor command");
	return;
    }
    if(args[1] == "iportinfo"){
	Array1<clString> info(iports.size());
	for(int i=0;i<iports.size();i++){
	    IPort* port=iports[i];
	    info[i]=args.make_list(port->get_colorname(),
				   to_string(port->nconnections()>0));
	}
	args.result(args.make_list(info));
    } else if(args[1] == "oportinfo"){
	Array1<clString> info(oports.size());
	for(int i=0;i<oports.size();i++){
	    OPort* port=oports[i];
	    info[i]=args.make_list(port->get_colorname(),
				   to_string(port->nconnections()>0));
	}
	args.result(args.make_list(info));
    } else if(args[1] == "needexecute"){
	if(!abort_flag){
	    abort_flag=1;
	    want_to_execute();
	}
    } else {
	args.error("Unknown minor command for module: "+args[1]);
    }
}

// Error conditions
void Module::error(const clString& string)
{
    netedit->add_text(name+": "+string);
}


int Module::should_execute()
{
    if(sched_state == SchedNewData)
	return 0; // Already maxed out...
    int changed=0;
    if(sched_class != Sink){
	// See if any outputs are connected...
	int have_outputs=0;
	for(int i=0;i<oports.size();i++){
	    if(oports[i]->nconnections() > 0){
		have_outputs=1;
		break;
	    }
	}
	if(!have_outputs)cerr << "Not executing - not hooked up...\n";
	if(!have_outputs)return 0; // Don't bother checking stuff...
    }
    if(sched_state == SchedDormant){
	// See if we should be in the regen state
	for(int i=0;i<oports.size();i++){
	    OPort* port=oports[i];
	    for(int c=0;c<port->nconnections();c++){
		Module* mod=port->connection(c)->iport->get_module();
		if(mod->sched_state != SchedNewData
		   && mod->sched_state != SchedRegenData){
		    sched_state=SchedRegenData;
		    changed=1;
		    break;
		}
	    }
	}
    }

    // See if there is new data upstream...
    if(sched_class != Source){
	for(int i=0;i<iports.size();i++){
	    IPort* port=iports[i];
	    for(int c=0;c<port->nconnections();c++){
		Module* mod=port->connection(c)->oport->get_module();
		if(mod->sched_state != SchedNewData){
		    sched_state=SchedNewData;
		    changed=1;
		    break;
		}
	    }
	}
    }
    return changed;
}

void Module::do_execute()
{
    abort_flag=0;
    // Reset all of the ports...
    for(int i=0;i<oports.size();i++){
	OPort* port=oports[i];
	port->reset();
    }
    for(i=0;i<iports.size();i++){
	IPort* port=iports[i];
	port->reset();
    }
    // Reset the TCL variables...
    reset_vars();

    // Call the User's execute function...
    update_state(Executing);
    timer.clear();
    timer.start();
    execute();
    timer.stop();
    update_state(Completed);

    // Call finish on all ports...
    for(i=0;i<iports.size();i++){
	IPort* port=iports[i];
	port->finish();
    }
    for(i=0;i<oports.size();i++){
	OPort* port=oports[i];
	port->finish();
    }
}

void Module::reconfigure_iports()
{
    if(id.len()==0)
	return;
    TCL::execute("configureOPorts "+id);
}

void Module::reconfigure_oports()
{
    if(id.len()==0)
	return;
    TCL::execute("configureOPorts "+id);
}

