//static char *id="@(#) $Id$";

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

#ifdef _WIN32
#pragma warning(disable:4355)
#endif

#include <PSECore/Dataflow/Module.h>

#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Connection.h>
#include <PSECore/Dataflow/ModuleHelper.h>
#include <PSECore/Dataflow/Network.h>
#include <PSECore/Dataflow/NetworkEditor.h>
#include <PSECore/Dataflow/Port.h>
#include <SCICore/Geom/GeomPick.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/Thread/Thread.h>

using SCICore::Thread::Thread;

#include <iostream>
using std::cerr;
#include <stdlib.h>

namespace PSECore {
namespace Dataflow {

using SCICore::Containers::to_string;

bool global_remote = false;

Module::Module(const clString& name, const clString& id,
	       SchedClass sched_class)
: state(NeedData), helper(0), have_own_dispatch(0),
    mailbox("Module execution FIFO", 100),
  name(name), abort_flag(0), need_execute(0), sched_class(sched_class),
  id(id), progress(0), handle(0), remote(0), skeleton(0),
  notes("notes", id, this), show_status(1)
{
  packageName="error: unset package name";
  categoryName="error: unset category name";
  moduleName="error: unset module name";

}

Module::~Module()
{
}

int Module::clone(int)
{
    ASSERTFAIL("Module::clone should not get called!\n");
    return 0;
}

void Module::update_state(State st)
{
    if (!show_status) return;
    state=st;
    char* s="unknown";
    switch(st){
    case NeedData:
	s="NeedData";
	break;
    case JustStarted:
	s="JustStarted";
	break;
    case Executing:
	s="Executing";
	break;
    case Completed:
	s="Completed";
	break;
    }
    double time=timer.time();
    TCL::execute(id+" set_state "+s+" "+to_string(time));
}

void Module::update_progress(double p)
{
    if (!show_status) return;
    if (state == JustStarted)
	update_state(Executing);
    int opp=(int)(progress*100);
    int npp=(int)(p*100);
    if(opp != npp){
	double time=timer.time();
	TCL::execute(id+" set_progress "+to_string(p)+" "+to_string(time));
	progress=p;
    }
}

void Module::update_progress(double p, Timer &t)
{
    if (!show_status) return;
    if (state == JustStarted)
	update_state(Executing);
    int opp=(int)(progress*100);
    int npp=(int)(p*100);
    if(opp != npp){
	double time=t.time();
	TCL::execute(id+" set_progress "+to_string(p)+" "+to_string(time));
	progress=p;
    }
}

void Module::update_progress(int n, int max)
{
    update_progress(double(n)/double(max));
}

void Module::update_progress(int n, int max, Timer &t)
{
    
    update_progress(double(n)/double(max), t);
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
    helper=scinew ModuleHelper(this);
    Thread* t=new Thread(helper, name());
    t->detach();
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
    need_execute=1;
    netedit->mailbox.send(scinew Module_Scheduler_Message);
}

void
Module::geom_pick(GeomPick*, Roe*, int, const BState&)
{
  NOT_FINISHED("Module::geom_pick: This version of geom_pick is only here to stop the compiler from complaining, it should never be used.");
}

void
Module::geom_pick(GeomPick* gp, void* userdata, int)
{
  geom_pick(gp, userdata);
}

void
Module::geom_pick(GeomPick*, void*)
{
    cerr << "Caught stray pick event!\n";
}

void
Module::geom_release(GeomPick*, int, const BState&)
{
  NOT_FINISHED("Module::geom_release: This version of geom_release is only here to stop the compiler from complaining, it should never be used.");
}

void Module::geom_release(GeomPick* gp, void* userdata, int)
{
  geom_release(gp, userdata);
}

void Module::geom_release(GeomPick*, void*)
{
    cerr << "Caught stray release event!\n";
}

void
Module::geom_moved(GeomPick*, int, double, const Vector&, 
		   int, const BState&)
{
  NOT_FINISHED("Module::geom_moved: This version of geom_moved is only here to stop the compiler from complaining, it should never be used.");
}

void
Module::geom_moved(GeomPick*, int, double, const Vector&, 
		   const BState&, int)
{
  NOT_FINISHED("Module::geom_moved: This version of geom_moved is only here to stop the compiler from complaining, it should never be used.");
}


void Module::geom_moved(GeomPick* gp, int which, double delta,
			const Vector& dir, void* cbdata, int)
{
  geom_moved(gp, which, delta, dir, cbdata);
}

void Module::geom_moved(GeomPick*, int, double, const Vector&, void*)
{
    cerr << "Caught stray moved event!\n";
}

void Module::widget_moved(int)
{
    cerr << "Caught stray widget_moved event!\n";
}

void Module::get_position(int& x, int& y)
{
    clString result;
    if(!TCL::eval(id+" get_x", result)){
        error("Error getting x coordinate");
	return;
    }
    if(!result.get_int(x)){
        error("Error parsing x coordinate");
	return;
    }
    if(!TCL::eval(id+" get_y", result)){
        error("Error getting y coordinate");
	return;
    }
    if(!result.get_int(y)){
        error("Error parsing y coordinate");
	return;
    }
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
	    Array1<clString> pi;
	    pi.add(port->get_colorname());
	    pi.add(to_string(port->nconnections()>0));
	    pi.add(port->get_typename());
	    pi.add(port->get_portname());
	    info[i]=args.make_list(pi);
	}
	args.result(args.make_list(info));
    } else if(args[1] == "oportinfo"){
	Array1<clString> info(oports.size());
	for(int i=0;i<oports.size();i++){
	    OPort* port=oports[i];
	    Array1<clString> pi;
	    pi.add(port->get_colorname());
	    pi.add(to_string(port->nconnections()>0));
	    pi.add(port->get_typename());
	    pi.add(port->get_portname());
	    info[i]=args.make_list(pi);
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
// ZZZ- what should I do with this on remote side?
void Module::error(const clString& string)
{
    netedit->add_text(name+": "+string);
}


#if 0
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
#endif

void Module::do_execute()
{
    abort_flag=0;
    // Reset all of the ports...
    int i;

    clString result;
    if (!TCL::eval(id+" get_show_status", result)) {
	error("Error getting show_status");
    } else if (!result.get_int(show_status)) {
	error("Error parsing show_status");
    }
//    cerr << "show_status = "<<show_status<<"\n";

    for(i=0;i<oports.size();i++){
	OPort* port=oports[i];
	port->reset();
    }
//    if (iports.size()) {
//	update_state(NeedData);
//	reset_vars();
//    }

    for(i=0;i<iports.size();i++){
	IPort* port=iports[i];
	port->reset();
    }
    // Reset the TCL variables, if not slave
    if (!global_remote)
    	reset_vars();

    // Call the User's execute function...
    update_state(JustStarted);
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
    if (global_remote)
	return;
    if(id.len()==0)
	return;
    TCL::execute("configureIPorts "+id);
}

void Module::reconfigure_oports()
{
    if (global_remote)
	return;
    else if (id.len()==0)
	return;
    TCL::execute("configureOPorts "+id);
}

void Module::multisend(OPort* p1, OPort* p2)
{
    //cerr << "Module: " << name << " called multisend on port " << p1 << endl;
    netedit->mailbox.send(new Module_Scheduler_Message(p1, p2));
}

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.8  1999/11/10 23:24:30  dmw
// added show_status flag to module interface -- if you turn it off, the timer and port lights won't update
//
// Revision 1.7  1999/10/07 02:07:19  sparker
// use standard iostreams and complex type
//
// Revision 1.6  1999/08/30 18:47:52  kuzimmer
// Modified so that dataflow scripts can be read and written properly
//
// Revision 1.5  1999/08/28 17:54:28  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/18 20:20:18  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.3  1999/08/17 06:38:22  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:57  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//
