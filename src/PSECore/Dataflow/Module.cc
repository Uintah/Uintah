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
#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Geom/GeomPick.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Util/soloader.h>

using SCICore::Thread::Thread;

#include <iostream>
using std::cerr;
using std::endl;
#include <stdlib.h>
#include <stdio.h>

namespace PSECore {
namespace Dataflow {

using SCICore::Containers::to_string;

bool global_remote = false;

extern PackageDB packageDB;

typedef std::map<int,IPortInfo*>::iterator iport_iter;
typedef std::map<int,OPortInfo*>::iterator oport_iter;
typedef std::map<clString,IPort*>::iterator auto_iport_iter;
typedef std::map<clString,OPort*>::iterator auto_oport_iter;

ModuleInfo* GetModuleInfo(const clString& name, const clString& catname,
			  const clString& packname)
{
  Packages* db=(Packages*)packageDB.d_db;
 
  Package* package;
  if (!db->lookup(packname,package))
    return 0;

  Category* category;
  if (!package->lookup(catname,category))
    return 0;

  ModuleInfo* info;
  if (category->lookup(name,info))
    return info;
  return 0;
}

void *FindLibrarySymbol(const char* package, const char* type, 
			const char* symbol)
{
  char* libname = new char[strlen(package)+strlen(symbol)+15];
  void* SymbolAddress = 0;
  LIBRARY_HANDLE so = 0;

  sprintf(libname,"lib%s_Datatypes_%s.so",package,type);
  so = GetLibraryHandle(libname);
  if (so) {
    SymbolAddress = GetHandleSymbolAddress(so,symbol);
    if (SymbolAddress) goto found;
  }
  
  sprintf(libname,"lib%s_Datatypes.so",package);
  so = GetLibraryHandle(libname);
  if (so) {
    SymbolAddress = GetHandleSymbolAddress(so,symbol);
    if (SymbolAddress) goto found;
  }

  sprintf(libname,"lib%s.so",package);
  so = GetLibraryHandle(libname);
  if (so) {
    SymbolAddress = GetHandleSymbolAddress(so,symbol);
    if (SymbolAddress) goto found;
  }

  sprintf(libname,"%s",package);
  so = GetLibraryHandle(libname);
  if (so) {
    SymbolAddress = GetHandleSymbolAddress(so,symbol);
    if (SymbolAddress) goto found;
  }

 found:
  delete[] libname;
  return SymbolAddress;
}

iport_maker FindIPort(const char* package, const char* datatype)
{
  iport_maker maker = 0;
  char* maker_symbol = new char[strlen(datatype)+11];

  sprintf(maker_symbol,"make_%sIPort",datatype);

  maker = (iport_maker)FindLibrarySymbol(package,datatype,maker_symbol);

  delete[] maker_symbol;

  return maker;
}  

oport_maker FindOPort(const char* package, const char* datatype)
{
  oport_maker maker = 0;
  char* maker_symbol = new char[strlen(datatype)+11];

  sprintf(maker_symbol,"make_%sOPort",datatype);

  maker = (oport_maker)FindLibrarySymbol(package,datatype,maker_symbol);

  delete[] maker_symbol;

  return maker;
}  

Module::Module(const clString& name, const clString& id, 
	       SchedClass sched_class, const clString& cat,
	       const clString& pack)
: state(NeedData), helper(0), have_own_dispatch(0),
    mailbox("Module execution FIFO", 100),
  name(name), abort_flag(0), need_execute(0), sched_class(sched_class),
  id(id), progress(0), handle(0), remote(0), skeleton(0),
  notes("notes", id, this), show_status("show_status", id, this)
{
  packageName=pack;
  categoryName=cat;
  moduleName=name;
  stacksize=0;

  IPort* iport;
  OPort* oport;

  lastportdynamic = 0;
  dynamic_port_maker = 0;

  // Auto allocate all ports listed in the .xml file for this module,
  // execpt those whose datatype tag has contents that start with '*'.
  ModuleInfo* info = GetModuleInfo(moduleName,categoryName,packageName);
  if (info) {
    oport_maker maker;
    for (oport_iter i2=info->oports->begin();
	 i2!=info->oports->end();
	 i2++) {
      int strlength = strlen(((*i2).second)->datatype());
      char* package = new char[strlength+1];
      char* datatype = new char[strlength+1];
      sscanf(((*i2).second)->datatype(),"%[^:]::%s",package,datatype);
      if (package[0]=='*')
	maker = FindOPort(&package[1],datatype);
      else
	maker = FindOPort(package,datatype);	
      if (maker && package[0]!='*') {
	oport = maker(this,((*i2).second)->name);
	if (oport) {
	  add_oport(oport);
	  auto_oports.insert(std::pair<clString,
			     OPort*>(((*i2).second)->name,oport));
	}
      }
      delete[] package;
      delete[] datatype;
    }  
    for (iport_iter i1=info->iports->begin();
	 i1!=info->iports->end();
	 i1++) {
      int strlength = strlen(((*i1).second)->datatype());
      char* package = new char[strlength+1];
      char* datatype = new char[strlength+1];
      sscanf(((*i1).second)->datatype(),"%[^:]::%s",package,datatype);
      if (package[0]=='*')
	dynamic_port_maker = FindIPort(&package[1],datatype);
      else
	dynamic_port_maker = FindIPort(package,datatype);	
      if (dynamic_port_maker && package[0]!='*') {
	iport = dynamic_port_maker(this,((*i1).second)->name);
	if (iport) {
	  add_iport(iport);
	  auto_iports.insert(std::pair<clString,
			     IPort*>(((*i1).second)->name,iport));
	}
      } else
	dynamic_port_maker = 0;
      delete[] package;
      delete[] datatype;
    }
  }

  // the last port listed in the .xml file may or may not be dynamic.
  // if found and lastportdynamic is true, the port is dynamic.
  // otherwise it is not dynamic.
  if (lastportdynamic && !dynamic_port_maker)
    lastportdynamic = 0;
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
    if (!show_stat) return;
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
    if (!show_stat) return;
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
    if (!show_stat) return;
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

IPort* Module::get_iport(char* portname)
{
  auto_iport_iter i;
  i = auto_iports.find(clString(portname));
  if (i!=auto_iports.end())
    return (*i).second;
  return 0;
}

OPort* Module::get_oport(char* portname)
{
  auto_oport_iter i;
  i = auto_oports.find(clString(portname));
  if (i!=auto_oports.end())
    return (*i).second;
  return 0;
}

void Module::connection(ConnectionMode mode, int which, int)
{
  // do nothing by default
}

void Module::set_context(NetworkEditor* _netedit, Network* _network)
{
    netedit=_netedit;
    network=_network;

    // Start up the event loop
    helper=scinew ModuleHelper(this);
    Thread* t=new Thread(helper, name(), 0, Thread::NotActivated);
    if(stacksize)
       t->setStackSize(stacksize);
    t->activate(false);
    t->detach();
}

void Module::setStackSize(unsigned long s)
{
   stacksize=s;
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
//Module::geom_pick(GeomPick* gp, void* userdata, int)
Module::geom_pick(GeomPick* gp, void* userdata, GeomObj*)
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

//void Module::geom_release(GeomPick* gp, void* userdata, int)
void Module::geom_release(GeomPick* gp, void* userdata, GeomObj*)
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
			//const Vector& dir, void* cbdata, int)
			const Vector& dir, void* cbdata, GeomObj*)
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

//    clString result;
    show_stat=show_status.get();
//    if (!TCL::eval(id+" get_show_status", result)) {
//	error("Error getting show_status");
//    } else if (!result.get_int(show_status)) {
//	error("Error parsing show_status");
//    }
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
// Revision 1.17  2000/12/05 19:08:29  moulding
// added support for dynamic ports to the auto port facility, although it is not
// yet fully operational.
//
// Revision 1.16  2000/11/29 09:46:49  moulding
// removed a debug print statement
//
// Revision 1.15  2000/11/29 09:37:08  moulding
// fixed a nasty bug that caused certain machine types to catch fire when
// using auto ports (nah - it just caused crashes :)
//
// Revision 1.14  2000/11/22 19:10:38  moulding
// auto-port facility is now operational.
//
// Revision 1.13  2000/11/21 22:44:30  moulding
// initial commit of auto-port facility (not yet operational).
//
// Revision 1.12  2000/08/11 15:44:43  bigler
// Changed geom_* functions that took an int index to take a GeomObj* picked_obj.
//
// Revision 1.11  2000/07/27 05:22:34  sparker
// Added a setStackSize method to the module
//
// Revision 1.10  1999/12/07 02:53:33  dmw
// made show_status variable persistent with network maps
//
// Revision 1.9  1999/11/12 01:38:30  ikits
// Added ANL AVTC site visit modifications to make the demos work.
// Fixed bugs in PSECore/Datatypes/SoundPort.[h,cc] and PSECore/Dataflow/NetworkEditor.cc
// Put in temporary scale_changed fix into PSECore/Widgets/BaseWidget.cc
//
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
