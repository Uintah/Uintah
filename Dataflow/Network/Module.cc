/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

#include <Dataflow/Network/Module.h>

#include <Core/Util/NotFinished.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/ModuleHelper.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/PackageDB.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/soloader.h>



#include <iostream>
using std::cerr;
using std::endl;
#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {


bool global_remote = false;

extern PackageDB packageDB;

typedef std::map<int,IPortInfo*>::iterator iport_iter;
typedef std::map<int,OPortInfo*>::iterator oport_iter;

ModuleInfo* GetModuleInfo(const string& name, const string& catname,
			  const string& packname)
{
  Packages* db=(Packages*)packageDB.db_;
 
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

void *FindLibrarySymbol(const string &package, const string &/* type */, 
			const string &symbol)
{
  void* SymbolAddress = 0;
  LIBRARY_HANDLE so = 0;

  string pak_bname, cat_bname;
  if (package == "SCIRun") {
    pak_bname = "libDataflow.so";
    cat_bname = "libDataflow_Ports.so";
  } else {
    pak_bname = "libPackages_" + package + "_Dataflow.so";
    cat_bname = "libPackages_" + package + "_Dataflow_Ports.so";
  }

  // maybe it's in the small version of the .so
  so = GetLibraryHandle(cat_bname.c_str());
  if (so) {
    SymbolAddress = GetHandleSymbolAddress(so, symbol.c_str());
    if (SymbolAddress) goto found;
  }

  // maybe it's in the large version of the .so
  so = GetLibraryHandle(pak_bname.c_str());
  if (so) {
    SymbolAddress = GetHandleSymbolAddress(so, symbol.c_str());
    if (SymbolAddress) goto found;
  }

  // maybe it's in a .so that doesn't conform to the naming convention
  so = GetLibraryHandle(package.c_str());
  if (so) {
    SymbolAddress = GetHandleSymbolAddress(so,symbol.c_str());
    if (SymbolAddress) goto found;
  }

 found:
  return SymbolAddress;
}

iport_maker FindIPort(const string &package, const string &datatype)
{
  string maker_symbol = "make_" + datatype + "IPort";
  iport_maker maker =
    (iport_maker)FindLibrarySymbol(package, datatype, maker_symbol);
  return maker;
}  

oport_maker FindOPort(const string &package, const string &datatype)
{
  string maker_symbol = "make_" + datatype + "OPort";
  oport_maker maker =
    (oport_maker)FindLibrarySymbol(package, datatype, maker_symbol);
  return maker;
}  

Module::Module(const string& name, const string& id, 
	       SchedClass sched_class, const string& cat,
	       const string& pack)
  : notes("notes", id, this),
    show_status("show_status", id, this),
    msgStream_("msgStream", id, this),
    pid_(0),
    state(NeedData),
    helper(0),
    helper_done("Module helper finished flag"),
    have_own_dispatch(0),
    progress(0),
    mailbox("Module execution FIFO", 100),
    name(name),
    abort_flag(0),
    need_execute(0),
    sched_class(sched_class),
    id(id),
    handle(0),
    remote(0),
    skeleton(0)
{
  packageName=pack;
  categoryName=cat;
  moduleName=name;
  stacksize=0;

  IPort* iport;
  OPort* oport;
  
  first_dynamic_port = -1;
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
      int strlength = strlen(((*i2).second)->datatype.c_str());
      char* package = new char[strlength+1];
      char* datatype = new char[strlength+1];
      sscanf(((*i2).second)->datatype.c_str(),"%[^:]::%s",package,datatype);
      if (package[0]=='*')
	maker = FindOPort(&package[1],datatype);
      else
	maker = FindOPort(package,datatype);	
      if (maker && package[0]!='*') {
	oport = maker(this,((*i2).second)->name);
	if (oport)
	  add_oport(oport);
      }
      delete[] package;
      delete[] datatype;
    }  
    for (iport_iter i1=info->iports->begin();
	 i1!=info->iports->end();
	 i1++) {
      int strlength = strlen(((*i1).second)->datatype.c_str());
      char* package = new char[strlength+1];
      char* datatype = new char[strlength+1];
      sscanf(((*i1).second)->datatype.c_str(),"%[^:]::%s",package,datatype);
      if (package[0]=='*')
	dynamic_port_maker = FindIPort(&package[1],datatype);
      else
	dynamic_port_maker = FindIPort(package,datatype);	
      if (dynamic_port_maker && package[0]!='*') {
	iport = dynamic_port_maker(this,((*i1).second)->name);
	if (iport) {
	  lastportname = string(((*i1).second)->name);
	  add_iport(iport);
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
  if (lastportdynamic && !dynamic_port_maker){
    lastportdynamic = 0;
  }
  first_dynamic_port = iports.size()-1;
}


Module::~Module()
{
  // kill the helper thread
  MessageBase msg(MessageTypes::GoAway);
  mailbox.send(&msg);
  helper_done.receive();
}

int Module::clone(int)
{
    ASSERTFAIL("Module::clone should not get called!\n");
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
  if(lastportdynamic && dynamic_port_maker) {
    TCL::execute(id+" module_grow "+to_string(iports.size()));
  }
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


void Module::remove_iport(int which)

{
  // remove the indicated port, then
  // collapse the remaining ports together
  int loop1,loop2;
  string omod,imod,ip,op;
  string command;
  Connection *conn = 0;

  // remove (and collapse)
  iports.remove(which);  

  // rename the collapsed ports and their connections
  // to reflect the positions they collapsed to.
  for (loop1=which;loop1<iports.size();loop1++) {

    iports[loop1]->set_which_port(iports[loop1]->get_which_port()-1);

    for (loop2=0;loop2<iports[loop1]->nconnections();loop2++) {

      conn = iports[loop1]->connection(loop2);
      omod = conn->oport->get_module()->id;
      op = to_string(conn->oport->get_which_port());
      imod = conn->iport->get_module()->id;
      ip = to_string(conn->iport->get_which_port());

      iports[loop1]->connection(loop2)->id = 
	omod+"_p"+op+"_to_"+imod+"_p"+ip;

      command = "global netedit_canvas\n$netedit_canvas itemconfigure " +
	omod + "_p" + op + "_to_" + imod + "_p" + to_string(loop1+1) +
	" -tags " + iports[loop1]->connection(loop2)->id;
      TCL::execute(command);

      command = "global netedit_mini_canvas\n$netedit_mini_canvas itemconfigure " +
	omod + "_p" + op + "_to_" + imod + "_p" + to_string(loop1+1) +
	" -tags " + iports[loop1]->connection(loop2)->id;
      TCL::execute(command);
      
      command = "global netedit_canvas\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <ButtonPress-3> \"destroyConnection " +
	iports[loop1]->connection(loop2)->id +
	" " + omod + " " + imod + "\"";
      TCL::execute(command);

      command = "global netedit_canvas\nset temp \"a\"\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <ButtonPress-1> \"lightPipe $temp "+ omod + " " + op + " " +
	imod + " " + ip + "\"";
      TCL::execute(command);

      command = "global netedit_canvas\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <ButtonRelease-1> \"resetPipe $temp " + omod + " " + imod + "\"";
      TCL::execute(command);

      command = "global netedit_canvas\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <Control-Button-1> \"raisePipe " +
	iports[loop1]->connection(loop2)->id + "\"";
      TCL::execute(command);

    }
  }
  TCL::execute(id+" module_shrink"); 
  reconfigure_iports();
}

void Module::remove_oport(int)
{
    NOT_FINISHED("Module::remove_oport");
}

void Module::rename_iport(int, const string&)
{
    NOT_FINISHED("Module::rename_iport");
}


IPort *Module::get_iport(const string &name)
{
  if (get_iports(name).first==get_iports(name).second) {
    //postMessage("Unable to initialize "+name+"'s iports\n");
    return 0;
  }
  return get_iport(get_iports(name).first->second);
}

OPort *Module::get_oport(const string &name)
{
  if (get_oports(name).first==get_oports(name).second) {
    //postMessage("Unable to initialize "+name+"'s oports\n");
    return 0;
  }
  return get_oport(get_oports(name).first->second);
}

void Module::connection(ConnectionMode mode, int which_port, int is_oport)
{
  if(!is_oport && lastportdynamic && dynamic_port_maker && (which_port >= first_dynamic_port)) {
    if(mode == Disconnected) {
      remove_iport(which_port);
    } else {
      add_iport(dynamic_port_maker(this,lastportname));
    }
  }
  
  // do nothing by default
}

void Module::set_context(NetworkEditor* _netedit, Network* _network)
{
    netedit=_netedit;
    network=_network;
    
    // Start up the event loop
    helper=scinew ModuleHelper(this);
    Thread* t=new Thread(helper, name.c_str(), 0, Thread::NotActivated);
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

#if 0
void
Module::geom_pick(GeomPick*, ViewWindow*, int, const BState&)
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
#endif

void Module::widget_moved(int)
{
}

void Module::get_position(int& x, int& y)
{
    string result;
    if(!TCL::eval(id+" get_x", result)){
        error("Error getting x coordinate");
	return;
    }
    if(!string_to_int(result, x)) {
        error("Error parsing x coordinate");
	return;
    }
    if(!TCL::eval(id+" get_y", result)){
        error("Error getting y coordinate");
	return;
    }
    if(!string_to_int(result, y)) {
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
	Array1<string> info(iports.size());
	for(int i=0;i<iports.size();i++){
	    IPort* port=iports[i];
	    Array1<string> pi;
	    pi.add(port->get_colorname());
	    pi.add(to_string(port->nconnections()>0));
	    pi.add(port->get_typename());
	    pi.add(port->get_portname());
	    info[i]=args.make_list(pi);
	}
	args.result(args.make_list(info));
    } else if(args[1] == "oportinfo"){
	Array1<string> info(oports.size());
	for(int i=0;i<oports.size();i++){
	    OPort* port=oports[i];
	    Array1<string> pi;
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
    } else if(args[1] == "getpid"){
      args.result(to_string(pid_));
    } else {
	args.error("Unknown minor command for module: "+args[1]);
    }
}

// Error conditions
// ZZZ- what should I do with this on remote side?
void Module::error(const string& str)
{
    netedit->add_text(name + ": " + str);
}

void Module::warning(const string& str)
{
    netedit->add_text(name + ": " + str);
}

void Module::remark(const string& str)
{
    netedit->add_text(name + ": " + str);
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

//    string result;
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
    if(id.size()==0)
	return;
    TCL::execute("configureIPorts "+id);
}

void Module::reconfigure_oports()
{
    if (global_remote)
	return;
    else if (id.size()==0)
	return;
    TCL::execute("configureOPorts "+id);
}

void Module::multisend(OPort* p1, OPort* p2)
{
    //cerr << "Module: " << name << " called multisend on port " << p1 << endl;
    netedit->mailbox.send(new Module_Scheduler_Message(p1, p2));
}



} // End namespace SCIRun
