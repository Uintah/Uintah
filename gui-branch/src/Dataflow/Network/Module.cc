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
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Resources/Resources.h> 
#include <Dataflow/Network/Services.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/soloader.h>
#include <Core/GuiInterface/GuiManager.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {

void
postMessage( const string &msg )
{
  gm->post_msg( msg );
}

bool global_remote = false;

Module::Module(const string& name, const string& id, 
	       SchedClass sched_class, const string& cat,
	       const string& pack)
  : Part( 0, id, name ),
    notes("notes", id, this),
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
  type = pack+"_"+cat+"_"+name;
  ModuleInfo* info = resources.get_module_info(type);
  if ( info ) {
    for (unsigned i=0; i<info->oports_.size(); i++ ) {
      ModulePortInfo *pi = info->oports_[i];
      oport = services.make_oport( pi->type_, pi->name_, this );
      if ( oport )
	add_oport( oport );
    }
    
    for (unsigned i=0; i<info->iports_.size(); i++ ) {
      ModulePortInfo *pi = info->iports_[i];
      iport = services.make_iport( pi->type_, pi->name_, this );
      if ( iport )
	add_iport( iport );
    }
    lastportdynamic = info->has_dynamic_port_;
  } else {
    cerr << "ModuleInfo not found: " << type << endl;
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
  tcl_execute(id+" set_state "+s+" "+to_string(time));
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
    tcl_execute(id+" set_progress "+to_string(p)+" "+to_string(time));
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
    tcl_execute(id+" set_progress "+to_string(p)+" "+to_string(time));
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
    tcl_execute(id+" module_grow "+to_string(iports.size()));
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
      tcl_execute(command);
      
      command = "global netedit_mini_canvas\n$netedit_mini_canvas itemconfigure " +
	omod + "_p" + op + "_to_" + imod + "_p" + to_string(loop1+1) +
	" -tags " + iports[loop1]->connection(loop2)->id;
      tcl_execute(command);
      
      command = "global netedit_canvas\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <ButtonPress-3> \"destroyConnection " +
	iports[loop1]->connection(loop2)->id +
	" " + omod + " " + imod + "\"";
      tcl_execute(command);
      
      command = "global netedit_canvas\nset temp \"a\"\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <ButtonPress-1> \"lightPipe $temp "+ omod + " " + op + " " +
	imod + " " + ip + "\"";
      tcl_execute(command);
      
      command = "global netedit_canvas\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <ButtonRelease-1> \"resetPipe $temp " + omod + " " + imod + "\"";
      tcl_execute(command);
      
      command = "global netedit_canvas\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <Control-Button-1> \"raisePipe " +
	iports[loop1]->connection(loop2)->id + "\"";
      tcl_execute(command);

    }
  }
  tcl_execute(id+" module_shrink"); 
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

void Module::set_context(Scheduler* _scheduler, Network* _network)
{
  scheduler = _scheduler;
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
  scheduler->mailbox.send(scinew Module_Scheduler_Message);
}


void Module::widget_moved(int)
{
}

void Module::get_position(int& x, int& y)
{
  string result;
  tcl_eval(id+" get_x", result);
  if(!string_to_int(result, x)) {
    error("Error parsing x coordinate");
    return;
  }
  tcl_eval(id+" get_y", result);
  if(!string_to_int(result, y)) {
    error("Error parsing y coordinate");
    return;
  }

#ifdef Yarden
  if(!tcl_eval(id+" get_x", result)){
    error("Error getting x coordinate");
    return;
  }
  if(!string_to_int(result, x)) {
    error("Error parsing x coordinate");
    return;
  }
  if(!tcl_eval(id+" get_y", result)){
    error("Error getting y coordinate");
    return;
  }
  if(!string_to_int(result, y)) {
    error("Error parsing y coordinate");
    return;
  }

#endif
}

void Module::tcl_command(TCLArgs& args, void*)
{ 
  if(args.count() < 2){
    args.error("netedit needs a minor command");
    return;
  }
  if(args[1] == "iportinfo"){
    vector<string> info(iports.size());
    for(int i=0;i<iports.size();i++){
      IPort* port=iports[i];
      vector<string> pi;
      pi.push_back(port->get_colorname());
      pi.push_back(to_string(port->nconnections()>0));
      pi.push_back(port->get_typename());
      pi.push_back(port->get_portname());
      info[i]=args.make_list(pi);
    }
    args.result(args.make_list(info));
  } else if(args[1] == "oportinfo"){
    vector<string> info(oports.size());
    for(int i=0;i<oports.size();i++){
      OPort* port=oports[i];
      vector<string> pi;
      pi.push_back(port->get_colorname());
      pi.push_back(to_string(port->nconnections()>0));
      pi.push_back(port->get_typename());
      pi.push_back(port->get_portname());
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
  gm->add_text(name + ": " + str);
}

void Module::warning(const string& str)
{
  gm->add_text(name + ": " + str);
}

void Module::remark(const string& str)
{
  gm->add_text(name + ": " + str);
}



void Module::do_execute()
{
  abort_flag=0;
  // Reset all of the ports...
  int i;
  
  //    string result;
  show_stat=show_status.get();
  //    if (!tcl_eval(id+" get_show_status", result)) {
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
  tcl_execute("configureIPorts "+id);
}

void Module::reconfigure_oports()
{
  if (global_remote)
    return;
  else if (id.size()==0)
    return;
  tcl_execute("configureOPorts "+id);
}

void Module::multisend(OPort* p1, OPort* p2)
{
  //cerr << "Module: " << name << " called multisend on port " << p1 << endl;
  scheduler->mailbox.send(new Module_Scheduler_Message(p1, p2));
}



} // End namespace SCIRun
