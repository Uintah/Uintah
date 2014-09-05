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
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/ModuleHelper.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Scheduler.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/Thread/Thread.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/soloader.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stack>

using namespace std;
using namespace SCIRun;
typedef std::map<int,IPortInfo*>::iterator iport_iter;
typedef std::map<int,OPortInfo*>::iterator oport_iter;

static void *FindLibrarySymbol(const string &package, const string &/* type */, 
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

static iport_maker FindIPort(const string &package, const string &datatype)
{
  string maker_symbol = "make_" + datatype + "IPort";
  iport_maker maker =
    (iport_maker)FindLibrarySymbol(package, datatype, maker_symbol);
  return maker;
}  

static oport_maker FindOPort(const string &package, const string &datatype)
{
  string maker_symbol = "make_" + datatype + "OPort";
  oport_maker maker =
    (oport_maker)FindLibrarySymbol(package, datatype, maker_symbol);
  return maker;
}  

Module::Module(const string& name, GuiContext* ctx,
	       SchedClass sched_class, const string& cat,
	       const string& pack)
  : mailbox("Module execution FIFO", 100),
    gui(ctx->getInterface()),
    ctx(ctx),
    name(name),
    moduleName(name),
    packageName(pack),
    categoryName(cat),
    sched(0),
    pid_(0),
    have_own_dispatch(0),
    helper_done("Module helper finished flag"),
    id(ctx->getfullname()), 
    abort_flag(0),
    msgStream_(ctx->subVar("msgStream")),
    need_execute(0),
    sched_class(sched_class),
    state(NeedData),
    msg_state(Reset), 
    progress(0),
    show_stat(false),
    helper(0),
    network(0), 
    notes(ctx->subVar("notes")),
    show_status(ctx->subVar("show_status"))
{
  stacksize=0;

  IPort* iport;
  OPort* oport;
  
  first_dynamic_port = -1;
  lastportdynamic = 0;
  dynamic_port_maker = 0;

  // Auto allocate all ports listed in the .xml file for this module,
  // execpt those whose datatype tag has contents that start with '*'.
  ModuleInfo* info = packageDB->GetModuleInfo(moduleName, categoryName,
					      packageName);
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
      } else {
	cerr << "Cannot create port: " << datatype << '\n';
	dynamic_port_maker = 0;
      }
      delete[] package;
      delete[] datatype;
    }
  } else {
    cerr << "Cannot find module info for module: " << moduleName << '\n';
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
  double time = timer.time();
  if(time<0)
    time=0;
  time = Min(time, 1.0e10); // Clamp NaN
  gui->execute(id+" set_state " + s + " " + to_string(time));
}


void Module::update_msg_state(MsgState st)
{
  if (!show_stat) return;

  // only change the state if the new state
  // is of higher priority
  if( !( ((msg_state == Error) && (st != Reset))  || 
	 ((msg_state == Warning) && (st == Remark)) ) ) {
    msg_state=st;
    char* s="unknown";
    switch(st){
    case Remark:
      s="Remark";
      break;
    case Warning:
      s="Warning";
      break;
    case Error:
      s="Error";
      break;
    case Reset:
      s="Reset";
      break;
    }
    gui->execute(id+" set_msg_state " + s);
  }
}

void Module::update_progress(double p)
{
  if (!show_stat) return;
  if (state == JustStarted)
    update_state(Executing);
  if (p < 0.0) p = 0.0;
  if (p > 1.0) p = 1.0;
  int opp=(int)(progress*100);
  int npp=(int)(p*100);
  if(opp != npp){
    double time=timer.time();
    gui->execute(id+" set_progress "+to_string(p)+" "+to_string(time));
    progress=p;
  }
}

void Module::update_progress(double p, Timer &t)
{
  if (!show_stat) return;
  if (state == JustStarted)
    update_state(Executing);
  if (p < 0.0) p = 0.0;
  if (p > 1.0) p = 1.0;
  int opp=(int)(progress*100);
  int npp=(int)(p*100);
  if(opp != npp){
    double time=t.time();
    gui->execute(id+" set_progress "+to_string(p)+" "+to_string(time));
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

void Module::light_module()
{
  gui->execute(id+" light_module ");
}

void Module::reset_module_color()
{
  gui->execute(id+" reset_module_color ");
}

// Port stuff
void Module::add_iport(IPort* port)
{
  if(lastportdynamic && dynamic_port_maker) {
    gui->execute(id+" module_grow "+to_string(iports.size()));
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
      gui->execute(command);

      command = "global netedit_mini_canvas\n$netedit_mini_canvas itemconfigure " +
	omod + "_p" + op + "_to_" + imod + "_p" + to_string(loop1+1) +
	" -tags " + iports[loop1]->connection(loop2)->id;
      gui->execute(command);
      
      command = "global netedit_canvas\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <ButtonPress-3> \"destroyConnection " +
	iports[loop1]->connection(loop2)->id +
	" " + omod + " " + imod + "\"";
      gui->execute(command);

      command = "global netedit_canvas\nset temp \"a\"\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <ButtonPress-1> \"lightPipe $temp "+ omod + " " + op + " " +
	imod + " " + ip + "\"";
      gui->execute(command);

      command = "global netedit_canvas\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <ButtonRelease-1> \"resetPipe $temp " + omod + " " + imod + "\"";
      gui->execute(command);

      command = "global netedit_canvas\n$netedit_canvas bind " +
	iports[loop1]->connection(loop2)->id +
	" <Control-Button-1> \"raisePipe " +
	iports[loop1]->connection(loop2)->id + "\"";
      gui->execute(command);

    }
  }
  gui->execute(id+" module_shrink"); 
  reconfigure_iports();
}

port_range_type Module::get_iports(const string &name)
{
  return iports[name];
}

port_range_type Module::get_oports(const string &name)
{
  return oports[name];
}

IPort* Module::get_iport(int item)
{
  return iports[item];
}

OPort* Module::get_oport(int item)
{
  return oports[item];
}

IPort* Module::get_iport(const string& name)
{
  return getIPort(name);
}

OPort* Module::get_oport(const string& name)
{
  return getOPort(name);
}

IPort* Module::getIPort(const string &name)
{
  if (iports[name].first==iports[name].second) {
    return 0;
  }
  return getIPort(iports[name].first->second);
}

OPort* Module::getOPort(const string &name)
{
  if (oports[name].first==oports[name].second) {
    //postMessage("Unable to initialize "+name+"'s oports\n");
    return 0;
  }
  return getOPort(oports[name].first->second);
}

IPort* Module::getIPort(int item)
{
  return iports[item];
}

OPort* Module::getOPort(int item)
{
  return oports[item];
}

void Module::connection(Port::ConnectionState mode, int which_port, bool is_oport)
{
  if(!is_oport && lastportdynamic && dynamic_port_maker && (which_port >= first_dynamic_port)) {
    if(mode == Port::Disconnected) {
      remove_iport(which_port);
    } else {
      add_iport(dynamic_port_maker(this,lastportname));
    }
  }
  
  // do nothing by default
}

void Module::set_context(Scheduler* sched, Network* network)
{
  this->network=network;
  this->sched=sched;
  // Start up the event loop
  helper=scinew ModuleHelper(this);
  Thread* t=new Thread(helper, moduleName.c_str(), 0, Thread::NotActivated);
  if(stacksize)
    t->setStackSize(stacksize);
  t->activate(false);
  t->detach();
}

void Module::setStackSize(unsigned long s)
{
   stacksize=s;
}

void Module::want_to_execute()
{
    need_execute=true;
    sched->do_scheduling();
}

void Module::widget_moved(bool)
{
}

void Module::get_position(int& x, int& y)
{
  string result;
  if(!gui->eval(id+" get_x", result)){
    error("Error getting x coordinate.");
    return;
  }
  if(!string_to_int(result, x)) {
    error("Error parsing x coordinate.");
    return;
  }
  if(!gui->eval(id+" get_y", result)){
    error("Error getting y coordinate.");
    return;
  }
  if(!string_to_int(result, y)) {
    error("Error parsing y coordinate.");
    return;
  }
}


// Simple and limited parsing for help descriptions.
static string
parse_description(const string &in)
{
  std::stack<int> state;
  string out;
  state.push(0);
  state.push(2);
  for (unsigned int i=0; i < in.size(); i++)
  {
    char c = in[i];
    if (c == '\n' || c == '\t')
    {
      c = ' ';
    }
    if (state.top() == 0)
    {
      if (c == '<')
      {
	state.push(1);
      }
      else if (c == ' ')
      {
	out.push_back(' ');
	state.push(2);
      }
      else if (c == '.' || c == '!' || c == '?')
      {
	out.push_back(c);
	out.push_back(' ');
	out.push_back(' ');
	state.push(2);
      }
      else
      {
	out.push_back(c);
      }
    }
    else if (state.top() == 1)
    {
      if (c == '>')
      {
	if (i > 1 && in[i-2] == '/' && in[i-1] == 'p')
	{
	  out.push_back('\n');
	  out.push_back('\n');
	  state.pop();
	  state.push(2);
	}
	else
	{
	  state.pop();
	}
      }
    }
    else if (state.top() == 2)
    {
      if (c == '<')
      {
	state.push(1);
      }
      else if (c != ' ')
      {
	state.pop();
	i--;
      }
    }
  }
  return out;
}


void Module::tcl_command(GuiArgs& args, void*)
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
  } else if(args[1] == "help"){
    args.result(parse_description(description));
  } else {
    args.error("Unknown minor command for module: "+args[1]);
  }
}

void Module::setPid(int pid)
{
  pid_=pid;
}

// Error conditions
void Module::error(const string& str)
{
  //gui->postMessage("ERROR: " + moduleName + ": " + str, true);
  msgStream_ << "ERROR: " << str << '\n';
  msgStream_.flush();
  update_msg_state(Error); 
}

void Module::warning(const string& str)
{
  // gui->postMessage("WARNING: " + moduleName + ": " + str, false);
  msgStream_ << "WARNING: " << str << '\n';
  msgStream_.flush();
  update_msg_state(Warning); 
}

void Module::remark(const string& str)
{
  //gui->postMessage("REMARK: " + moduleName + ": " + str, false);
  msgStream_ << "REMARK: " << str << '\n';
  msgStream_.flush();
  update_msg_state(Remark); 
}

void Module::postMessage(const string& str)
{
  gui->postMessage(moduleName + ": " + str, false);
}

void Module::do_execute()
{
  abort_flag=0;
  show_stat=show_status.get();

  // Reset all of the ports...
  for(int i=0;i<oports.size();i++){
    OPort* port=oports[i];
    port->reset();
  }
  for(int i=0;i<iports.size();i++){
    IPort* port=iports[i];
    port->reset();
  }
  // Reset the TCL variables
  reset_vars();

  // Call the User's execute function...
  update_msg_state(Reset);
  update_state(JustStarted);
  timer.clear();
  timer.start();
  execute();
  timer.stop();
  update_state(Completed);

  // Call finish on all ports...
  for(int i=0;i<iports.size();i++){
    IPort* port=iports[i];
    port->finish();
  }
  for(int i=0;i<oports.size();i++){
    OPort* port=oports[i];
    port->finish();
  }
}

void Module::reconfigure_iports()
{
  if(id.size()==0)
    return;
  gui->execute("configureIPorts "+id);
}

void Module::reconfigure_oports()
{
  if (id.size()==0)
    return;
  gui->execute("configureOPorts "+id);
}

void Module::request_multisend(OPort* p1)
{
  sched->request_multisend(p1);
}

int Module::numOPorts()
{
  return oports.size();
}

int Module::numIPorts()
{
  return iports.size();
}

void Module::emit_vars(std::ostream& out, const std::string& modname)
{
  ctx->emit(out, modname);
}

bool Module::showStats()
{
  return show_stat;
}

GuiInterface* Module::getGui()
{
  return gui;
}

void Module::reset_vars()
{
  ctx->reset();
}

bool Module::haveUI()
{
  string result;
  if(!gui->eval(id+" have_ui", result)){
    error("Could not find UI tcl function.");
    return false;
  }
  istringstream res(result);
  int flag;
  res >> flag;
  if(!res){
    error("Could not run UI tcl function.");
    return false;
  }
  return flag == 1;
}

void Module::popupUI()
{
  gui->execute(id+" popup_ui");
}
