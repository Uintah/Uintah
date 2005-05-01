/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Util/Environment.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stack>

using namespace std;
using namespace SCIRun;
typedef std::map<int,IPortInfo*>::iterator iport_iter;
typedef std::map<int,OPortInfo*>::iterator oport_iter;

#ifdef __APPLE__
const string ext = ".dylib";
#elif defined(_WIN32)
const string ext = ".dll";
#else
const string ext = ".so";
#endif

static void *FindLibrarySymbol(const string &package, const string &/* type */, 
			const string &symbol)
{
  void* SymbolAddress = 0;
  LIBRARY_HANDLE so = 0;

  string pak_bname, cat_bname;
  if (package == "SCIRun") {
    pak_bname = "libDataflow" + ext;
    cat_bname = "libDataflow_Ports" + ext;
  } else {
    pak_bname =  "libPackages_" + package + "_Dataflow" + ext;
    cat_bname = "libPackages_" + package + "_Dataflow_Ports" + ext;
  }

  // maybe it's in the small version of the shared library
  so = GetLibraryHandle(cat_bname.c_str());
  if (so) {
    SymbolAddress = GetHandleSymbolAddress(so, symbol.c_str());
    if (SymbolAddress) goto found;
  }

  // maybe it's in the large version of the shared library
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
    id(ctx->getfullname()), 
    abort_flag(0),
    need_execute(0),
    sched_class(sched_class),
    state(NeedData),
    msg_state(Reset), 
    progress(0),
    helper(0),
    helper_thread(0),
    network(0), 
    show_stats_(true),
    log_string_(ctx->subVar("log_string", false))
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
      unsigned long strlength = ((*i2).second)->datatype.length();
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
      unsigned long strlength = ((*i1).second)->datatype.length();
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
    cerr << "Cannot find module info for module: " 
	 << packageName << "/" << categoryName << "/" << moduleName << '\n';
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
  // kill_helper joins the helper thread & waits until module execution is done
  kill_helper();
  // After execution is done, delete the TCL command: $this-c
  gui->delete_command(id+"-c" );
  // Remove the dummy proc $this that was eating up all the TCL commands
  // while the module finished up executing
  gui->eval("catch {rename "+id+" \"\"}");
  
}

int Module::addOPortByName(std::string name, std::string d_type)
{
  OPort* oport;
  dynamic_port_maker = 0;

  oport_maker maker;

  unsigned long strlength = d_type.length();
  char* package = new char[strlength+1];
  char* datatype = new char[strlength+1];
  sscanf(d_type.c_str(),"%[^:]::%s",package,datatype);
  if (package[0]=='*')
    maker = FindOPort(&package[1],datatype);
  else
    maker = FindOPort(package,datatype);	
  if (maker && package[0]!='*') {
    oport = maker(this,name);
    if (oport)
      add_oport(oport);
    else {
      cerr << "Cannot create port: " << datatype << '\n';
      dynamic_port_maker = 0;
    }
  }
  delete[] package;
  delete[] datatype; 

  return 0;
}

int Module::addIPortByName(std::string name, std::string d_type)
{
  IPort* iport;
  dynamic_port_maker = 0;

  unsigned long strlength = d_type.length();
  char* package = new char[strlength+1];
  char* datatype = new char[strlength+1];
  sscanf(d_type.c_str(),"%[^:]::%s",package,datatype);
  if (package[0]=='*')
    dynamic_port_maker = FindIPort(&package[1],datatype);
  else
    dynamic_port_maker = FindIPort(package,datatype);	
  if (dynamic_port_maker && package[0]!='*') {
    iport = dynamic_port_maker(this,name);
    if (iport) {
      lastportname = name;
      add_iport(iport);
    }
  } else {
    cerr << "Err: Cannot create port: " << datatype << '\n';
    dynamic_port_maker = 0;
  }
  delete[] package;
  delete[] datatype;
  
  return 0;
}

void Module::delete_warn() 
{
  set_show_stats(false);
  MessageBase *msg = scinew MessageBase(MessageTypes::GoAwayWarn);
  mailbox.send(msg);
}

void Module::kill_helper()
{
  if (helper_thread)
  {
    // kill the helper thread
    MessageBase *msg = scinew MessageBase(MessageTypes::GoAway);
    mailbox.send(msg);
    helper_thread->join();
    helper_thread = 0;
  }
}


void Module::report_progress( ProgressState state )
{
  switch ( state ) {
  case ProgressReporter::Starting:
    break;
  case ProgressReporter::Compiling:
    gui->execute(id+" set_compiling_p 1");
    remark("Dynamically compiling some code.");
    break;
  case ProgressReporter::CompilationDone:
    gui->execute(id+" set_compiling_p 0");
    remark("Dynamic compilation completed.");
    break;
  case ProgressReporter::Done:
    break;
  }
}

void Module::update_state(State st)
{
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

  if (sci_getenv_p("SCI_REGRESSION_TESTING") && st == Completed)
  {
    cout << id << ":RUNTIME: " << time << "\n";
  }
}


void Module::update_msg_state(MsgState st)
{
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


void Module::update_progress(unsigned int n, unsigned int max)
{
  update_progress(double(n)/double(max));
}


// Port stuff
void Module::add_iport(IPort* port)
{
  port->set_which_port(iports.size());
  iports.add(port);
  gui->execute("drawPorts "+id+" i");
}

void Module::add_oport(OPort* port)
{
  port->set_which_port(oports.size());
  oports.add(port);
  gui->execute("drawPorts "+id+" o");
}

void Module::remove_iport(int which)
{
  // remove the indicated port, then
  // collapse the remaining ports together
  iports.remove(which);  
  gui->execute("removePort {"+id+" "+to_string(which)+" i}");
  // rename the collapsed ports and their connections
  // to reflect the positions they collapsed to.
  for (int port=which;port<iports.size();port++) {
    iports[port]->set_which_port(iports[port]->get_which_port()-1);
    for (int connNum=0;connNum<iports[port]->nconnections();connNum++)
      iports[port]->connection(connNum)->makeID();
  }
  gui->execute("drawPorts "+id+" i");
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
  IPort *p = getIPort(name);
  if (p == 0) throw "Unable to initialize iport '" + name + "'.";
  return p;
}

OPort* Module::get_oport(const string& name)
{
  OPort *p = getOPort(name);
  if (p == 0) throw "Unable to initialize oport '" + name + "'.";
  return p;
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
  if(!is_oport && lastportdynamic && 
     dynamic_port_maker && (which_port >= first_dynamic_port)) {
    if(mode == Port::Disconnected) {
      remove_iport(which_port);
    } else if (which_port == iports.size()-1) {
      add_iport(dynamic_port_maker(this,lastportname));
    }
  }
  
  // do nothing by default
}

void Module::set_context(Scheduler* sched, Network* network)
{
  this->network=network;
  this->sched=sched;
  ASSERT(helper == 0);
  // Start up the event loop
  helper=scinew ModuleHelper(this);
  helper_thread = 
    scinew Thread(helper, moduleName.c_str(), 0, Thread::NotActivated);
  if(stacksize)
  {
    helper_thread->setStackSize(stacksize);
  }
  helper_thread->activate(false);
  //helper_thread->detach();  // Join on delete.
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

void Module::widget_moved(bool,BaseWidget*)
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


// Simple and limited parser for help descriptions.
// gcc-2.95.3 compiler does not support string push_back, use + instead.
//
// states are
//  0 toplevel
//  1 parsing tag
//  2 parsing spaces.
static string
parse_description(const string &in)
{
  std::stack<int> state;
  // Initialization of 999999999 is to remove compiler warning and
  // to hopefully make this code trip up in a way that it can be 
  // debugged more easily if the input (in) is incorrect.
  string::size_type tagstart = 999999999;
  string out;
  state.push(0);
  state.push(2); // start by eating spaces.
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
	tagstart = i;
	state.push(1);
      }
      else if (c == ' ')
      {
	out = out + ' ';
	state.push(2);
      }
      else if (c == '.' || c == '!' || c == '?')
      {
	out = out + c + ' ' + ' ';
	state.push(2);
      }
      else
      {
	out = out + c;
      }
    }
    else if (state.top() == 1)
    {
      if (c == '>')
      {
	if (i > 1 && in[i-2] == '/' && in[i-1] == 'p')
	{
	  out = out + '\n' + '\n';
	  state.pop();
	  state.push(2);
	}
	else
	{
	  const string tag = in.substr(tagstart, i-tagstart+1);
	  if (tag.find("<modref") == 0)
	  {
	    const string tmp = tag.substr(tag.find("name=\"") + 6);
	    const string modname = tmp.substr(0, tmp.find('"'));
	    out = out + modname;
	    state.pop();
	    if (state.top() == 2) state.pop();
	  }
	  else if (tag.find("<listitem") == 0)
	  {
	    out = out + "  * ";
	    state.pop();
	    if (state.top() != 2) state.push(2);
	  }
	  else
	  {
	    // Just eat this tag.
	    state.pop();
	  }
	}
      }
    }
    else if (state.top() == 2)
    {
      if (c == '<')
      {
	tagstart = i;
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
  if(args[1] == "oportcolor") {
    if (args.count() != 3)
      args.error(args[0]+" "+args[1]+" takes a port #");
    int pnum;
    if (!string_to_int(args[2], pnum))
      args.error(args[0]+" "+args[1]+" cant parse port #"+args[2]);
    if (pnum >= oports.size() || pnum < 0)
      args.error(args[0]+" "+args[1]+" port #"+args[2]+" invalid");
    else
      args.result(oports[pnum]->get_colorname());

  } else if(args[1] == "iportcolor") {
    if (args.count() != 3)
      args.error(args[0]+" "+args[1]+" takes a port #");
    int pnum;
    if (!string_to_int(args[2], pnum))
      args.error(args[0]+" "+args[1]+" cant parse port #"+args[2]);
    if (lastportdynamic && pnum >= iports.size())
      pnum = iports.size()-1;
    if (pnum >= iports.size() || pnum < 0)
      args.error(args[0]+" "+args[1]+" port #"+args[2]+" invalid");
    else
      args.result(iports[pnum]->get_colorname());

  } else if(args[1] == "oportname") {
    if (args.count() != 3)
      args.error(args[0]+" "+args[1]+" takes a port #");
    int pnum;
    if (!string_to_int(args[2], pnum))
      args.error(args[0]+" "+args[1]+" cant parse port #"+args[2]);
    if (pnum >= oports.size() || pnum < 0)
      args.error(args[0]+" "+args[1]+" port #"+args[2]+" invalid");
    else
      args.result(oports[pnum]->get_typename()+" "+oports[pnum]->get_portname());

  } else if(args[1] == "iportname") {
    if (args.count() != 3)
      args.error(args[0]+" "+args[1]+" takes a port #");
    int pnum;
    if (!string_to_int(args[2], pnum))
      args.error(args[0]+" "+args[1]+" cant parse port #"+args[2]);

    if (lastportdynamic && pnum >= iports.size())
      pnum = iports.size()-1;

    if (pnum >= iports.size() || pnum < 0)
      args.error(args[0]+" "+args[1]+" port #"+args[2]+" invalid");
    else
      args.result(iports[pnum]->get_typename()+" "+iports[pnum]->get_portname());
  } else if(args[1] == "oportcount") {
    if (args.count() != 2)
      args.error(args[0]+" "+args[1]+" takes no arguments");
    args.result(to_string(oports.size()));
  } else if(args[1] == "iportcount") {
    if (args.count() != 2)
      args.error(args[0]+" "+args[1]+" takes no arguments");
    args.result(to_string(iports.size()));
  } else if(args[1] == "needexecute"){
    if(!abort_flag){
      abort_flag=1;
      want_to_execute();
    }
  } else if(args[1] == "getpid"){
    args.result(to_string(pid_));
  } else if(args[1] == "help"){
    args.result(parse_description(description));
  } else if(args[1] == "remark"){
    remark(args[2]);
  } else if(args[1] == "warning"){
    warning(args[2]);
  } else if(args[1] == "error"){
    error(args[2]);
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
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    cout << id << ":ERROR: " << str << "\n";
  }
  msgStream_flush();
  msgStream_ << "ERROR: " << str << '\n';
  gui->execute(id + " append_log_msg {" + msgStream_.str() + "} red");
  msgStream_.str("");
  update_msg_state(Error); 
}

void Module::warning(const string& str)
{
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    cout << id << ":WARNING: " << str << "\n";
  }
  msgStream_flush();
  msgStream_ << "WARNING: " << str << '\n';
  gui->execute(id + " append_log_msg {" + msgStream_.str() + "} yellow");
  msgStream_.str("");
  update_msg_state(Warning); 
}

void Module::remark(const string& str)
{
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    cout << id << ":REMARK: " << str << "\n";
  }
  msgStream_flush();
  msgStream_ << "REMARK: " << str << '\n';
  gui->execute(id + " append_log_msg {" + msgStream_.str() + "} blue");
  msgStream_.str("");
  update_msg_state(Remark); 
}


void Module::msgStream_flush()
{
  if (msgStream_.str() != "")
  {
    gui->execute(id + " append_log_msg {" + msgStream_.str() + "} black");
    msgStream_.str("");
  }
}


void Module::postMessage(const string& str)
{
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    cout << id << ":postMessage: " << str << "\n";
  }
  gui->postMessage(moduleName + ": " + str, false);
}

void Module::do_execute()
{
  int i;
  abort_flag=0;

  // Reset all of the ports.
  for (i=0; i<oports.size(); i++)
  {
    oports[i]->reset();
  }
  for (i=0; i<iports.size(); i++)
  {
    iports[i]->reset();
  }

  // Reset the TCL variables.
  reset_vars();

  // Call the User's execute function.
  update_msg_state(Reset);
  update_state(JustStarted);
  timer.clear();
  timer.start();

  try {
    execute();
  }
  catch (const Exception &e)
  {
    error("Module crashed with the following exception:");
    error(e.message());
    if (e.stackTrace())
      {
	error("Thread Stacktrace:");
	error(e.stackTrace());
      }
  }
  catch (const string a)
  {
    error(a);
  }
  catch (const char *a)
  {
    error(string(a));
  }
  catch (...)
  {
    error("Module crashed with no reason given.");
  }

  timer.stop();
  update_state(Completed);

  // Call finish on all ports.
  for (i=0;i<iports.size(); i++)
  {
    iports[i]->finish();
  }
  for (i=0; i<oports.size(); i++)
  {
    oports[i]->finish();
  }
}


void
Module::do_synchronize()
{
  for (int i=0; i<oports.size(); i++)
  {
    oports[i]->synchronize();
  }
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
  return (flag > 1);
}

void Module::popupUI()
{
  gui->execute(id+" initialize_ui");
}
