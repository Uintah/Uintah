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
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/GuiInterface/GuiContext.h>
#include <Dataflow/GuiInterface/GuiInterface.h>
#include <Core/Thread/Thread.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stack>

using namespace std;
using namespace SCIRun;

#ifdef __APPLE__
const string ext = ".dylib";
#elif defined(_WIN32)
const string ext = ".dll";
#else
const string ext = ".so";
#endif

static void *
FindLibrarySymbol(const string &package, const string &/* type */, 
                  const string &symbol)
{
  void* SymbolAddress = 0;
  LIBRARY_HANDLE so = 0;

  string pak_bname, cat_bname;
  if (package == "SCIRun") {
    pak_bname = "Dataflow" + ext;
    cat_bname = "Dataflow_Network" + ext;
  } else {
    pak_bname =  "Packages_" + package + "_Dataflow" + ext;
    cat_bname = "Packages_" + package + "_Dataflow_Ports" + ext;
  }

#ifndef _WIN32
  pak_bname = string("lib") + pak_bname;
  cat_bname = string("lib") + cat_bname;
#endif

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

static iport_maker
FindIPort(const string &package, const string &datatype)
{
  string maker_symbol = "make_" + datatype + "IPort";
  iport_maker maker =
    (iport_maker)FindLibrarySymbol(package, datatype, maker_symbol);
  return maker;
}  

static oport_maker
FindOPort(const string &package, const string &datatype)
{
  string maker_symbol = "make_" + datatype + "OPort";
  oport_maker maker =
    (oport_maker)FindLibrarySymbol(package, datatype, maker_symbol);
  return maker;
}  

Module::Module(const string& name, GuiContext* ctx,
	       SchedClass sched_class, const string& cat,
	       const string& pack) : 
  ProgressReporter(),
  GuiCallback(),
  mailbox_("Module execution FIFO", 100),
  gui_(ctx->getInterface()),
  module_name_(name),
  package_name_(pack),
  category_name_(cat),
  sched_(0),
  pid_(0),
  have_own_dispatch_(0),
  id_(ctx->getfullname()), 
  abort_flag_(0),
  need_execute_(false),
  sched_class_(sched_class),
  inputs_changed_(false),
  execute_error_(false),
  ctx_(ctx),
  state_(Completed),
  msg_state_(Reset), 
  helper_(0),
  helper_thread_(0),
  network_(0),
  show_stats_(true),
  log_string_(ctx->subVar("log_string", false)),
  lock_("Module lock")
{
  stacksize_=0;

  IPort* iport;
  OPort* oport;
  
  first_dynamic_port_ = -1;
  lastportdynamic_ = 0;
  dynamic_port_maker_ = 0;

  // Auto allocate all ports listed in the .xml file for this module,
  // execpt those whose datatype tag has contents that start with '*'.
  ModuleInfo* info = packageDB->GetModuleInfo(module_name_, category_name_,
					      package_name_);
  if (info) 
  {
    oport_maker maker;
    vector<OPortInfo*>::iterator i2 = info->oports_.begin();
    while (i2 < info->oports_.end())
    {
      OPortInfo* op = *i2++;
      unsigned long strlength = op->datatype.length();
      char* package = new char[strlength+1];
      char* datatype = new char[strlength+1];
      sscanf(op->datatype.c_str(),"%[^:]::%s",package,datatype);
      if (package[0]=='*')
        maker = FindOPort(&package[1],datatype);
      else
        maker = FindOPort(package,datatype);	
      if (maker && package[0]!='*') 
      {
        oport = maker(this, op->name);
        if (oport)
        add_oport(oport);
      }
      delete[] package;
      delete[] datatype;
    }  
    vector<IPortInfo*>::iterator i1 = info->iports_.begin();
    while (i1 < info->iports_.end())
    {
      IPortInfo* ip = *i1++;
      unsigned long strlength = ip->datatype.length();
      char* package = new char[strlength+1];
      char* datatype = new char[strlength+1];
      sscanf(ip->datatype.c_str(),"%[^:]::%s",package,datatype);
      if (package[0]=='*')
        dynamic_port_maker_ = FindIPort(&package[1],datatype);
      else
        dynamic_port_maker_ = FindIPort(package,datatype);	
      if (dynamic_port_maker_ && package[0]!='*') {
        iport = dynamic_port_maker_(this, ip->name);
        if (iport) {
          lastportname_ = ip->name;
          add_iport(iport);
        }
      } else {
	cerr << "Cannot create port: " << datatype << '\n';
	dynamic_port_maker_ = 0;
      }
      delete[] package;
      delete[] datatype;
    }
  } else {
    cerr << "Cannot find module info for module: " 
	 << package_name_ << "/" << category_name_ << "/" << module_name_ << '\n';
  }

  // the last port listed in the .xml file may or may not be dynamic.
  // if found and lastportdynamic is true, the port is dynamic.
  // otherwise it is not dynamic.
  if (lastportdynamic_ && !dynamic_port_maker_){
    lastportdynamic_ = 0;
  }
  first_dynamic_port_ = iports_.size()-1;
}

Module::~Module()
{
  // kill_helper joins the helper thread & waits until module execution is done
  kill_helper();
  // After execution is done, delete the TCL command: $this-c
  gui_->delete_command(id_+"-c" );
  // Remove the dummy proc $this that was eating up all the TCL commands
  // while the module finished up executing
  gui_->eval("catch {rename "+id_+" \"\"}");
  
}

int
Module::add_output_port_by_name(std::string name, std::string d_type)
{
  OPort* oport;
  dynamic_port_maker_ = 0;

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
      dynamic_port_maker_ = 0;
    }
  }
  delete[] package;
  delete[] datatype; 

  return 0;
}

int
Module::add_input_port_by_name(std::string name, std::string d_type)
{
  IPort* iport;
  dynamic_port_maker_ = 0;

  unsigned long strlength = d_type.length();
  char* package = new char[strlength+1];
  char* datatype = new char[strlength+1];
  sscanf(d_type.c_str(),"%[^:]::%s",package,datatype);
  if (package[0]=='*')
    dynamic_port_maker_ = FindIPort(&package[1],datatype);
  else
    dynamic_port_maker_ = FindIPort(package,datatype);	
  if (dynamic_port_maker_ && package[0]!='*') {
    iport = dynamic_port_maker_(this,name);
    if (iport) {
      lastportname_ = name;
      add_iport(iport);
    }
  } else {
    cerr << "Err: Cannot create port: " << datatype << '\n';
    dynamic_port_maker_ = 0;
  }
  delete[] package;
  delete[] datatype;
  
  return 0;
}

void
Module::delete_warn() 
{
  set_show_stats(false);
  MessageBase *msg = scinew MessageBase(MessageTypes::GoAwayWarn);
  mailbox_.send(msg);
}

void
Module::kill_helper()
{
  if (helper_thread_)
  {
    // kill the helper thread
    MessageBase *msg = scinew MessageBase(MessageTypes::GoAway);
    mailbox_.send(msg);
    helper_thread_->join();
    helper_thread_ = 0;
  }
}


void
Module::report_progress( ProgressState state )
{
  switch ( state ) {
  case ProgressReporter::Starting:
    break;
  case ProgressReporter::Compiling:
    gui_->execute(id_+" set_compiling_p 1");
    remark("Dynamically compiling some code.");
    break;
  case ProgressReporter::CompilationDone:
    gui_->execute(id_+" set_compiling_p 0");
    remark("Dynamic compilation completed.");
    break;
  case ProgressReporter::Done:
    break;
  }
}

void
Module::update_state(State st)
{
  state_= st;
  char* s;

  switch(st){
  case JustStarted:
    s = "JustStarted";
    break;
  case NeedData:
    s = "NeedData";
    break;
  case Executing:
    s = "Executing";
    break;
  case Completed:
    s = "Completed";
    break;
  default:
    s = "unknown";
    break;

  }
  double time = timer_.time();
  if(time<0)
    time=0;
  time = Min(time, 1.0e10); // Clamp NaN
  gui_->execute(id_+" set_state " + s + " " + to_string(time));

  if (sci_getenv_p("SCI_REGRESSION_TESTING") && st == Completed)
  {
    cout << id_  + ":RUNTIME: " + to_string(time) + "\n";
  }
}


void
Module::update_msg_state(MsgState st)
{
  // only change the state if the new state
  // is of higher priority
  if( !( ((msg_state_ == Error) && (st != Reset))  || 
	 ((msg_state_ == Warning) && (st == Remark)) ) ) {
    msg_state_=st;
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
    gui_->execute(id_+" set_msg_state " + s);
  }
}


#define PROGRESS_GRANULARITY 32


void
Module::update_progress(double p)
{
  if (state_ != Executing) { update_state(Executing); }
  const double cp = Clamp(p, 0.0, 1.0);
  const int crp = progress_current_ * PROGRESS_GRANULARITY / progress_max_;
  const int nrp = (int)(cp * PROGRESS_GRANULARITY);
  if (crp != nrp)
  {
    progress_current_.set((int)(cp * progress_max_));
    string str = to_string(((double)nrp) / PROGRESS_GRANULARITY);
    gui_->execute(id_+" set_progress "+str+" "+to_string(timer_.time()));
  }
}


void
Module::update_progress(int current, int maxpr)
{
  if (state_ != Executing) { update_state(Executing); }
  if( maxpr ) {
    const int crp = progress_current_ * PROGRESS_GRANULARITY / progress_max_;
    const int nrp = current * PROGRESS_GRANULARITY / maxpr;
    if (crp != nrp || maxpr != progress_max_) {
      progress_max_ = maxpr;
      progress_current_.set(current);
      string str = to_string(((double)nrp) / PROGRESS_GRANULARITY);
      gui_->execute(id_+" set_progress "+str+" "+to_string(timer_.time()));
    }
  }
}


void
Module::increment_progress()
{
  if (state_ != Executing) { update_state(Executing); }
  unsigned int crp = progress_current_ * PROGRESS_GRANULARITY / progress_max_;
  progress_current_++;
  unsigned int nrp = progress_current_ * PROGRESS_GRANULARITY / progress_max_;
  if (crp != nrp)
  {
    string str = to_string(((double)nrp) / PROGRESS_GRANULARITY);
    gui_->execute(id_+" set_progress "+str+" "+to_string(timer_.time()));
  }
}



// Port stuff
void
Module::add_iport(IPort* port)
{
  port->set_which_port(iports_.size());
  IPortHandle handle(port);
  iports_.add(handle);
  gui_->execute("drawPorts "+id_+" i");
}

void
Module::add_oport(OPort* port)
{
  port->set_which_port(oports_.size());
  OPortHandle handle(port);
  oports_.add(handle);
  gui_->execute("drawPorts "+id_+" o");
}

void
Module::remove_iport(int which)
{
  // remove the indicated port, then
  // collapse the remaining ports together
  iports_[which]->deactivate();
  iports_.remove(which);  
  gui_->execute("removePort {"+id_+" "+to_string(which)+" i}");
  // rename the collapsed ports and their connections
  // to reflect the positions they collapsed to.
  for (int port=which;port<iports_.size();port++) {
    iports_[port]->set_which_port(iports_[port]->get_which_port()-1);
    for (int connNum=0;connNum<iports_[port]->nconnections();connNum++)
      iports_[port]->connection(connNum)->makeID();
  }
  gui_->execute("drawPorts "+id_+" i");
}

port_range_type
Module::get_iports(const string &name)
{
  return (iports_[name]);
}

IPort*
Module::get_iport(int item)
{
  IPortHandle h = iports_[item];
  return (h.get_rep());
}

OPort*
Module::get_oport(int item)
{
  OPortHandle h = oports_[item];
  return (h.get_rep());
}

IPort*
Module::get_iport(const string& name)
{
  IPortHandle h;
  if (!(get_iport_handle(name,h))) throw "Unable to initialize oport '" + name + "'.";
  return (h.get_rep());
}

OPort*
Module::get_oport(const string& name)
{
  OPortHandle h;
  if (!(get_oport_handle(name,h))) throw "Unable to initialize oport '" + name + "'.";
  return (h.get_rep());
}

port_range_type
Module::get_input_ports(const string &name)
{
  return get_iports(name);
}

IPort*
Module::get_input_port(const string &name)
{
  IPortHandle h;
  get_iport_handle(name,h);
  return (h.get_rep());
}

OPort*
Module::get_output_port(const string &name)
{
  OPortHandle h;
  get_oport_handle(name,h);
  return (h.get_rep());
}

IPort*
Module::get_input_port(int item)
{
  IPortHandle h;
  get_iport_handle(item,h);
  return (h.get_rep());
}

OPort*
Module::get_output_port(int item)
{
  OPortHandle h;
  get_oport_handle(item,h);
  return (h.get_rep());
}


bool
Module::oport_cached(const string &name)
{
  OPortHandle h;
  if (!(get_oport_handle(name,h))) throw "Unable to initialize oport '" + name + "'.";
  return (h->have_data());
}

bool
Module::oport_supports_cache_flag(int p)
{
  OPortHandle h;
  if (!(get_oport_handle(p,h))) return (false);
  return (h->cache_flag_supported());
}

bool
Module::get_oport_cache_flag(int p)
{
  OPortHandle h;
  if (!(get_oport_handle(p,h))) return (false);
  return (h->get_cache());
}

void
Module::set_oport_cache_flag(int p, bool val)
{
  OPortHandle h;
  if (!(get_oport_handle(p,h))) return;
  h->set_cache(val);
}


void
Module::connection(Port::ConnectionState mode, int which_port, bool is_oport)
{
  if(!is_oport && lastportdynamic_ && 
     dynamic_port_maker_ && (which_port >= first_dynamic_port_)) {
    if(mode == Port::Disconnected) {
      remove_iport(which_port);
    } else if (which_port == iports_.size()-1) {
      add_iport(dynamic_port_maker_(this,lastportname_));
    }
  }
  
  // do nothing by default
}

void
Module::set_context(Network* network)
{
  this->network_=network;
  this->sched_=network->get_scheduler();
  ASSERT(helper_ == 0);
  // Start up the event loop
  helper_=scinew ModuleHelper(this);
  helper_thread_ = 
    scinew Thread(helper_, module_name_.c_str(), 0, Thread::NotActivated);
  if(stacksize_)
  {
    helper_thread_->setStackSize(stacksize_);
  }
  helper_thread_->activate(false);
  //helper_thread_->detach();  // Join on delete.
}

void
Module::set_stack_size(unsigned long s)
{
   stacksize_=s;
}

void
Module::want_to_execute()
{
    sched_->lockNeedExecute();
    need_execute_ = true;
    sched_->unlockNeedExecute();
    sched_->do_scheduling();
}

void
Module::widget_moved(bool,BaseWidget*)
{
}

void
Module::get_position(int& x, int& y)
{
  string result;
  if(!gui_->eval(id_+" get_x", result)){
    error("Error getting x coordinate.");
    return;
  }
  if(!string_to_int(result, x)) {
    error("Error parsing x coordinate.");
    return;
  }
  if(!gui_->eval(id_+" get_y", result)){
    error("Error getting y coordinate.");
    return;
  }
  if(!string_to_int(result, y)) {
    error("Error parsing y coordinate.");
    return;
  }
}


void
Module::tcl_command(GuiArgs& args, void*)
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
    if (pnum >= oports_.size() || pnum < 0)
      args.error(args[0]+" "+args[1]+" port #"+args[2]+" invalid");
    else
      args.result(oports_[pnum]->get_colorname());

  } else if(args[1] == "iportcolor") {
    if (args.count() != 3)
      args.error(args[0]+" "+args[1]+" takes a port #");
    int pnum;
    if (!string_to_int(args[2], pnum))
      args.error(args[0]+" "+args[1]+" cant parse port #"+args[2]);
    if (lastportdynamic_ && pnum >= iports_.size())
      pnum = iports_.size()-1;
    if (pnum >= iports_.size() || pnum < 0)
      args.error(args[0]+" "+args[1]+" port #"+args[2]+" invalid");
    else
      args.result(iports_[pnum]->get_colorname());

  } else if(args[1] == "oportname") {
    if (args.count() != 3)
      args.error(args[0]+" "+args[1]+" takes a port #");
    int pnum;
    if (!string_to_int(args[2], pnum))
      args.error(args[0]+" "+args[1]+" cant parse port #"+args[2]);
    if (pnum >= oports_.size() || pnum < 0)
      args.error(args[0]+" "+args[1]+" port #"+args[2]+" invalid");
    else
      args.result(oports_[pnum]->get_typename()+" "+oports_[pnum]->get_portname());

  } else if(args[1] == "iportname") {
    if (args.count() != 3)
      args.error(args[0]+" "+args[1]+" takes a port #");
    int pnum;
    if (!string_to_int(args[2], pnum))
      args.error(args[0]+" "+args[1]+" cant parse port #"+args[2]);

    if (lastportdynamic_ && pnum >= iports_.size())
      pnum = iports_.size()-1;

    if (pnum >= iports_.size() || pnum < 0)
      args.error(args[0]+" "+args[1]+" port #"+args[2]+" invalid");
    else
      args.result(iports_[pnum]->get_typename()+" "+iports_[pnum]->get_portname());
  } else if(args[1] == "oportcount") {
    if (args.count() != 2)
      args.error(args[0]+" "+args[1]+" takes no arguments");
    args.result(to_string(oports_.size()));
  } else if(args[1] == "iportcount") {
    if (args.count() != 2)
      args.error(args[0]+" "+args[1]+" takes no arguments");
    args.result(to_string(iports_.size()));
  } else if(args[1] == "needexecute"){
    if(!abort_flag_){
      abort_flag_=1;
      // Back up the network on any module executes
      gui_->eval("backupNetwork");
      want_to_execute();
    }
  } else if(args[1] == "getpid"){
    args.result(to_string(pid_));
  } else if(args[1] == "help"){
    args.result("http://software.sci.utah.edu/SCIRunDocs/index.php/CIBC:Documentation:SCIRun:Reference:" + package_name_ + ":" + module_name_);
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


void
Module::set_pid(int pid)
{
  pid_ = pid;
}


static string
esc_brackets(const string &str)
{
  string result;
  for (unsigned int i = 0; i < str.size(); i++)
  {
    if (str[i] == '{' || str[i] == '}') result.push_back('\\');
    result.push_back(str[i]);
  }
  return result;
}


// Error conditions
void
Module::error(const string& str)
{
  const string newstr = "ERROR: " + str + "\n";
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    cout << id_ + ":" + newstr;
    cout.flush();
  }
  msg_stream_flush();
  gui_->execute(id_ + " append_log_msg {" + esc_brackets(newstr) + "} red");
  update_msg_state(Error); 
}

void
Module::warning(const string& str)
{
  const string newstr = "WARNING: " + str + "\n";
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    cout << id_ + ":" + newstr;
    cout.flush();
  }
  msg_stream_flush();
  gui_->execute(id_ + " append_log_msg {" + esc_brackets(newstr) + "} yellow");
  update_msg_state(Warning); 
}

void
Module::remark(const string& str)
{
  const string newstr = "REMARK: " + str + "\n";
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    cout << id_ + ":" + newstr;
    cout.flush();
  }
  msg_stream_flush();
  gui_->execute(id_ + " append_log_msg {" + esc_brackets(newstr) + "} blue");
  update_msg_state(Remark); 
}


void
Module::add_raw_message(const string& str)
{
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    cout << str;
    cout.flush();
  }
  msg_stream_flush();
  gui_->execute(id_ + " append_log_msg {" + esc_brackets(str) + "} black");
}


void
Module::compile_error(const string& filename)
{
  const string newstr = "COMPILE ERROR IN FILE: " + filename + "cc\n";
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    cout << id_ + ":" + newstr;
    cout.flush();
  }

  msg_stream_flush();
  gui_->execute(id_ + " append_log_msg {" + newstr + "} OrangeRed");
  update_msg_state(Error); 
  
  gui_->eval(get_id() + " compile_error " + filename);
}

void
Module::msg_stream_flush()
{
  if (msg_stream_.str() != "")
  {
    gui_->execute(id_ + " append_log_msg {" + msg_stream_.str() + "} black");
    msg_stream_.str("");
  }
}

bool
Module::in_power_app()
{
  return (gui_->eval("in_power_app") == "1");
}

void
Module::do_execute()
{
  int i;
  abort_flag_=0;

  // Reset all of the ports.
  for (i=0; i<oports_.size(); i++)
  {
    oports_[i]->reset();
  }
  for (i=0; i<iports_.size(); i++)
  {
    iports_[i]->reset();
  }

  // Reset the TCL variables.
  reset_vars();

  // This gets flagged if the data on any input port has changed.
  // Also used in the various execute functions to note gui_vars changing.
  inputs_changed_ = false;

  // Call the User's execute function.
  update_msg_state(Reset);
  timer_.clear();
  timer_.start();

  update_state(JustStarted);

  try 
  {
    execute();
  }
  catch (const Exception &e)
  {
    error(string("Module crashed with the following exception:\n  ")+
	  e.message());
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

  timer_.stop();
  update_state(Completed);

  // Call finish on all ports.
  size_t s = iports_.size();
  for (i=0;i<s; i++)
  {
    IPortHandle h = iports_.get_port(i);
    if (h.get_rep()) h->finish();
  }
  
  s = oports_.size();
  for (i=0; i<s; i++)
  {
    OPortHandle h = oports_.get_port(i);
    if (h.get_rep()) h->finish();
  }
}


void
Module::do_synchronize()
{
  size_t s = oports_.size();
  for (int i=0; i< s; i++)
  {
    OPortHandle h = oports_.get_port(i);
    if (h.get_rep()) h->synchronize();
  }
}


void
Module::request_multisend(OPort* p1)
{
  sched_->request_multisend(p1);
}

int
Module::num_output_ports()
{
  return oports_.size();
}

int
Module::num_input_ports()
{
  return iports_.size();
}

GuiInterface*
Module::get_gui()
{
  return gui_;
}

void
Module::reset_vars()
{
  ctx_->reset();
}

bool
Module::have_ui()
{
  string result;
  if(!gui_->eval(id_+" have_ui", result)){
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

void
Module::popup_ui()
{
  gui_->execute(id_+" initialize_ui");
}


// Used to send handles for geometry with error checking.
bool
Module::send_output_handle( string port_name,
			    GeomHandle& handle,
			    string obj_name )
{
  // Don't send on empty, assume cached version is more valid instead.
  // Dunno if false return value is correct.  We don't check returns
  // on this one.
  //if (!handle.get_rep()) return false;

  LockingHandle<GeometryOPort> dataport;

  // We always require the port to be there.
  if ( !(get_oport_handle(port_name,dataport))) {
    throw "Incorrect data type sent to output port '" + port_name +
      "' (dynamic_cast failed).";
    return false;
  }

  dataport->delAll();

  if (handle.get_rep())
    dataport->addObj( handle, obj_name );

  dataport->flushViews();

  return true;
}

// Used to send handles for geometry with error checking.
bool
Module::send_output_handle( string port_name,
			    vector<GeomHandle> &handles,
			    vector<string> &obj_names )
{
  // Don't send on empty, assume cached version is more valid instead.
  // Dunno if false return value is correct.  We don't check returns
  // on this one.
  //if (!handles.size()==0) return false;

  LockingHandle<GeometryOPort> dataport;

  // We always require the port to be there.
  if ( !(get_oport_handle(port_name,dataport)) ) {
    throw "Incorrect data type sent to output port '" + port_name +
      "' (dynamic_cast failed).";
    return false;
  }

  dataport->delAll();

  for (unsigned int i=0; i<handles.size(); i++)
    if (handles[i].get_rep())
      dataport->addObj( handles[i], obj_names[i] );

  dataport->flushViews();

  return true;
}
