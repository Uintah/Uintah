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
  Moduleions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  Module.h: Classes for module ports
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCIRun_Dataflow_Network_Module_h
#define SCIRun_Dataflow_Network_Module_h

#include <Dataflow/Network/Port.h>
#include <Core/Util/Assert.h>
#include <Core/GuiInterface/TCLstrbuff.h>
#include <Core/GeomInterface/Pickable.h>
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/FutureValue.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Util/Timer.h>
#include <Core/Util/ProgressReporter.h>
#include <iosfwd>
#include <string>
#include <map>

#define DECLARE_MAKER(name) \
extern "C" Module* make_##name(GuiContext* ctx) \
{ \
  return new name(ctx); \
}

// CollabVis code begin
// Stupid ViewServer hack
#ifdef HAVE_COLLAB_VIS
#undef ASIP_SHORT_NAMES
#endif
// End of hack
// CollabVis code end

namespace SCIRun {
using namespace std;
class IPort;
class OPort;
class GuiContext;
class GuiInterface;
class MessageBase;
class Module;
class ModuleHelper;
class Network;
class Scheduler;

typedef IPort* (*iport_maker)(Module*, const string&);
typedef OPort* (*oport_maker)(Module*, const string&);
typedef Module* (*ModuleMaker)(GuiContext* ctx);
typedef std::multimap<string,int> port_map_type;
typedef std::pair<port_map_type::iterator,
		  port_map_type::iterator> port_range_type;


template<class T> 
class PortManager {
private:
  port_map_type namemap_;
  vector<T> ports_;
    
public:
  int size();
  void add(const T &item);
  void remove(int item);
  const T& operator[](int);
  port_range_type operator[](string);
};

template<class T>
int
PortManager<T>::size()
{ 
  return ports_.size(); 
}

template<class T>
void
PortManager<T>::add(const T &item)
{ 
  namemap_.insert(pair<string, int>(item->get_portname(), ports_.size())); 
  ports_.push_back(item);
}

template<class T>
void
PortManager<T>::remove(int item)
{
  string name = ports_[item]->get_portname();
  port_map_type::iterator erase_me;

  port_range_type p = namemap_.equal_range(name);
  for (port_map_type::iterator i=p.first;i!=p.second;i++)
    if ((*i).second>item)
      (*i).second--;
    else if ((*i).second==item)
      erase_me = i;
    
  ports_.erase(ports_.begin() + item);
  namemap_.erase(erase_me);
}

template<class T>
const T&
PortManager<T>::operator[](int item)
{
  ASSERT(size() > item);
  return ports_[item];
}

template<class T>
port_range_type
PortManager<T>::operator[](string item)
{
  return port_range_type(namemap_.equal_range(item));
}

class Module : public ProgressReporter, public ModulePickable, public GuiCallback
{
public:
  enum SchedClass {
    Sink,
    Source,
    Filter,
    Iterator,
    ViewerSpecial
  };
  Module(const std::string& name, GuiContext* ctx, SchedClass,
	 const string& cat="unknown", const string& pack="unknown");
  virtual ~Module();
  void kill_helper();

  // Used by SCIRun2
  Network* getNetwork() {
    return network;
  }
  bool haveUI();
  void popupUI();

  bool showStats() { return true; }

  // ProgressReporter function
  virtual void error(const std::string&);
  virtual void warning(const std::string&);
  virtual void remark(const std::string&);
  virtual void postMessage(const std::string&);
  virtual std::ostream &msgStream() { return msgStream_; }
  virtual void msgStream_flush() { msgStream_.flush(); }

  virtual void report_progress( ProgressState );

  port_range_type getIPorts(const string &name);
  port_range_type getOPorts(const string &name);
  IPort* getIPort(const string &name);
  OPort* getOPort(const string &name);
  OPort* getOPort(int idx);
  IPort* getIPort(int idx);

  // next 6 are Deprecated
  port_range_type get_iports(const string &name);
  port_range_type get_oports(const string &name);
  IPort* get_iport(const string &name);
  OPort* get_oport(const string &name);
  OPort* get_oport(int idx);
  IPort* get_iport(int idx);

  int numIPorts();
  int numOPorts();

  // Used by widgets
  GuiInterface* getGui();

  // Used by ModuleHelper
  void setPid(int pid);
  virtual void do_execute();
  virtual void execute() = 0;

  void request_multisend(OPort*);
  Mailbox<MessageBase*> mailbox;
  void set_context(Scheduler* sched, Network* network);

  // Callbacks
  virtual void connection(Port::ConnectionState, int, bool);
  virtual void widget_moved(bool last);

  // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
  void getPosition(int& x, int& y){ return get_position(x, y); }
  string getID() const { return id; }
#endif
  // CollabVis code end
  
  template<class DC>
  bool module_dynamic_compile( CompileInfoHandle ci, DC &result )
  {
    return DynamicCompilation::compile( ci, result, this );
  }

protected:

  virtual void tcl_command(GuiArgs&, void*);

  GuiInterface* gui;
  GuiContext* ctx;

  friend class ModuleHelper;
  friend class NetworkEditor;
  friend class PackageDB;
  // name is deprecated
  string name;
  string moduleName;
  string packageName;
  string categoryName;
  string description;
  Scheduler* sched;
  bool lastportdynamic;
  int pid_;
  bool have_own_dispatch;
  friend class Network;
  friend class OPort;
  friend class IPort;
  string id;
  bool abort_flag;
  void want_to_execute();
  virtual void update_progress(double);
  virtual void update_progress(double, Timer &);
  virtual void update_progress(int, int);
  virtual void update_progress(int, int, Timer &);
  virtual void accumulate_progress(int);
  enum State {
    NeedData,
    JustStarted,
    Executing,
    Completed
  };
  enum MsgState {  
    Remark,
    Warning,
    Error,
    Reset
  };
  void update_state(State);
  virtual void light_module();
  virtual void reset_module_color();
  void update_msg_state(MsgState);  
  CPUTimer timer;
public:
  TCLstrbuff msgStream_;
protected:
  void get_position(int& x, int& y);
  virtual void emit_vars(std::ostream& out, const std::string& modname);
  void setStackSize(unsigned long stackSize);
  void reset_vars();

  // Used by Scheduler
  friend class Scheduler;
  bool need_execute;
  SchedClass sched_class;
private:
  void remove_iport(int);
  void add_iport(IPort*);
  void add_oport(OPort*);
  void reconfigure_iports();
  void reconfigure_oports();
  State state;
  MsgState msg_state;  
  double progress;
    
  PortManager<OPort*> oports;
  PortManager<IPort*> iports;
  iport_maker dynamic_port_maker;
  string lastportname;
  int first_dynamic_port;
  unsigned long stacksize;
  ModuleHelper* helper;
  Thread *helper_thread;
  Network* network;

  GuiString  notes ;

  Module(const Module&);
  Module& operator=(const Module&);
};


}

#endif
