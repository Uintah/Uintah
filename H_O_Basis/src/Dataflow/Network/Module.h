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
#include <Core/GeomInterface/Pickable.h>
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/FutureValue.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Util/Timer.h>
#include <Core/Util/ProgressReporter.h>
#include <sstream>
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
class BaseWidget;
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
  void delete_warn();
  bool show_stats() { return show_stats_; }
  void set_show_stats(bool v) {show_stats_ = v;}
  // ProgressReporter function
  virtual void error(const std::string&);
  virtual void warning(const std::string&);
  virtual void remark(const std::string&);
  virtual void postMessage(const std::string&);
  virtual std::ostream &msgStream() { return msgStream_; }
  virtual void msgStream_flush();
  virtual bool in_power_app();

  // Compilation progress.  Should probably have different name.
  virtual void report_progress( ProgressState );

  // Execution time progress.
  // Percent is number between 0.0-1.0
  virtual void update_progress(double percent); 
  virtual void update_progress(unsigned int n, unsigned int max);

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
  virtual void do_synchronize();
  virtual void execute() = 0;

  void request_multisend(OPort*);
  Mailbox<MessageBase*> mailbox;
  virtual void set_context(Network* network);

  // Callbacks
  virtual void connection(Port::ConnectionState, int, bool);
  virtual void widget_moved(bool last, BaseWidget *widget);

  void getPosition(int& x, int& y){ get_position(x, y); }
  string getID() const { return id; }
  // CollabVis code end
  
  template<class DC>
  bool module_dynamic_compile( CompileInfoHandle ci, DC &result )
  {
    return DynamicCompilation::compile( ci, result, this );
  }

  //Used by Bridge stuff for SR2
  int addIPortByName(std::string name, std::string d_type);
  int addOPortByName(std::string name, std::string d_type);
  void want_to_execute();

protected:
  virtual void tcl_command(GuiArgs&, void*);
  virtual void presave() {}

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
  void update_msg_state(MsgState);  
  CPUTimer timer;

  ostringstream msgStream_;
  void get_position(int& x, int& y);
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

  bool        show_stats_;

  GuiString log_string_;

  Module(const Module&);
  Module& operator=(const Module&);
};


}

#endif
