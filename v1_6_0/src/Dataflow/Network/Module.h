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
#include <Core/Util/Timer.h>
#include <iosfwd>
#include <string>
#include <map>

#define DECLARE_MAKER(name) \
extern "C" Module* make_##name(GuiContext* ctx) \
{ \
  return new name(ctx); \
}

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

  class Module : public ModulePickable, public GuiCallback {
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

    // Used by SCIRun2
    Network* getNetwork() {
      return network;
    }
    bool haveUI();
    void popupUI();

    // Used by Ports
    bool showStats();

    // Used by modules
    void error(const std::string&);
    void warning(const std::string&);
    void remark(const std::string&);
    void postMessage(const std::string&);

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

  protected:

    template <class DC>
    bool module_dynamic_compile(const CompileInfo &ci, DC &result);

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
    FutureValue<int> helper_done;
    friend class Network;
    friend class OPort;
    friend class IPort;
    string id;
    bool abort_flag;
    void want_to_execute();
    void update_progress(double);
    void update_progress(double, Timer &);
    void update_progress(int, int);
    void update_progress(int, int, Timer &);
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
    void light_module();
    void reset_module_color();
    void update_msg_state(MsgState);  
    CPUTimer timer;
  public:
    ostream &msgStream_;
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
    bool show_stat;
    
    PortManager<OPort*> oports;
    PortManager<IPort*> iports;
    iport_maker dynamic_port_maker;
    string lastportname;
    int first_dynamic_port;
    unsigned long stacksize;
    ModuleHelper* helper;
    Network* network;

    GuiString  notes ;
    GuiInt show_status;

    Module(const Module&);
    Module& operator=(const Module&);
  };


template <class DC>
bool
Module::module_dynamic_compile(const CompileInfo &ci, DC &result)
{
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().fetch(ci, algo_handle))
  {
    remark("Dynamically compiling some code.");
    light_module();
    const bool status =
      DynamicLoader::scirun_loader().compile_and_store(ci, false, msgStream_);
    reset_module_color();
    msgStream_.flush();
    remark("Dynamic compilation completed.");

    if (! (status && DynamicLoader::scirun_loader().fetch(ci, algo_handle)))
    {
      error("Could not compile algorithm for '" +
	    ci.template_class_name_ + "<" + ci.template_arg_ + ">'.");
      return false;
    }
  }

  result = dynamic_cast<typename DC::pointer_type>(algo_handle.get_rep());
  if (result.get_rep() == 0) 
  {
    error("Could not get algorithm for '" +
	  ci.template_class_name_ + "<" + ci.template_arg_ + ">'.");
    return false;
  }
  return true;
}


}

#endif
