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
 *  Module.h: Base class for modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Module_h
#define SCI_project_Module_h 1

#ifdef _WIN32
#pragma warning(disable:4355 4786)
#endif

#include <Dataflow/share/share.h>
#include <Dataflow/Network/Port.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Util/Timer.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCLstrbuff.h>
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/FutureValue.h>
#include <Core/Geom/Pickable.h>
#include <map>

namespace SCIRun {

class Vector;
class GeomPick;
class GeomObj;
class Connection;
class Network;
class MessageBase;
class AI;
class ViewWindow;
class Module;

typedef IPort* (*iport_maker)(Module*, const string&);
typedef OPort* (*oport_maker)(Module*, const string&);
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



class PSECORESHARE Module : public TCL, public ModulePickable {
  /*
   * This exists to trip people up that still have clone and
   * copy constructors in the modules - they should be removed.
   */
  Module(const Module&);
  
  // Added by Mohamed Dekhil for the CSAFE project
  GuiString  notes ;
  GuiInt show_status;
  unsigned long stacksize;

protected:
  //////////
  // Log stream for the module
  TCLstrbuff msgStream_;
  int pid_;
public:
  enum State {
    NeedData,
    JustStarted,
    Executing,
    Completed
  };
public:
  int show_stat;
  
  friend class ModuleHelper;
  virtual void do_execute();
  virtual void execute()=0;
  void setStackSize(unsigned long stackSize);

  State state;
  PortManager<OPort*> oports;
  PortManager<IPort*> iports;
  int first_dynamic_port;
  char lastportdynamic;
  iport_maker dynamic_port_maker;
  string lastportname;
  ModuleHelper* helper;
  FutureValue<int> helper_done;
  int have_own_dispatch;
    
  double progress;
  CPUTimer timer;

public:
  enum ConnectionMode {
    Connected,
    Disconnected
  };
  enum SchedClass {
    Sink,
    Source,
    Filter,
    Iterator,
    ViewerSpecial
  };
  Module(const string& name, const string& id, SchedClass,
	 const string& cat="unknown", const string& pack="unknown");
  virtual ~Module();

  /*
   * This exists to trip people up that still have clone and
   * copy constructors in the modules - they shoudl be removed.
   */
  virtual int clone(int deep);

  Mailbox<MessageBase*> mailbox;

  inline State get_state(){ return state;}
  inline double get_progress(){ return progress;}

  void get_position(int& x, int& y);

  // Callbacks
  virtual void connection(Module::ConnectionMode, int, int);
  virtual void widget_moved(int);
  virtual void widget_moved2(int last, void *) { widget_moved(last); }

  // Port manipulations
  void add_iport(IPort*);
  void add_oport(OPort*);
  void remove_iport(int);
  void remove_oport(int);
  void rename_iport(int, const string&);
  void rename_oport(int, const string&);
  virtual void reconfigure_iports();
  virtual void reconfigure_oports();
  // return port at position
  IPort* get_iport(int item) { return iports[item]; }
  OPort* get_oport(int item) { return oports[item]; }
  IPort* get_iport(const string &name);
  OPort* get_oport(const string &name);

  // return port(s) with name
  port_range_type get_iports(const string &name) { return iports[name]; }
  port_range_type get_oports(const string &name) { return oports[name]; }

  // Used by Module subclasses
  void error(const string&);
  void warning(const string&);
  void remark(const string&);
  void update_state(State);
  void update_progress(double);
  void update_progress(double, Timer &);
  void update_progress(int, int);
  void update_progress(int, int, Timer &);
  void want_to_execute();

  // User Interface information
  NetworkEditor* netedit;
  Network* network;
  string name;
  string categoryName;   
  string packageName;    
  string moduleName;  
  int abort_flag;
public:
  int niports();
  int noports();
  IPort* iport(int);
  OPort* oport(int);
  void multisend(OPort*, OPort* =0);
  void set_context(NetworkEditor*, Network*);
  int need_execute;
  SchedClass sched_class;
  // virtual int should_execute();

  string id;
  int handle; 	// mm-skeleton and remote share the same handle
  bool remote;        // mm-is this a remote module?  not used.
  bool skeleton;	// mm-is this a skeleton module?
  bool isSkeleton()	{ return skeleton; }
  void tcl_command(TCLArgs&, void*);

  bool get_abort() { return abort_flag; }
};

typedef Module* (*ModuleMaker)(const string& id);


} // End namespace SCIRun

#endif /* SCI_project_Module_h */
