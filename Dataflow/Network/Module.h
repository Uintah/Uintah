
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
#include <Core/Containers/Array1.h>
#include <Core/Containers/String.h>
#include <Core/Util/Timer.h>
#include <Core/TclInterface/TCL.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/TclInterface/TCLstrbuff.h>
#include <Core/Thread/Mailbox.h>
#include <Core/Geom/Pickable.h>
#include <map>

namespace SCIRun {

class Vector;
class GeomPick;
class GeomObj;
class Connection;
class Network;
class NetworkEditor;
class MessageBase;
class AI;
class ViewWindow;
class Module;

typedef IPort* (*iport_maker)(Module*, const clString&);
typedef OPort* (*oport_maker)(Module*, const clString&);
typedef std::multimap<clString,int> port_map;
typedef port_map::iterator port_iter;
typedef std::pair<clString,int> port_pair;
typedef std::pair<port_map::iterator,port_map::iterator> dynamic_port_range;

template<class T> 
class PortManager {
public:
  port_map namemap;
  Array1<T> ports;

  int size();
  void add(T);
  void remove(int);
  const T& operator[](int);
  dynamic_port_range* operator[](clString);
};

class PSECORESHARE Module : public TCL, public Pickable {
  /*
   * This exists to trip people up that still have clone and
   * copy constructors in the modules - they should be removed.
   */
  Module(const Module&);
  
  // Added by Mohamed Dekhil for the CSAFE project
  TCLstring  notes ;
  TCLint show_status;
  unsigned long stacksize;

protected:
  //////////
  // Log stream for the module
  TCLstrbuff msgStream_;

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
    char lastportdynamic;
    iport_maker dynamic_port_maker;
    clString lastportname;
    ModuleHelper* helper;
    int have_own_dispatch;

    double progress;
    CPUTimer timer;
    //WallClockTimer timer;

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
    Module(const clString& name, const clString& id, SchedClass,
	   const clString& cat="unknown", const clString& pack="unknown");
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
    virtual void geom_pick(GeomPick*, void*, GeomObj*);
  //virtual void geom_pick(GeomPick*, void*, int);
    virtual void geom_pick(GeomPick*, void*);
    virtual void geom_pick(GeomPick*, ViewWindow*, int, const BState& bs);
    virtual void geom_release(GeomPick*, int, const BState& bs);
    virtual void geom_release(GeomPick*, void*, GeomObj*);
  //virtual void geom_release(GeomPick*, void*, int);
    virtual void geom_release(GeomPick*, void*);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, void*, GeomObj*);
  //virtual void geom_moved(GeomPick*, int, double, const Vector&, void*, int);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, void*);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, 
			    int, const BState&);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, 
			    const BState&, int);
    virtual void widget_moved(int);
    virtual void widget_moved2(int last, void *) {
	widget_moved(last);
    }
    // Port manipulations
    void add_iport(IPort*);
    void add_oport(OPort*);
    void remove_iport(int);
    void remove_oport(int);
    void rename_iport(int, const clString&);
    void rename_oport(int, const clString&);
    virtual void reconfigure_iports();
    virtual void reconfigure_oports();
    IPort* get_iport(int);   // return port at position
    OPort* get_oport(int);
    dynamic_port_range* get_iport(char*); // return port(s) with name
    dynamic_port_range* get_oport(char*);

    // Used by Module subclasses
    void error(const clString&);
    void update_state(State);
    void update_progress(double);
    void update_progress(double, Timer &);
    void update_progress(int, int);
    void update_progress(int, int, Timer &);
    void want_to_execute();

    // User Interface information
    NetworkEditor* netedit;
    Network* network;
    clString name;
    clString categoryName;   
    clString packageName;    
    clString moduleName;  
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

    clString id;
    int handle; 	// mm-skeleton and remote share the same handle
    bool remote;        // mm-is this a remote module?  not used.
    bool skeleton;	// mm-is this a skeleton module?
    bool isSkeleton()	{ return skeleton; }
    void tcl_command(TCLArgs&, void*);

    bool get_abort() { return abort_flag; }
};

typedef Module* (*ModuleMaker)(const clString& id);


template<class T>
int PortManager<T>::size() { 
  return ports.size(); 
}

template<class T>
void PortManager<T>::add(T item) { 
  namemap.insert(port_pair(item->get_portname(),ports.size())); 
  ports.add(item);
}

template<class T>
void PortManager<T>::remove(int item) {
  clString name = ports[item]->get_portname();
  port_iter erase_me;

  dynamic_port_range p = namemap.equal_range(name);
  for (port_iter i=p.first;i!=p.second;i++)
    if ((*i).second>item)
      (*i).second--;
    else if ((*i).second==item)
      erase_me = i;

  ports.remove(item);
  namemap.erase(erase_me);
}

template<class T>
const T& PortManager<T>::operator[](int item) {
  return ports[item];
}

template<class T>
dynamic_port_range* PortManager<T>::operator[](clString item) {
  return new dynamic_port_range(namemap.equal_range(item));
}


} // End namespace SCIRun

#endif /* SCI_project_Module_h */
