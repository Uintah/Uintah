
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

#include <Classlib/Array1.h>
#include <Classlib/String.h>
#include <Classlib/Timer.h>
#include <Multitask/ITC.h>
#include <TCL/TCL.h>
class Connection;
class IPort;
class MessageBase;
class Network;
class NetworkEditor;
class OPort;
class Vector;
class GeomPick;

class Module : public TCL {
public:
    enum State {
	NeedData,
	JustStarted,
	Executing,
	Completed
    };
public:
    friend class ModuleHelper;
    virtual void do_execute();
    virtual void execute()=0;

    State state;
    Array1<OPort*> oports;
    Array1<IPort*> iports;
    ModuleHelper* helper;
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
	SalmonSpecial
    };
    Module(const clString& name, const clString& id, SchedClass);
    virtual ~Module();
    Module(const Module&, int deep);
    virtual Module* clone(int deep)=0;

    Mailbox<MessageBase*> mailbox;

    inline State get_state(){ return state;}
    inline double get_progress(){ return progress;}

    void get_position(int& x, int& y);

    // Callbacks
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void geom_pick(GeomPick*, void*);
    virtual void geom_release(GeomPick*, void*);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, void*);
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
    void tcl_command(TCLArgs&, void*);
};

#endif /* SCI_project_Module_h */
