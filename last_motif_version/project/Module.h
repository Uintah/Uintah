
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
#include <Multitask/ITC.h>
class CallbackData;
class Connection;
class IPort;
class MessageBase;
class Network;
class NetworkEditor;
class OPort;
class Vector;

class Module {
public:
    enum State {
	NeedData,
	Executing,
	Completed,
    };
protected:
    friend class ModuleHelper;
    virtual void do_execute()=0;

    int need_update;
    double progress;
    State state;
    Array1<OPort*> oports;
    Array1<IPort*> iports;
    ModuleHelper* helper;
public:
    enum ConnectionMode {
	Connected,
	Disconnected,
    };
    enum SchedClass {
	Sink,
	Source,
	Filter,
    };
    Module(const clString& name, SchedClass);
    virtual ~Module();
    Module(const Module&, int deep);
    virtual Module* clone(int deep)=0;

    Mailbox<MessageBase*> mailbox;

    inline State get_state(){ return state;}
    inline double get_progress(){ return progress;}

    // Callbacks
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void geom_pick(void*);
    virtual void geom_release(void*);
    virtual void geom_moved(int, double, const Vector&, void*);

    // Port manipulations
    void add_iport(IPort*);
    void add_oport(OPort*);
    void remove_iport(int);
    void remove_oport(int);
    void rename_iport(int, const clString&);
    void rename_oport(int, const clString&);
    virtual void get_iport_coords(int, int&, int&);
    virtual void get_oport_coords(int, int&, int&);
    virtual void reconfigure_iports()=0;
    virtual void reconfigure_oports()=0;

    // Used by Module subclasses
    void update_progress(double);
    void update_progress(int, int);
    void want_to_execute();

    // User Interface information
    NetworkEditor* netedit;
    Network* network;
    clString name;
public:
    int niports();
    int noports();
    IPort* iport(int);
    OPort* oport(int);
    int xpos, ypos;
    int width, height;
    void set_context(NetworkEditor*, Network*);

    // For the scheduler
    enum SchedState {
	SchedDormant,
	SchedRegenData,
	SchedNewData,
    };
    SchedState sched_state;
    SchedClass sched_class;
    virtual int should_execute()=0;
private:
    virtual void create_interface()=0;
};

#endif /* SCI_project_Module_h */
