
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

#include <MessageBase.h>
#include <Classlib/Array1.h>
#include <Classlib/String.h>
#include <Multitask/ITC.h>
#include <Multitask/Task.h>
class Connection;
class IPort;
class Module;
class Network;
class NetworkEditor;
class OPort;

class ModuleHelper : public Task {
    Module* module;
public:
    ModuleHelper(Module* module);
    virtual ~ModuleHelper();

    virtual int body(int);
};

class Module {
public:
    enum State {
	NeedData,
	Executing,
	Completed,
    };
private:
    ModuleHelper* helper;
protected:
    friend class ModuleHelper;
    virtual void do_execute()=0;

    clString name;
    int need_update;
    double progress;
    State state;
    Array1<OPort*> oports;
    Array1<IPort*> iports;
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

    // Port manipulations
    void add_iport(IPort*);
    void add_oport(OPort*);
    void remove_iport(int);
    void remove_oport(int);
    void rename_iport(int, const clString&);
    void rename_oport(int, const clString&);

    // Used by Module subclasses
    void update_progress(double);
    void update_progress(int, int);

    // User Interface information
    NetworkEditor* netedit;
    Network* network;
public:
    IPort* iport(int);
    OPort* oport(int);
    int xpos;
    int ypos;
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
    virtual void create_widget()=0;
};

#endif /* SCI_project_Module_h */
