
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

#include <Classlib/String.h>
#include <Classlib/Timer.h>
#include <Multitask/Task.h>
class ModuleWidgetCallbackData;
class IPort;
class OPort;

class Module : public Task {
public:
    enum State {
	NeedData,
	Executing,
	Completed,
    };
private:
    clString name;
    int need_update;
    State state;
    double progress;
    virtual int body(int);
    WallClockTimer timer;
public:
    Module(const clString& name);
    virtual ~Module();
    Module(const Module&, int deep);
    virtual Module* clone(int deep)=0;

    // Used by NetworkEditor
    double get_progress();
    State get_state();
    clString get_name();
    double get_execute_time();
    int needs_update();
    void updated();
    int interface_initialized;
    int xpos, ypos;
    int width, height;
    int ytitle;
    int ygraphtop;
    int ygraphbot;
    int xgraphleft;
    int xgraphright;
    int ytime;
    void* drawing_a;
    ModuleWidgetCallbackData* wcbdata;

    // Used by Scheduler

    // Used by NetworkEditor and Scheduler
    int niports();
    IPort* iport(int);
    int noports();
    OPort* oport(int);

    // Used by Module subclasses
    void update_progress(double);
    void update_progress(int, int);

    // Implemented by Module subclasses
    virtual void execute()=0;
};

#endif /* SCI_project_Module_h */
