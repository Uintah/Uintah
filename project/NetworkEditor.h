
/*
 *  NetworkEditor.h: Interface to Network Editor class from project
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_NetworkEditor_h
#define SCI_project_NetworkEditor_h 1

#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <Xm/Xm.h> // For Widget and XtPointer
class Module;
class Network;
class Scheduler;

class NetworkEditor : public Task {
    Network* net;
    Scheduler* sched;
    int initialized;
    Mutex lock;

    // The user interface...
    Widget drawing_a;
    int update_needed;
    void redraw(XtPointer);
    friend void do_redraw(Widget w, XtPointer ud, XtPointer xcbdata);
    friend void do_mod_redraw(Widget w, XtPointer ud, XtPointer);
    void module_move(Module*, XButtonEvent*, String);
    friend void do_module_move(Widget w, XButtonEvent*, String*, int*);
    void timer();
    friend void do_timer(XtPointer, XtIntervalId*);
    
    void draw_shadow(Display*, Window, GC,
		     int xmin, int ymin, int xmax, int ymax,
		     int width,
		     Pixel top_color, Pixel bot_color);
    void build_ui();
    void draw_module(Module*);
    void update_display();
    void update_module(Module*, int);
    void initialize(Module*);
public:
    NetworkEditor(Network*);
    ~NetworkEditor();
    void set_sched(Scheduler*);

    int body(int);
};

#endif
