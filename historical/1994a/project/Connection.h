
/*
 *  Connection.h: A Connection between two modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Connection_h
#define SCI_project_Connection_h 1

#include <X11/Xlib.h>
class CallbackData;
class DrawingAreaC;
class IPort;
class Module;
class NetworkEditor;
class OPort;

class Connection {
    GC gc;
    DrawingAreaC* drawing_a[5];
    NetworkEditor* netedit;
    int connected;
    void redraw(CallbackData*, void*);
    void calc_portwindow_size(int, int&, int&, int&, int&);
public:
    Connection(Module*, int, Module*, int);
    ~Connection();
    void attach(OPort*);
    void attach(IPort*);

    OPort* oport;
    IPort* iport;
    int local;

    void wait_ready();

    void set_context(NetworkEditor*);
    void connect();
    void move();
};

#endif /* SCI_project_Connection_h */
