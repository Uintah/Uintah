
/*
 *  Port.h: Classes for module ports
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Port_h
#define SCI_project_Port_h 1

#include <Classlib/Array1.h>
#include <Classlib/String.h>
#include <X11/Xlib.h>
class ColorManager;
class Connection;
class DrawingAreaC;
class Module;
class XQColor;

class Port {
    Module* module;
    int which_port;
    clString typename;
    clString portname;
    clString colorname;
    int protocols;
    int u_proto;
protected:
    Array1<Connection*> connections;
    DrawingAreaC* drawing_a;
    int xlight, ylight;
    GC gc;
public:
    Port(Module*, const clString&, const clString&,
	 const clString&, int protocols);
    void set_port(int which_port);
    void set_context(int xlight_, int ylight_, DrawingAreaC* drawing_a_,
		     GC gc_);
    int using_protocol();
    int nconnections();
    Connection* connection(int);
    Module* get_module();
    int get_which_port();
    void set_which_port(int);
    void attach(Connection*);
    void detach(Connection*);
    virtual void reset()=0;
    virtual void finish()=0;
    void get_colors(ColorManager*);
    XQColor* bgcolor;
    XQColor* top_shadow;
    XQColor* bottom_shadow;
    XQColor* port_on_color;
    XQColor* port_off_color;
    void move();
    clString get_typename();
    clString get_portname();
};

class IPort : public Port {
    int port_on;
protected:
    IPort(Module*, const clString&, const clString&,
	  const clString&, int protocols);
    void turn_on();
    void turn_off();
public:
    void update_light();
};

class OPort : public Port {
    int port_on;
protected:
    OPort(Module*, const clString&, const clString&,
	  const clString&, int protocols);
    void turn_on();
    void turn_off();
public:
    void update_light();
};

#endif /* SCI_project_Port_h */
