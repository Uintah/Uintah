
/*
 *  Port.cc: Classes for module ports
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"

#include <Port.h>
#include <ColorManager.h>
#include <Connection.h>
#include <Module.h>
#include <ModuleShape.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <XQColor.h>
#include <Mt/DrawingArea.h>
#include <iostream.h>
extern MtXEventLoop* evl;

Port::Port(Module* module, const clString& typename,
	   const clString& portname, const clString& colorname,
	   int protocols)
: module(module), typename(typename),
  portname(portname), colorname(colorname),
  protocols(protocols), u_proto(0), bgcolor(0), top_shadow(0),
  bottom_shadow(0), drawing_a(0)
{
}

IPort::IPort(Module* module, const clString& typename,
	     const clString& portname, const clString& colorname,
	     int protocols)
: Port(module, typename, portname, colorname, protocols), port_on(0)
{
}

OPort::OPort(Module* module, const clString& typename,
	     const clString& portname, const clString& colorname,
	     int protocols)
: Port(module, typename, portname, colorname, protocols), port_on(0)
{
}

void Port::attach(Connection* conn)
{
    connections.add(conn);
    module->connection(Module::Connected, which_port, this==conn->oport);
}

int Port::nconnections()
{
    return connections.size();
}

int Port::using_protocol()
{
    return u_proto;
}

void Port::get_colors(ColorManager* cm)
{
    if(bgcolor)return;
    bgcolor=new XQColor(cm, colorname());
    top_shadow=bgcolor->top_shadow();
    bottom_shadow=bgcolor->bottom_shadow();
    port_on_color=new XQColor(cm, PORT_ON_COLOR);
    port_off_color=new XQColor(cm, PORT_OFF_COLOR);
}

Connection* Port::connection(int i)
{
    return connections[i];
}

Module* Port::get_module()
{
    return module;
}

int Port::get_which_port()
{
    return which_port;
}

void Port::set_which_port(int wp)
{
    which_port=wp;
}

void IPort::update_light()
{
    if(!drawing_a)
	return;
    evl->lock();
    Display* dpy=XtDisplay(*drawing_a);
    Drawable win=XtWindow(*drawing_a);
    if(port_on)
	XSetForeground(dpy, gc, port_on_color->pixel());
    else
	XSetForeground(dpy, gc, port_off_color->pixel());
    XFillRectangle(dpy, win, gc, xlight, ylight,
		   MODULE_PORTPAD_WIDTH, MODULE_PORTLIGHT_HEIGHT);
    evl->unlock();
}

void OPort::update_light()
{
    if(!drawing_a)
	return;
    evl->lock();
    Display* dpy=XtDisplay(*drawing_a);
    Drawable win=XtWindow(*drawing_a);
    if(port_on)
	XSetForeground(dpy, gc, port_on_color->pixel());
    else
	XSetForeground(dpy, gc, port_off_color->pixel());
    XFillRectangle(dpy, win, gc, xlight, ylight,
		   MODULE_PORTPAD_WIDTH, MODULE_PORTLIGHT_HEIGHT);
    evl->unlock();
}

void IPort::turn_on()
{
    if(!port_on){
	port_on=1;
	update_light();
    }
}

void IPort::turn_off()
{
    if(port_on){
	port_on=0;
	update_light();
    }
}

void OPort::turn_on()
{
    if(!port_on){
	port_on=1;
	update_light();
    }
}

void OPort::turn_off()
{
    if(port_on){
	port_on=0;
	update_light();
    }
}

void Port::set_context(int xlight_, int ylight_, DrawingAreaC* drawing_a_,
		       GC gc_)
{
    drawing_a=drawing_a_;
    xlight=xlight_;
    ylight=ylight_;
    gc=gc_;
}

void Port::move()
{
    for(int i=0;i<connections.size();i++){
	connections[i]->move();
    }
}

clString Port::get_typename()
{
    return typename;
}

clString Port::get_portname()
{
    return portname;
}
