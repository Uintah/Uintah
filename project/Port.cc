
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

#include <Port.h>
#include <ColorManager.h>
#include <Module.h>
#include <NetworkEditor.h>
#include <XQColor.h>
#include <iostream.h>

Port::Port(Module* module, const clString& typename,
	   const clString& portname, const clString& colorname,
	   int protocols)
: module(module), typename(typename),
  portname(portname), colorname(colorname),
  protocols(protocols), u_proto(0), bgcolor(0), top_shadow(0),
  bottom_shadow(0)
{
}

IPort::IPort(Module* module, const clString& typename,
	     const clString& portname, const clString& colorname,
	     int protocols)
: Port(module, typename, portname, colorname, protocols)
{
}

OPort::OPort(Module* module, const clString& typename,
	     const clString& portname, const clString& colorname,
	     int protocols)
: Port(module, typename, portname, colorname, protocols)
{
}

void Port::attach(Connection* conn)
{
    connections.add(conn);
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
}

Connection* Port::connection(int i)
{
    return connections[i];
}

Module* Port::get_module()
{
    return module;
}

