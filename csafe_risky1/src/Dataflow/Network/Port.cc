//static char *id="@(#) $Id$";

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

#include <PSECore/Dataflow/Port.h>

#include <PSECore/Dataflow/Connection.h>
#include <PSECore/Dataflow/Module.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

namespace PSECore {
namespace Dataflow {

Port::Port(Module* module, const clString& type_name,
	   const clString& portname, const clString& colorname,
	   int protocols)
: type_name(type_name), portname(portname), colorname(colorname),
  protocols(protocols), u_proto(0), module(module), which_port(-1),
  portstate(Off)
{
}

IPort::IPort(Module* module, const clString& type_name,
	     const clString& portname, const clString& colorname,
	     int protocols)
: Port(module, type_name, portname, colorname, protocols)
{
}

OPort::OPort(Module* module, const clString& type_name,
	     const clString& portname, const clString& colorname,
	     int protocols)
: Port(module, type_name, portname, colorname, protocols)
{
}

void Port::attach(Connection* conn)
{
    connections.add(conn);
    module->connection(Module::Connected, which_port, this==conn->oport);
}

void Port::detach(Connection* conn)
{
    int i;
    for (i=0; i<connections.size(); i++)
	if (connections[i] == conn) break;
    if (i == connections.size()) {
	cerr << "Error: connection not found!\n";
	return;
    }
    connections.remove(i);
    module->connection(Module::Disconnected, which_port, this==conn->oport);
}

int Port::nconnections()
{
    return connections.size();
}

int Port::using_protocol()
{
    return u_proto;
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
    char* color;
    switch(portstate){
    case Resetting:
	color="blue";
	break;
    case Finishing:
	color="\"dark violet\"";
	break;
    case On:
	color="red";
	break;
    case Off:
    default:
	color="black";
	break;
    }
    char str[1000];
    sprintf(str,"%s lightIPort %d %s",module->id(),which_port,color);
    TCL::execute(str);
}

void OPort::update_light()
{
    char* color;
    switch(portstate){
    case Resetting:
	color="blue";
	break;
    case Finishing:
	color="\"dark violet\"";
	break;
    case On:
	color="red";
	break;
    case Off:
    default:
	if(have_data()){
	    color="darkslateblue";
	} else {
	    color="black";
	}
	break;
    }
    char str[1000];
    sprintf(str,"%s lightOPort %d %s",module->id(),which_port,color);
    TCL::execute(str);
}

void IPort::turn_on(PortState st)
{
    portstate=st;
    update_light();
}

void IPort::turn_off()
{
    portstate=Off;
    update_light();
}

void OPort::turn_on(PortState st)
{
    portstate=st;
    update_light();
}

void OPort::turn_off()
{
    portstate=Off;
    update_light();
}

clString Port::get_typename()
{
    return type_name;
}

clString Port::get_portname()
{
    return portname;
}

Port::~Port()
{
}

IPort::~IPort()
{
}

OPort::~OPort()
{
}


clString Port::get_colorname()
{
    return colorname;
}

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:20  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/08 02:26:42  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/17 06:38:25  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:00  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//
