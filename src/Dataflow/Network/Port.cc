/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Dataflow/Network/Port.h>

#include <Core/Parts/Part.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {

Port::Port(Module* module, const string& type_name,
	   const string& portname, const string& colorname,
	   int protocols)
: type_name(type_name), portname(portname), colorname(colorname),
  protocols(protocols), u_proto(0), module(module), which_port(-1),
  portstate(Off)
{
}

IPort::IPort(Module* module, const string& type_name,
	     const string& portname, const string& colorname,
	     int protocols)
: Port(module, type_name, portname, colorname, protocols)
{
}

OPort::OPort(Module* module, const string& type_name,
	     const string& portname, const string& colorname,
	     int protocols)
: Port(module, type_name, portname, colorname, protocols)
{
}

void Port::attach(Connection* conn)
{
    connections.push_back(conn);
    module->connection(Module::Connected, which_port, this==conn->oport);
}

void Port::detach(Connection* conn)
{
    unsigned int i;
    for (i=0; i<connections.size(); i++)
	if (connections[i] == conn) break;
    if (i == connections.size()) {
	cerr << "Error: connection not found!\n";
	return;
    }
    connections.erase(connections.begin() + i);
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
    sprintf(str,"%s lightIPort %d %s",module->id.c_str(),which_port,color);
    Part::tcl_execute(str);
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
    sprintf(str,"%s lightOPort %d %s",module->id.c_str(),which_port,color);
    Part::tcl_execute(str);
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

string Port::get_typename()
{
    return type_name;
}

void Port::set_portname(string& newname)
{
  portname = newname;
}

string Port::get_portname()
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


string Port::get_colorname()
{
    return colorname;
}

} // End namespace SCIRun

