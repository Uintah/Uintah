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
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace SCIRun;

Port::Port(Module* module, const string& type_name,
	   const string& port_name, const string& color_name)
: module(module), which_port(-1), portstate(Off),
  type_name(type_name), port_name(port_name), color_name(color_name)
{
}

int 
Port::num_unblocked_connections()
{
  int count = 0;
  std::vector<Connection*>::iterator iter = connections.begin();
  while (iter != connections.end()) {
    Connection *c = *iter;
    ++iter;
    if (! c->is_blocked()) ++count;
  }
  return count;
}

int Port::nconnections()
{
  return connections.size();
}

Module* Port::get_module()
{
  return module;
}

int Port::get_which_port()
{
  return which_port;
}

void Port::attach(Connection* conn)
{
  connections.push_back(conn);
  module->connection(Connected, which_port, this==conn->oport);
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
  module->connection(Disconnected, which_port, this==conn->oport);
}

Connection* Port::connection(int i)
{
  return connections[i];
}

IPort::IPort(Module* module, const string& type_name,
	     const string& port_name, const string& color_name)
  : Port(module, type_name, port_name, color_name)
{
}

OPort::OPort(Module* module, const string& type_name,
	     const string& port_name, const string& color_name)
  : Port(module, type_name, port_name, color_name)
{
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
    module->getGui()->execute(str);
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
    module->getGui()->execute(str);
}

void Port::turn_on(PortState st)
{
    portstate=st;
    update_light();
}

void Port::turn_off()
{
    portstate=Off;
    update_light();
}

string Port::get_typename()
{
  return type_name;
}

string Port::get_portname()
{
  return port_name;
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
  return color_name;
}
