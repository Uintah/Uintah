/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

void Port::detach(Connection* conn, bool blocked)
{
  unsigned int i;
  for (i=0; i<connections.size(); i++)
    if (connections[i] == conn) break;
  if (i == connections.size()) {
    cerr << "Error: connection not found!\n";
    return;
  }
  connections.erase(connections.begin() + i);
  module->connection(Disconnected, which_port, this==conn->oport || blocked);
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
    sprintf(str,"lightPort {%s %d i} %s",module->id.c_str(),which_port,color);
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
    sprintf(str,"lightPort {%s %d o} %s",module->id.c_str(),which_port,color);
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


void Port::synchronize()
{
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


void OPort::issue_no_port_caching_warning()
{
  // Don't really care about thread safety since the worst that happens is
  // multiple warnings are printed.  Not worth creating the lock over.
  static bool warnonce = true;
  if (warnonce)
  {
    warnonce = false;
    std::cerr << "Warning: SCIRUN_NO_PORT_CACHING on!!!\n";
    std::cerr << "Warning: This will break interactive SCIRun usage.\n";
    std::cerr << "Warning: See .scirunrc for details.\n";
  }
}
