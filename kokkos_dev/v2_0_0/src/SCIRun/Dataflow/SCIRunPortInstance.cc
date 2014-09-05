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
 *  SCIRunPortInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Dataflow/SCIRunPortInstance.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/Port.h>
#include <iostream>
using namespace SCIRun;
using namespace std;

SCIRunPortInstance::SCIRunPortInstance(SCIRunComponentInstance* component,
				       Port* port, PortType porttype)
  : component(component), port(port), porttype(porttype)
{
}

SCIRunPortInstance::~SCIRunPortInstance()
{
}

string SCIRunPortInstance::getUniqueName()
{
  return (porttype==Input?"Input: ":"Output: ")+port->get_portname();
}

PortInstance::PortType SCIRunPortInstance::portType()
{
  if(porttype == Output)
    return PortInstance::From;
  else
    return PortInstance::To;
}

bool SCIRunPortInstance::connect(PortInstance* to)
{
  // TODO - follow inheritance somehow...
  if(!canConnectTo(to))
    return false;
  SCIRunPortInstance* p2 = dynamic_cast<SCIRunPortInstance*>(to);

  Network* net = port->get_module()->getNetwork();
  if(porttype == Output){
    net->connect(port->get_module(), port->get_which_port(),
		 p2->port->get_module(), p2->port->get_which_port());
  } else {
    net->connect(p2->port->get_module(), p2->port->get_which_port(),
		 port->get_module(), port->get_which_port());
  }
  return true;
}

bool SCIRunPortInstance::disconnect(PortInstance*)
{
  cerr << "SCIRunPortInstance::disconnect() not finished\n";
  return false;
}

bool SCIRunPortInstance::canConnectTo(PortInstance *to)
{
  // TODO - follow inheritance somehow...
  SCIRunPortInstance* p2 = dynamic_cast<SCIRunPortInstance*>(to);
  if( p2 && porttype != p2->porttype &&
      port->get_typename() == p2->port->get_typename()){
    if(porttype == Input && port->nconnections() > 0)
      return false;
    if(p2->porttype == Input && p2->port->nconnections() > 0)
      return false;
    return true;
  }
  return false;
}
