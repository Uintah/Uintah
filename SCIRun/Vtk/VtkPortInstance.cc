/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  VtkPortInstance.cc: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <SCIRun/Vtk/VtkPortInstance.h>
#include <SCIRun/Vtk/Port.h>
#include <iostream>
using namespace SCIRun;
using namespace vtk;
using namespace std;

VtkPortInstance::VtkPortInstance(VtkComponentInstance* ci,
				       Port* port, PortType porttype)
  : ci(ci), port(port), porttype(porttype)
{
}

VtkPortInstance::~VtkPortInstance()
{
}

string VtkPortInstance::getUniqueName()
{
  return port->getName();
}

PortInstance::PortType VtkPortInstance::portType()
{
  if(porttype == Output)
    return PortInstance::From;
  else
    return PortInstance::To;
}

bool VtkPortInstance::connect(PortInstance* to)
{
  // TODO: build the connection, do we really need a separate 
  // data struture? I guess the framework/component/port structure already 
  // record the connections.
  if(!canConnectTo(to))
    return false;
  // VtkPortInstance* p2 = dynamic_cast<VtkPortInstance*>(to);

  //  Network* net = port->get_module()->getNetwork();
  if(porttype == Output){
    //    net->connect(port->get_module(), port->get_which_port(),
    //		 p2->port->get_module(), p2->port->get_which_port());
  } else {
    //net->connect(p2->port->get_module(), p2->port->get_which_port(),
    //		 port->get_module(), port->get_which_port());
  }
  return true;
}

bool VtkPortInstance::disconnect(PortInstance*)
{
  //TODO:
  return false;
}

bool VtkPortInstance::canConnectTo(PortInstance *to)
{
  // TODO: use Port's interface to decide if can be connected. 
  //VtkPortInstance* p2 = dynamic_cast<VtkPortInstance*>(to);
 
  return true;/*
  if(porttype == Input){
    if(to->porttype ==Input) return false;
    

  }else{
    if(to->porttype ==Output) return false;

  }

  // if( p2 && porttype != p2->porttype &&
  //    port->get_typename() == p2->port->get_typename()){
  //  if(porttype == Input && port->nconnections() > 0)
  //    return false;
  //  if(p2->porttype == Input && p2->port->nconnections() > 0)
  //    return false;
  //  return true;
  //}
  return false;*/
}
