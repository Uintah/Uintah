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
#include <SCIRun/Vtk/InPort.h>
#include <SCIRun/Vtk/OutPort.h>
#include <iostream>
using namespace SCIRun;
using namespace vtk;
using namespace std;

VtkPortInstance::VtkPortInstance(VtkComponentInstance* ci,
				       Port* port, PortType porttype)
  : ci(ci), port(port), porttype(porttype)
{
  nConnections=0;
}

VtkPortInstance::~VtkPortInstance()
{
}

std::string
VtkPortInstance::getModel(){
  return "vtk";
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
  VtkPortInstance* peer=(VtkPortInstance*)to;
  nConnections++;
  
  //  Network* net = port->get_module()->getNetwork();
  if(porttype == Output){
    ((InPort*)peer->port)->connect((OutPort*)port);
  } else {
    ((InPort*)port)->connect((OutPort*)port);
  }
  return true;
}

bool VtkPortInstance::disconnect(PortInstance*)
{
  //TODO: need decrement nConnections for both instances.
  return false;
}

bool VtkPortInstance::canConnectTo(PortInstance *to)
{
  //skip connections between different component models
  //particuarlly connections between UI ports (CCA) and Vtk ports. 
  if(getModel()!=to->getModel()) return false;
  if(porttype == Input){
    if(((VtkPortInstance*)to)->porttype ==Input) return false;
    // Input port does not allow multiple connections.
    InPort* inport = (InPort*)port;
    OutPort* outport =(OutPort*)(((VtkPortInstance*)to)->port);
    if(nConnections>1) return false; 
    if(inport->accept(outport)) return true;
  }else{
    if(((VtkPortInstance*)to)->porttype ==Output) return false;
    // Input port does not allow multiple connections.
    OutPort* outport = (OutPort*)port;
    InPort* inport =(InPort*)(((VtkPortInstance*)to)->port);
    if(((VtkPortInstance*)to)->nConnections>1) return false; 
    if(inport->accept(outport)) return true;
  }
  return false;
}
