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
  nConnections=0;
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
  VtkPortInstance* peer=(VtkPortInstance*)to;
  nConnections++;
  
  //  Network* net = port->get_module()->getNetwork();
  if(porttype == Output){
    peer->port->connect(port);
  } else {
    port->connect(port);
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
  // TODO: use Port's interface to decide if can be connected. 
  //VtkPortInstance* p2 = dynamic_cast<VtkPortInstance*>(to);
  cerr<<"#1"<<endl;
  if(porttype == Input){
    cerr<<"#2"<<endl;
    if(((VtkPortInstance*)to)->porttype ==Input) return false;
    // Input port does not allow multiple connections.
    cerr<<"#3"<<endl;
    if(nConnections>1) return false; 
    cerr<<"#4"<<endl;
    if(port->accept(((VtkPortInstance*)to)->port)) return true;
    cerr<<"#5"<<endl;
  }else{
    cerr<<"#6"<<endl;
    if(((VtkPortInstance*)to)->porttype ==Output) return false;
    cerr<<"#7"<<endl;
    // Input port does not allow multiple connections.
    if(((VtkPortInstance*)to)->nConnections>1) return false; 
    cerr<<"#8"<<endl;
    if(((VtkPortInstance*)to)->port->accept(port)) return true;
    cerr<<"#9"<<endl;
  }
  cerr<<"#10"<<endl;
  return false;
}
