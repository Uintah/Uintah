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
#include <SCIRun/Vtk/VtkComponentInstance.h>
#include <iostream>

using namespace SCIRun;
using namespace vtk;

VtkPortInstance::VtkPortInstance(VtkComponentInstance* ci,
                                 vtk::Port* port, PortType porttype)
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

std::string VtkPortInstance::getUniqueName()
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

  port->addConnectedPort(peer->port);
  peer->port->addConnectedPort(port);

  nConnections++;
  peer->nConnections++;
  
  //  Network* net = port->get_module()->getNetwork();
  if(porttype == Output){
    ((vtk::InPort*)peer->port)->connect((vtk::OutPort*)port);
    port->update(Port::REFRESH);
    //    ((vtk::OutPort*)port)->setOutput(((vtk::OutPort*)port)->getOutput());
  } else {
    ((vtk::InPort*)port)->connect((vtk::OutPort*)peer->port);
    port->update(Port::REFRESH);
  }
  return true;
}

bool VtkPortInstance::disconnect(PortInstance*)
{
  //TODO: need decrement nConnections for both instances.
  //  nConnections++;
  //  peer->nConnections++;
  /*  if(porttype == Output){
    ((vtk::InPort*)peer->port)->disconnect((vtk::OutPort*)port);
    port->update();
    //    ((vtk::OutPort*)port)->setOutput(((vtk::OutPort*)port)->getOutput());
  } else {
    ((vtk::InPort*)port)->connect((vtk::OutPort*)peer->port);
    port->update(); //((vtk::OutPort*)peer->port)->setOutput(((vtk::OutPort*)peer->port)->getOutput());
  }
  */
  
// need setOutput(0) ?
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
    vtk::InPort* inport = (vtk::InPort*)port;
    vtk::OutPort* outport =(vtk::OutPort*)(((VtkPortInstance*)to)->port);
    if(nConnections>1) return false; 
    if(inport->accept(outport)) return true;
  }else{
    if(((VtkPortInstance*)to)->porttype ==Output) return false;
    // Input port does not allow multiple connections.
    vtk::OutPort* outport = (vtk::OutPort*)port;
    vtk::InPort* inport =(vtk::InPort*)(((VtkPortInstance*)to)->port);
    if(((VtkPortInstance*)to)->nConnections>1) return false; 
    if(inport->accept(outport)) return true;
  }
  return false;
}


