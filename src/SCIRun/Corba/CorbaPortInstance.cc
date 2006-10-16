/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  CorbaPortInstance.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <SCIRun/Corba/CorbaPortInstance.h>
#include <SCIRun/Corba/Port.h>
#include <SCIRun/Corba/UsesPort.h>
#include <SCIRun/Corba/ProvidesPort.h>
#include <SCIRun/Corba/CorbaComponentInstance.h>
#include <iostream>

namespace SCIRun {

CorbaPortInstance::CorbaPortInstance(CorbaComponentInstance* ci,
                                     SCIRun::corba::Port* port,
                                     const sci::cca::TypeMap::pointer& properties,
                                     SCIRun::PortInstance::PortType porttype)
  : ci(ci), port(port), porttype(porttype), properties(properties), nConnections(0)
{
}

CorbaPortInstance::~CorbaPortInstance()
{
}

std::string
CorbaPortInstance::getModel()
{
  return "corba";
}

std::string CorbaPortInstance::getUniqueName()
{
  return port->getName();
}

SCIRun::PortInstance::PortType CorbaPortInstance::portType()
{
  return porttype;
}

bool CorbaPortInstance::connect(PortInstance* to)
{
  // TODO: build the connection, do we really need a separate
  // data struture? I guess the framework/component/port structure already
  // record the connections.
  if (!canConnectTo(to))
    return false;

  // CorbaPortInstance* p2 = dynamic_cast<CorbaPortInstance*>(to);
  CorbaPortInstance* peer = (CorbaPortInstance*)to;

  port->addConnectedPort(peer->port);
  peer->port->addConnectedPort(port);

  nConnections++;
  peer->nConnections++;

  return true;
}

bool CorbaPortInstance::disconnect(PortInstance*)
{
  //TODO: need decrement nConnections for both instances.
  //  nConnections++;
  //  peer->nConnections++;
  /*  if(porttype == Output){
    ((SCIRun::corba::InPort*)peer->port)->disconnect((SCIRun::corba::OutPort*)port);
    port->update();
    //    ((SCIRun::corba::OutPort*)port)->setOutput(((SCIRun::corba::OutPort*)port)->getOutput());
  } else {
    ((SCIRun::corba::InPort*)port)->connect((SCIRun::corba::OutPort*)peer->port);
    port->update(); //((SCIRun::corba::OutPort*)peer->port)->setOutput(((SCIRun::corba::OutPort*)peer->port)->getOutput());
  }
  */

// need setOutput(0) ?
  return false;
}

bool CorbaPortInstance::canConnectTo(PortInstance *to)
{
  //skip connections between different component models
  //particuarlly connections between UI ports (CCA) and Corba ports.
  if (getModel() != to->getModel()) {
    return false;
  }
  if (porttype == Uses) {
    if ( ( (CorbaPortInstance*)to)->porttype == Uses) {
      return false;
    }
    // Uses port does not allow multiple connections.
    if (nConnections >= 1) {
      return false;
    }
    return port->getType() == ( (CorbaPortInstance*)to)->port->getType();
  } else {
    if ( ( (CorbaPortInstance*)to)->porttype == Provides) {
      return false;
    }
    // Uses port does not allow multiple connections.
    if ( ( (CorbaPortInstance*)to)->nConnections >= 1) {
      return false;
    }
    return port->getType()==((CorbaPortInstance*)to)->port->getType();
  }
  return false;
}


std::string
CorbaPortInstance::getType()
{
  return port->getType();
}

void CorbaPortInstance::setProperties(const sci::cca::TypeMap::pointer& tm)
{
  properties = tm;
  setDefaultProperties();
}

void CorbaPortInstance::setDefaultProperties()
{
  properties->putString(PortInstance::NAME, port->getName());
  properties->putString(PortInstance::TYPE, port->getType());
  properties->putString(PortInstance::MODEL, this->getModel());
}

}
