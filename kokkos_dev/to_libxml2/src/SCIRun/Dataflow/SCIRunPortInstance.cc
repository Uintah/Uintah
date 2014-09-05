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

namespace SCIRun {

SCIRunPortInstance::SCIRunPortInstance(SCIRunComponentInstance* component,
                                       Port* port, PortType porttype)
  : component(component), port(port), porttype(porttype)
{
}

SCIRunPortInstance::~SCIRunPortInstance()
{
}

std::string SCIRunPortInstance::getUniqueName()
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
  if(porttype == Output)
    {
    net->connect(port->get_module(), port->get_which_port(),
                 p2->port->get_module(), p2->port->get_which_port());
    }
  else
    {
    net->connect(p2->port->get_module(), p2->port->get_which_port(),
                 port->get_module(), port->get_which_port());
    }
  return true;
}

bool SCIRunPortInstance::disconnect(PortInstance*)
{
  std::cerr << "SCIRunPortInstance::disconnect() not finished"
            << std::endl;
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


std::string SCIRunPortInstance::getType() {
  return (port->get_typename());
}

std::string SCIRunPortInstance::getModel() {
  return "dataflow";	
}

} // end namespace SCIRun
