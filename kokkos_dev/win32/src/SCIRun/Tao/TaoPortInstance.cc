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
 *  TaoPortInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Tao/TaoPortInstance.h>
#include <SCIRun/TypeMap.h>
#include <iostream>

namespace SCIRun {

TaoPortInstance::TaoPortInstance(const std::string& name,
                                 const std::string& type,
                                 PortType porttype)
  : name(name), type(type), porttype(porttype),
    useCount(0)
{
}

TaoPortInstance::TaoPortInstance(const std::string& name,
                                 const std::string& type,
                                 const sci::cca::Port::pointer& port,
                                 PortType porttype)
  : name(name), type(type), port(port),
    porttype(porttype), useCount(0)
{
}

TaoPortInstance::~TaoPortInstance()
{
}

bool TaoPortInstance::connect(PortInstance* to)
{
  if (!canConnectTo(to)) {
    return false;
  }
  //TaoPortInstance* p2 = dynamic_cast<TaoPortInstance*>(to);
  PortInstance* p2 = to;
  if (!p2) {
    return false;
  }

  if (portType() == From && p2->portType() == To) {
    connections.push_back(p2);
  } else {
      p2->connect(this);
  }
  return true;
}

PortInstance::PortType TaoPortInstance::portType()
{
    if (porttype == Uses) {
        return From;
    } else {
        return To;
    }
}


std::string TaoPortInstance::getType()
{
    return type;
}

std::string TaoPortInstance::getModel()
{
    return "tao";
}


std::string TaoPortInstance::getUniqueName()
{
    // Tao names are already guaranteed to be unique
    return name;
}

bool TaoPortInstance::disconnect(PortInstance* to)
{
  TaoPortInstance* p2 = dynamic_cast<TaoPortInstance*>(to);
  if (!p2) {
    return false;
  }

  if (porttype != Uses) {
    std::cerr<<"disconnect can be called only by user"<<std::endl; 
    return false;
  } 
  std::vector<PortInstance*>::iterator iter;
  for (iter=connections.begin(); iter<connections.end();iter++) {
    if (p2==(*iter)) {
      connections.erase(iter);
      return true;
    }
  }
  return false;
}

// Tao spec:
// n PROVIDES (To) : 1 USES (From)
// connections: vector of PortInstances...
// connect should fail for invalid components,
//   nonexistent ports
//   other???
bool TaoPortInstance::canConnectTo(PortInstance* to)
{
  //TaoPortInstance* p2 = dynamic_cast<TaoPortInstance*>(to);
  PortInstance* p2 = to;
  if (p2 && getType() == p2->getType() && portType() != p2->portType()) {
      if (available() && p2->available()) { return true; }
  }
  std::cerr << "TaoPortInstance::canConnectTo: can't connect" << std::endl;
  return false;
}

bool TaoPortInstance::available()
{
    return portType() == To || connections.size() == 0;
}

// return a PortInstance on the other side of the connection
// -- the USES (From) port
// called by: TaoComponentInstance::getPortNonblocking, framework::Services_impl::getPortNonblocking
PortInstance* TaoPortInstance::getPeer()
{
    return connections[0];
}

std::string TaoPortInstance::getName()
{
    return name;
}

void TaoPortInstance::incrementUseCount()
{
    useCount++;
}

bool TaoPortInstance::decrementUseCount()
{
    if (useCount <= 0) {
        return false;
    }
    useCount--;
    return true;
}

} // end namespace SCIRun
