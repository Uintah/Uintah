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
 *  CCAPortInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/CCA/CCAPortInstance.h>
#include <SCIRun/Internal/PropertyKeys.h>
#include <SCIRun/TypeMap.h>
#include <iostream>

namespace SCIRun {

CCAPortInstance::CCAPortInstance(const std::string& name,
                                 const std::string& type,
                                 const sci::cca::TypeMap::pointer& properties,
                                 PortType porttype)
  : name(name), type(type), properties(properties), porttype(porttype),
    useCount(0)
{
// std::cerr << "CCAPortInstance::CCAPortInstance: 4 args, " << name << ", " << type << std::endl;
//     if (properties.isNull()) {
//         this->properties = sci::cca::TypeMap::pointer(new TypeMap);
//         properties->putInt(PORT_MAX_CONNECTIONS, 1);
//         properties->putInt(PORT_MIN_CONNECTIONS, 0);
//         properties->putBool(PORT_ABLE_TO_PROXY, false);
//     } else {
//         //check for default properties -- see cca.sidl
//         if (properties->getInt(PORT_MAX_CONNECTIONS, -1) < 0) {
//             properties->putInt(PORT_MAX_CONNECTIONS, 1);
//         }

//         if (properties->getInt(PORT_MIN_CONNECTIONS, -1) < 0) {
//             properties->putInt(PORT_MIN_CONNECTIONS, 0);
//         }

//         if (properties->getBool(PORT_ABLE_TO_PROXY, false) == false) {
//             // set false again???
//             properties->putBool(PORT_ABLE_TO_PROXY, false);
//         }
//     }
}

CCAPortInstance::CCAPortInstance(const std::string& name,
                                 const std::string& type,
                                 const sci::cca::TypeMap::pointer& properties,
                                 const sci::cca::Port::pointer& port,
                                 PortType porttype)
  : name(name), type(type), properties(properties), port(port),
    porttype(porttype), useCount(0)
{
  std::cerr << "CCAPortInstance::CCAPortInstance: 5 args, " << name << ", " << type;
  if (Uses == porttype) { std::cerr << ", USES" << std::endl; }
  else { std::cerr << ", PROVIDES" << std::endl; }
}

CCAPortInstance::~CCAPortInstance()
{
}

bool CCAPortInstance::connect(PortInstance* to)
{
// test code
// CCAPortInstance* p4 = dynamic_cast<CCAPortInstance*>(to);
// if (p4) {
// std::cerr << "CCAPortInstance::connect: " << name << " to " << p4->getName() << std::endl;
// }
// test code
  if (!canConnectTo(to)) {
    return false;
  }
  //CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
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

PortInstance::PortType CCAPortInstance::portType()
{
    if (porttype == Uses) {
        return From;
    } else {
        return To;
    }
}


std::string CCAPortInstance::getType()
{
    return type;
}

std::string CCAPortInstance::getModel()
{
    return "cca";
}


std::string CCAPortInstance::getUniqueName()
{
    // CCA names are already guaranteed to be unique
    return name;
}

bool CCAPortInstance::disconnect(PortInstance* to)
{
  CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
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

// CCA spec:
// n PROVIDES (To) : 1 USES (From)
// connections: vector of PortInstances...
// connect should fail for invalid components,
//   nonexistent ports
//   other???
bool CCAPortInstance::canConnectTo(PortInstance* to)
{
  //CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
  PortInstance* p2 = to;
  if (p2 && getType() == p2->getType() && portType() != p2->portType()) {
      if (available() && p2->available()) { return true; }
  }
std::cerr << "CCAPortInstance::canConnectTo: can't connect" << std::endl;
  return false;
}

bool CCAPortInstance::available()
{
    return portType() == To || connections.size() == 0;
}

// return a PortInstance on the other side of the connection
// -- the USES (From) port
// called by: CCAComponentInstance::getPortNonblocking, framework::Services_impl::getPortNonblocking
PortInstance* CCAPortInstance::getPeer()
{
    return connections[0];
}

std::string CCAPortInstance::getName()
{
    return name;
}

void CCAPortInstance::incrementUseCount()
{
    useCount++;
}

bool CCAPortInstance::decrementUseCount()
{
    if (useCount <= 0) {
        return false;
    }
    useCount--;
    return true;
}

} // end namespace SCIRun
