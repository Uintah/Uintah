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
#include <SCIRun/TypeMap.h>
#include <iostream>

namespace SCIRun {

CCAPortInstance::CCAPortInstance(const std::string& name,
                                 const std::string& type,
                                 const sci::cca::TypeMap::pointer& properties,
                                 PortType porttype)
  : name(name), type(type), properties(properties), 
    lock_connections("CCAPortInstance::connections lock"), 
    porttype(porttype), useCount(0)
{
}

CCAPortInstance::CCAPortInstance(const std::string& name,
                                 const std::string& type,
                                 const sci::cca::TypeMap::pointer& properties,
                                 const sci::cca::Port::pointer& port,
                                 PortType porttype)
  : name(name), type(type), properties(properties), port(port),
    lock_connections("CCAPortInstance::connections lock"),
    porttype(porttype), useCount(0)
{
}

CCAPortInstance::~CCAPortInstance()
{
}

bool CCAPortInstance::connect(PortInstance* to)
{
  if (!canConnectTo(to)) {
    return false;
  }
  //CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
  PortInstance* p2 = to;
  if (!p2) {
    return false;
  }

  if (portType() == From && p2->portType() == To) {
    lock_connections.lock();
    connections.push_back(p2);
    lock_connections.unlock();
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
  SCIRun::Guard g1(&lock_connections);
  for (iter=connections.begin(); iter<connections.end();iter++) {
    if (p2==(*iter)) {
      connections.erase(iter);
      return true;
    }
  }
  return false;
}

/**
 * Allowing (according to the CCA spec) n PROVIDES (To) : 1 USES (From)
 * connections is framework-implementation dependent;
 * this is \em not allowed by SCIRun2.
 * The SCIRun2 framework allows 1 PROVIDES (To) : 1 USES (From) and
 * 1 PROVIDES (To) : n USES (From) connections.
 */

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

/**
 * Available either if this is a PROVIDES port or
 * a USES port that isn't connected.
 */
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
