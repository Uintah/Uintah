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

  using namespace sci::cca::internal;
  using namespace sci::cca::internal::cca;

CCAPortInstance::CCAPortInstance(const std::string& name,
                                 const std::string& type,
                                 const sci::cca::TypeMap::pointer& properties,
                                 PortUsage port_usage)
  : name(name), type(type), properties(properties), 
    lock_connections("CCAPortInstance::connections lock"), 
    port_usage(port_usage), useCount(0)
{
}

CCAPortInstance::CCAPortInstance(const std::string& name,
                                 const std::string& type,
                                 const sci::cca::TypeMap::pointer& properties,
                                 const sci::cca::Port::pointer& port,
                                 PortUsage port_usage)
  : name(name), type(type), properties(properties), 
    lock_connections("CCAPortInstance::connections lock"),
    port(port), port_usage(port_usage), useCount(0)
{
}

CCAPortInstance::~CCAPortInstance()
{
}

bool CCAPortInstance::connect(const PortInstance::pointer &to)
{
  if (!canConnectTo(to)) {
    return false;
  }
  //CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
  PortInstance::pointer p2 = to;
  if (p2.isNull()) {
    return false;
  }

  if (portType() == From && p2->portType() == To) {
    lock_connections.lock();
    connections.push_back(p2);
    lock_connections.unlock();
  } else {
      p2->connect(PortInstance::pointer(this));
  }
  return true;
}

PortInstance::PortType CCAPortInstance::portType()
{
  return port_usage == cca::Uses ? From : To;
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

bool CCAPortInstance::disconnect(const PortInstance::pointer &to)
{
  //CCAPortInstance::pointer p2 = pidl_cast<CCAPortInstance::pointer>(to);
  PortInstance::pointer p2 = to;
  if (p2.isNull()) {
    return false;
  }

  if (port_usage != Uses) {
    std::cerr<<"disconnect can be called only by user"<<std::endl; 
    return false;
  } 
  std::vector<PortInstance::pointer>::iterator iter;
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
bool CCAPortInstance::canConnectTo(const PortInstance::pointer &to)
{
  //CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
  PortInstance::pointer p2 = to;
  if (!p2.isNull() && getType() == p2->getType() && portType() != p2->portType()) {
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
PortInstance::pointer CCAPortInstance::getPeer()
{
    return connections[0];
}

int CCAPortInstance::numOfConnections()
{
  return connections.size();
}

PortUsage CCAPortInstance::portUsage()
{
    return port_usage;
}

sci::cca::Port::pointer CCAPortInstance::getPort()
{
  return port;
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
