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
 *  CCAComponentInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <Framework/TypeMap.h>
#include <Framework/SCIRunFramework.h>
#include <Framework/CCA/CCAComponentInstance.h>
#include <Framework/CCA/CCAPortInstance.h>
#include <Framework/CCA/CCAException.h>
#include <Core/Thread/Mutex.h>
#include <iostream>

#include <Core/Util/NotFinished.h>

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace SCIRun {

CCAComponentInstance::CCAComponentInstance(SCIRunFramework* framework,
                                           const std::string &instanceName,
                                           const std::string &typeName,
                                           const sci::cca::TypeMap::pointer &tm,
                                           const sci::cca::Component::pointer &component)
  : ComponentInstance(framework, instanceName, typeName, tm),
    lock_ports("CCAComponentInstance::ports lock"),
    lock_preports("CCAComponentInstance::preports lock"),
    lock_instance("CCAComponentInstance lock"),
    component(component), size(0), rank(0)
{
}

CCAComponentInstance::~CCAComponentInstance()
{
}

PortInstance*
CCAComponentInstance::getPortInstance(const std::string& portname)
{
  SCIRun::Guard g1(&lock_ports);
  CCAPortInstanceMap::iterator iter = ports.find(portname);
  if (iter == ports.end()) {
    return 0;
  } else {
    return iter->second;
  }
}

// Note: exceptions are not currently thrown for network errors and
// out of memory errors.
sci::cca::Port::pointer
CCAComponentInstance::getPort(const std::string& name)
{
#if DEBUG
  std::cerr << "CCAComponentInstance::getPort name=" << name << std::endl;
#endif

  Guard g1(&lock_instance);
  // search framework's service collection
  sci::cca::Port::pointer svc =
    framework->getFrameworkService(name, instanceName);
  if (! svc.isNull()) {
    return svc;
  }

  lock_ports.lock();
  PortInstanceMap::iterator iter = ports.find(name);
  lock_ports.unlock();
  if (iter == ports.end()) {
    throw sci::cca::CCAException::pointer(
      new CCAException("Port " + name + " not registered",
        sci::cca::PortNotDefined));
  }
  CCAPortInstance* pr = iter->second;
  if (pr->portType() == CCAPortInstance::Provides) {
    throw sci::cca::CCAException::pointer(
      new CCAException("Cannot call getPort on a Provides port", sci::cca::BadPortName));
  }
  if (pr->connections.size() != 1) {
    throw sci::cca::CCAException::pointer(
      new CCAException("Port " + name + " not connected", sci::cca::PortNotConnected));
  }
  pr->incrementUseCount();
  CCAPortInstance *pi = dynamic_cast<CCAPortInstance*>(pr->getPeer());
  return pi->port;
}

// Note: exceptions are not currently thrown for Network errors and
//       out of memory errors.
sci::cca::Port::pointer
CCAComponentInstance::getPortNonblocking(const std::string& name)
{
  sci::cca::Port::pointer svc =
    framework->getFrameworkService(name, instanceName);
  if (!svc.isNull()) {
    return svc;
  }
  lock_ports.lock();
  PortInstanceMap::iterator iter = ports.find(name);
  lock_ports.unlock();

  if (iter == ports.end()) {
    throw sci::cca::CCAException::pointer(
      new CCAException("Port " + name + " has not been registered.", sci::cca::PortNotDefined));
  }
  CCAPortInstance* pr = iter->second;
  if (pr->portType() == CCAPortInstance::Provides) {
    throw sci::cca::CCAException::pointer(
      new CCAException("Cannot call getPort on a Provides port", sci::cca::BadPortType));
  }
  // registered, but not yet connected
  if (pr->connections.size() != 1) {
    return sci::cca::Port::pointer(0);
  }

  lock_instance.lock();
  pr->incrementUseCount();
  lock_instance.unlock();
  CCAPortInstance *pi = dynamic_cast<CCAPortInstance*>(pr->getPeer());
  return pi->port;
}

void CCAComponentInstance::releasePort(const std::string& name)
{
  Guard g1(&lock_ports);
  if (framework->releaseFrameworkService(name, instanceName)) {
    return;
  }

  PortInstanceMap::iterator iter = ports.find(name);
  if (iter == ports.end()) {
    throw sci::cca::CCAException::pointer(
      new CCAException("Released an unknown port: " + name, sci::cca::PortNotDefined));
  }

  CCAPortInstance* pr = iter->second;
  if (pr->portType() == CCAPortInstance::Provides) {
    throw sci::cca::CCAException::pointer(
      new CCAException("Cannot call releasePort on a Provides port", sci::cca::PortNotDefined));
  }
  if (!pr->decrementUseCount()) {
    // negative port instance use count
    throw sci::cca::CCAException::pointer(
      new CCAException("Port released without correspond get", sci::cca::PortNotInUse));
  }
}

sci::cca::TypeMap::pointer CCAComponentInstance::createTypeMap()
{
  sci::cca::TypeMap::pointer tm(new TypeMap);
  // It is not clear why we need addReference here.
  // But removing it can cause random crash
  // when creating remote parallel components
  // TODO: possible memory leak?
  tm->addReference();
  return tm;
}

// TODO: throw OutOfMemory exception?
void CCAComponentInstance::registerUsesPort(const std::string& portName,
                                            const std::string& portType,
                                            const sci::cca::TypeMap::pointer& properties)
{
  SCIRun::Guard g1(&lock_ports);
  PortInstanceMap::iterator iter = ports.find(portName);
  if (iter != ports.end()) {
    if (iter->second->portType() == CCAPortInstance::Provides) {
      throw sci::cca::CCAException::pointer(
        new CCAException("name conflict between uses and provides ports for " + portName, sci::cca::PortAlreadyDefined));
    } else {
      throw sci::cca::CCAException::pointer(
        new CCAException("registerUsesPort called twice for " + portName + " " + portType + " " + instanceName, sci::cca::PortAlreadyDefined));
    }
  }
  ports.insert(std::make_pair(portName, new CCAPortInstance(portName, portType, properties, CCAPortInstance::Uses)));
}

void CCAComponentInstance::unregisterUsesPort(const std::string& portName)
{
  SCIRun::Guard g1(&lock_ports);
  PortInstanceMap::iterator iter = ports.find(portName);
  if (iter != ports.end()) {
    CCAPortInstance *pi = iter->second;
    if (pi->portType() == CCAPortInstance::Provides) {
      throw sci::cca::CCAException::pointer(
        new CCAException("name conflict between uses and provides ports for " +
          portName, sci::cca:: BadPortName));
    } else {
      if (pi->portInUse()) {
        throw sci::cca::CCAException::pointer(new CCAException("Uses port " +
          portName + " has not been released", sci::cca:: UsesPortNotReleased));
      }
      ports.erase(iter);
      delete pi;
    }
  } else {
    throw sci::cca::CCAException::pointer(
      new CCAException("port name not found for " + portName, sci::cca::PortNotDefined));
  }
}

void CCAComponentInstance::addProvidesPort(const sci::cca::Port::pointer& port,
                                           const std::string& portName,
                                           const std::string& portType,
                                           const sci::cca::TypeMap::pointer& properties)
{
  if (port.isNull()) {
    throw sci::cca::CCAException::pointer(
                                          new CCAException("Null port argument for " + portName + ", " + portType));
  }

  lock_ports.lock();
  PortInstanceMap::iterator iter = ports.find(portName);
  lock_ports.unlock();

  if (iter != ports.end()) {
    if (iter->second->portType() == CCAPortInstance::Uses) {
      throw sci::cca::CCAException::pointer(
        new CCAException("Name conflict between uses and provides ports for " + portName, sci::cca::PortAlreadyDefined));
    } else {
      throw sci::cca::CCAException::pointer(
        new CCAException("addProvidesPort called twice for " + portName, sci::cca::PortAlreadyDefined));
    }
  }
  if (!properties.isNull() && properties->getInt("size", 1) > 1) {
    // If port is collective.
    size = properties->getInt("size", 1);
    rank = properties->getInt("rank", 0);

    Guard g1(&lock_instance);
    PreportMap::iterator iter = preports.find(portName);
    if (iter == preports.end()) { // new preport
      std::vector<Object::pointer> urls(size);
      preports[portName] = urls;
      //preports[portName][rank] = port->getURL();
      preports[portName][rank] = port;
      precnt[portName] = 0;
      precond[portName] = new ConditionVariable("precond");
    } else { // existing preport
      iter->second[rank] = port;
    }

    if (++precnt[portName] == size) { // all member ports have arrived
      Object::pointer obj = PIDL::objectFrom(preports[portName], 1, 0);
      sci::cca::Port::pointer cport = pidl_cast<sci::cca::Port::pointer>(obj);

      lock_ports.lock();
      ports.insert(std::make_pair(portName,
                                  new CCAPortInstance(portName, portType, properties, cport, CCAPortInstance::Provides)));
      lock_ports.unlock();

      lock_preports.lock();
      preports.erase(portName);
      lock_preports.unlock();
      precond[portName]->conditionBroadcast();
      precnt[portName]--;
    } else {
      precond[portName]->wait(lock_instance);
      if (--precnt[portName] == 0) {
        precnt.erase(portName);
        delete precond[portName];
        precond.erase(portName);
      }
    }
    return;
  } else {
    Guard g2(&lock_ports);
    ports.insert(std::make_pair(portName,
                                new CCAPortInstance(portName, portType, properties, port, CCAPortInstance::Provides)));
  }
}

// should throw CCAException of type 'PortNotDefined'
void CCAComponentInstance::removeProvidesPort(const std::string& name)
{
  if (size < 1) {
#if DEBUG
    std::cerr << "CCAComponentInstance::removeProvidesPort: name="
              << name << std::endl;
#endif
    lock_ports.lock();
    PortInstanceMap::iterator iter = ports.find(name);
    lock_ports.unlock();
    if (iter == ports.end()) { // port can't be found
      throw sci::cca::CCAException::pointer(new CCAException("Port " + name + " is not defined.", sci::cca::PortNotDefined));
    }

    // check if port is in use???
    CCAPortInstance *pi = iter->second;
    //delete iter->second;
    Guard g1(&lock_ports);
    ports.erase(iter);
    delete pi;
  } else { // don't handle collective ports for now
    NOT_FINISHED("void CCAComponentInstance::removeProvidesPort(const std::string& name) for collective ports");
  }
}

sci::cca::TypeMap::pointer
CCAComponentInstance::getPortProperties(const std::string& portName)
{
  lock_ports.lock();
  PortInstanceMap::iterator iter = ports.find(portName);
  lock_ports.unlock();
  if (iter == ports.end()) {
    return sci::cca::TypeMap::pointer(new TypeMap);
  }

  return iter->second->getProperties();
}

void
CCAComponentInstance::setPortProperties(const std::string& portName, const sci::cca::TypeMap::pointer& tm)
{
  lock_ports.lock();
  PortInstanceMap::iterator iter = ports.find(portName);
  lock_ports.unlock();
  if (iter == ports.end()) {
    // with warning?
    return;
  }

  return iter->second->setProperties(tm);
}

sci::cca::ComponentID::pointer
CCAComponentInstance::getComponentID()
{
  sci::cca::ComponentID::pointer cid =
    framework->lookupComponentID(instanceName);
  if (! cid.isNull()) {
    return cid;
  }
  cid = sci::cca::ComponentID::pointer(
                                       new ComponentID(framework, instanceName));
  return cid;
}

PortInstanceIterator* CCAComponentInstance::getPorts()
{
  return new Iterator(this);
}


void
CCAComponentInstance::registerForRelease(const sci::cca::ComponentRelease::pointer &compRel)
{
  releaseCallback = compRel;
}


CCAComponentInstance::Iterator::Iterator(CCAComponentInstance* comp)
  : iter(comp->ports.begin()), comp(comp)
{
}

CCAComponentInstance::Iterator::~Iterator()
{
}

PortInstance* CCAComponentInstance::Iterator::get()
{
  return iter->second;
}

bool CCAComponentInstance::Iterator::done()
{
  return iter == comp->ports.end();
}

void CCAComponentInstance::Iterator::next()
{
  ++iter;
}

} // end namespace SCIRun
