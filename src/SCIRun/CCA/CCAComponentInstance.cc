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

#include <SCIRun/TypeMap.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAComponentInstance.h>
#include <SCIRun/CCA/CCAPortInstance.h>
#include <SCIRun/CCA/CCAException.h>
#include <Core/Thread/Mutex.h>
#include <iostream>

#include <Core/Util/NotFinished.h>

namespace SCIRun {

CCAComponentInstance::CCAComponentInstance(
    SCIRunFramework* framework,
    const std::string &instanceName,
    const std::string &typeName,
    const sci::cca::TypeMap::pointer &tm,
    const sci::cca::Component::pointer &component)
: ComponentInstance(framework, instanceName, typeName, tm),
  lock_ports("CCAComponentInstance::ports lock"),
  lock_preports("CCAComponentInstance::preports lock"),
  component(component), size(0), rank(0)
{
    mutex = new Mutex("getPort mutex");
}

CCAComponentInstance::~CCAComponentInstance()
{
    delete mutex;
}

PortInstance*
CCAComponentInstance::getPortInstance(const std::string& portname)
{
    SCIRun::Guard g1(&lock_ports);
    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(portname);
    if (iter == ports.end()) {
        return 0;
    } else {
        return iter->second;
    }
}

// Note: exceptions are not currently thrown for network errors and
//       out of memory errors.
sci::cca::Port::pointer
CCAComponentInstance::getPort(const std::string& name)
{
    mutex->lock();
    // search framework's service collection
    sci::cca::Port::pointer svc =
        framework->getFrameworkService(name, instanceName);
    if (! svc.isNull()) {
        mutex->unlock();
        return svc;
    }

    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(name);
    if (iter == ports.end()) {
        mutex->unlock();
        throw sci::cca::CCAException::pointer(
            new CCAException("Port " + name + " not registered",
                sci::cca::PortNotDefined));
    }
    CCAPortInstance* pr = iter->second;
    if (pr->porttype != CCAPortInstance::Uses) {
        mutex->unlock();
        throw sci::cca::CCAException::pointer(
            new CCAException("Cannot call getPort on a Provides port",
                sci::cca::BadPortName));
    }
    if (pr->connections.size() != 1) {
        mutex->unlock();
        throw sci::cca::CCAException::pointer(
            new CCAException("Port " + name + " not connected",
                sci::cca::PortNotConnected));
    }
    pr->incrementUseCount();
    CCAPortInstance *pi = dynamic_cast<CCAPortInstance*>(pr->getPeer());
    mutex->unlock();
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
    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(name);
    lock_ports.unlock(); 
    if (iter == ports.end()) {
        throw sci::cca::CCAException::pointer(
            new CCAException("Port " + name + " has not been registered.", sci::cca::PortNotDefined));
    }
    CCAPortInstance* pr = iter->second;
    if (pr->porttype != CCAPortInstance::Uses) {
        throw sci::cca::CCAException::pointer(new CCAException("Cannot call getPort on a Provides port", sci::cca::BadPortType));
    }
    // registered, but not yet connected
    if (pr->connections.size() != 1) {
        return sci::cca::Port::pointer(0);
    }
    pr->incrementUseCount();
    CCAPortInstance *pi = dynamic_cast<CCAPortInstance*>(pr->getPeer());
    return pi->port;
}

void CCAComponentInstance::releasePort(const std::string& name)
{
    if (framework->releaseFrameworkService(name, instanceName)) {
        return;
    }

    lock_ports.lock();
    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(name);
    lock_ports.unlock();
    if (iter == ports.end()) {
        throw sci::cca::CCAException::pointer(
            new CCAException("Released an unknown port: " + name, sci::cca::PortNotDefined));
    }

    CCAPortInstance* pr = iter->second;
    if (pr->porttype == CCAPortInstance::Provides) {
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
    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(portName);
    if (iter != ports.end()) {
        if (iter->second->porttype == CCAPortInstance::Provides) {
            throw sci::cca::CCAException::pointer(new CCAException("name conflict between uses and provides ports for " + portName, sci::cca::PortAlreadyDefined));
        } else {
            throw sci::cca::CCAException::pointer(new CCAException("registerUsesPort called twice for " + portName + " " + portType + " " + instanceName, sci::cca::PortAlreadyDefined));
        }
    }
    ports.insert(std::make_pair(portName, new CCAPortInstance(portName, portType, properties, CCAPortInstance::Uses)));
}

// TODO: Unregistering a port that is currently in use (i.e. an unreleased getPort() being outstanding) is an error
void CCAComponentInstance::unregisterUsesPort(const std::string& portName)
{
    SCIRun::Guard g1(&lock_ports);
    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(portName);
    if (iter != ports.end()) {
        CCAPortInstance *pi = iter->second;
        if (pi->porttype == CCAPortInstance::Provides) {
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
            new CCAException("port name not found for " +
                portName, sci::cca:: PortNotDefined));
    }
}

void CCAComponentInstance::addProvidesPort(const sci::cca::Port::pointer& port,
                       const std::string& portName,
                       const std::string& portType,
                       const sci::cca::TypeMap::pointer& properties)
{
  lock_ports.lock();
  std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(portName);
  lock_ports.unlock();
  if (iter != ports.end()) {
    if (iter->second->porttype == CCAPortInstance::Uses) {
        throw sci::cca::CCAException::pointer(new CCAException("name conflict between uses and provides ports for " + portName));
    } else {
        throw sci::cca::CCAException::pointer(new CCAException("addProvidesPort called twice for " + portName));
    }
  }
  if (!properties.isNull() && properties->getInt("size", 1) > 1) {
    //if port is collective.
    size = properties->getInt("size", 1);
    rank = properties->getInt("rank", 0);
    
    mutex->lock();
   
    lock_preports.lock(); 
    std::map<std::string, std::vector<Object::pointer> >::iterator iter
      = preports.find(portName);
    lock_preports.unlock();
    if (iter==preports.end()){
      if (port.isNull()) std::cerr<<"port is NULL\n";
      
      //new preport
      std::vector<Object::pointer> urls(size);
      lock_preports.lock();
      preports[portName]=urls;
      //      preports[portName][rank]=port->getURL();
      preports[portName][rank]=port;
      lock_preports.unlock();
      precnt[portName]=0;
      precond[portName]=new ConditionVariable("precond");
    } else {
      //existed preport  
      iter->second[rank]=port;
    }
    if (++precnt[portName]==size){
      //all member ports have arrived.
      Object::pointer obj=PIDL::objectFrom(preports[portName],1,0);
      sci::cca::Port::pointer cport=pidl_cast<sci::cca::Port::pointer>(obj);
      lock_ports.lock();
      ports.insert(std::make_pair(portName, new CCAPortInstance(portName, portType, properties, cport, CCAPortInstance::Provides)));
      lock_ports.unlock();
      lock_preports.lock();
      preports.erase(portName);
      lock_preports.unlock();
      precond[portName]->conditionBroadcast();
      precnt[portName]--;
    } else {
      precond[portName]->wait(*mutex);
      if (--precnt[portName]==0){
        precnt.erase(portName);
        delete precond[portName];
        precond.erase(portName);
      }
    }
    mutex->unlock();
    return;
  } else {
    lock_ports.lock();
    ports.insert(std::make_pair(portName, new CCAPortInstance(portName, portType, properties, port, CCAPortInstance::Provides)));
    lock_ports.unlock();
  }
}

// should throw CCAException of type 'PortNotDefined'
void CCAComponentInstance::removeProvidesPort(const std::string& name)
{
    if (size < 1) {
        lock_ports.lock();
        std::map<std::string, CCAPortInstance*>::iterator iter =
            ports.find(name);
        lock_ports.unlock();
        if (iter == ports.end()) { // port can't be found
            throw sci::cca::CCAException::pointer(new CCAException("Port " + name + " is not defined.", sci::cca::PortNotDefined));
        }

        // check if port is in use???
        CCAPortInstance *pi = iter->second;
        //delete iter->second;
        lock_ports.lock();
        ports.erase(iter);
        delete pi;
        lock_ports.unlock();
    } else { // don't handle parallel ports for now
        std::cerr << "CCAComponentInstance::removeProvidesPort is not implemented: name="
                  << name << std::endl;
    }
}

sci::cca::TypeMap::pointer
CCAComponentInstance::getPortProperties(const std::string& portName)
{ 
  NOT_FINISHED("sci::cca::TypeMap::pointer CCAComponentInstance::getPortProperties(const std::string& portName)");
  return sci::cca::TypeMap::pointer(0);
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


// TODO: implement registerForRelease
void
CCAComponentInstance::registerForRelease(const sci::cca::ComponentRelease::pointer &compRel)
{
    releaseCallback = compRel;
}


CCAComponentInstance::Iterator::Iterator(CCAComponentInstance* comp)
  :iter(comp->ports.begin()), comp(comp)
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
