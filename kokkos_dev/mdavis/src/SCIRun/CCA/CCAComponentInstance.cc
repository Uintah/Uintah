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

#include <SCIRun/CCA/CCAComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/TypeMap.h>
#include <SCIRun/CCA/CCAPortInstance.h>
#include <SCIRun/CCA/CCAException.h>
#include <iostream>
#include <Core/Thread/Mutex.h>

namespace SCIRun {

CCAComponentInstance::CCAComponentInstance(SCIRunFramework* framework,
                                           const std::string& instanceName,
                                           const std::string& typeName,
                            const sci::cca::TypeMap::pointer& com_properties,
                     const sci::cca::Component::pointer& component)
  : ComponentInstance(framework, instanceName, typeName),
        component(component), size(0), rank(0)
{
  if (com_properties.isNull()) {
    std::cerr << "### Null properties for cca comp" << std::endl;
    this->com_properties=createTypeMap();
  } else {
    this->com_properties=com_properties;
  }
  mutex=new Mutex("getPort mutex");
}

CCAComponentInstance::~CCAComponentInstance()
{
    delete mutex;
}

PortInstance*
CCAComponentInstance::getPortInstance(const std::string& portname)
{
    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(portname);
    if (iter == ports.end()) {
        return 0;
    } else {
        return iter->second;
    }
}

sci::cca::Port::pointer CCAComponentInstance::getPort(const std::string& name)
{
    mutex->lock();
    sci::cca::Port::pointer port=getPortNonblocking(name);
    mutex->unlock();
    return port;
}

sci::cca::Port::pointer
CCAComponentInstance::getPortNonblocking(const std::string& name)
{
    sci::cca::Port::pointer svc = framework->getFrameworkService(name, instanceName);
    if (!svc.isNull()) {
        return svc;
    }
    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(name);
    if (iter == ports.end()) {
        return sci::cca::Port::pointer(0);
    }
    CCAPortInstance* pr = iter->second;
    if (pr->porttype != CCAPortInstance::Uses) {
        throw CCAException("Cannot call getPort on a Provides port");
    }
    pr->incrementUseCount();
    if (pr->connections.size() != 1) {
        return sci::cca::Port::pointer(0);
    }
    CCAPortInstance *pi=dynamic_cast<CCAPortInstance*>(pr->getPeer());
    return pi->port;
}

void CCAComponentInstance::releasePort(const std::string& name)
{
    if (framework->releaseFrameworkService(name, instanceName)) {
        return;
    }

    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(name);
    if (iter == ports.end()) {
        throw CCAException("Released an unknown port: " + name);
    }

    CCAPortInstance* pr = iter->second;
    if (pr->porttype == CCAPortInstance::Provides) {
        throw CCAException("Cannot call releasePort on a Provides port");
    }
    if (!pr->decrementUseCount()) {
        // negative use count
        throw CCAException("Port released without correspond get");
    }
}

sci::cca::TypeMap::pointer CCAComponentInstance::createTypeMap()
{
    sci::cca::TypeMap::pointer tm(new TypeMap);
    //TODO: it is not clear why we need addReference here. But removing it can cause random crash
    // when creating remote parallel components
    tm->addReference();
    return tm; 
}

void CCAComponentInstance::registerUsesPort(const std::string& portName,
                                            const std::string& portType,
                             const sci::cca::TypeMap::pointer& properties)
{
    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(portName);
    if (iter != ports.end()) {
        if (iter->second->porttype == CCAPortInstance::Provides) {
            throw CCAException("name conflict between uses and provides ports for " + portName + " " + portType + " " + instanceName);
        } else {
            throw CCAException("registerUsesPort called twice for " + portName + " " + portType + " " + instanceName);
        }
    }
    ports.insert(make_pair(portName, new CCAPortInstance(portName, portType, properties, CCAPortInstance::Uses)));
}

void CCAComponentInstance::unregisterUsesPort(const std::string& portName)
{
std::cerr << "CCAComponentInstance::unregisterUsesPort: " << portName << std::endl;
    std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(portName);
    if (iter != ports.end()) {
        if (iter->second->porttype == CCAPortInstance::Provides) {
            throw CCAException("name conflict between uses and provides ports for " + portName);
        } else {
            ports.erase(portName);
        }
    } else {
        throw CCAException("port name not found for " + portName);
    }
}

void CCAComponentInstance::addProvidesPort(const sci::cca::Port::pointer& port,
					   const std::string& portName,
					   const std::string& portType,
					   const sci::cca::TypeMap::pointer& properties)
{
  std::map<std::string, CCAPortInstance*>::iterator iter = ports.find(portName);
  if (iter != ports.end()) {
    if (iter->second->porttype == CCAPortInstance::Uses) {
      throw CCAException("name conflict between uses and provides ports for " + portName); }
    else {
      throw CCAException("addProvidesPort called twice for " + portName); 
    }
  }
  if (!properties.isNull() && properties->getInt("size", 1) > 1) {
    //if port is collective.
    size = properties->getInt("size", 1);
    rank = properties->getInt("rank", 0);
    
    mutex->lock();
    
    std::map<std::string, std::vector<Object::pointer> >::iterator iter
      = preports.find(portName);
    if (iter==preports.end()){
      if (port.isNull()) std::cerr<<"port is NULL\n";
      
      //new preport
      std::vector<Object::pointer> urls(size);
      preports[portName]=urls;
      //      preports[portName][rank]=port->getURL();
      preports[portName][rank]=port;
      precnt[portName]=0;
      precond[portName]=new ConditionVariable("precond");
    }
    else {
      //existed preport  
      iter->second[rank]=port;
    }
    if (++precnt[portName]==size){
      //all member ports have arrived.
      Object::pointer obj=PIDL::objectFrom(preports[portName],1,0);
      sci::cca::Port::pointer cport=pidl_cast<sci::cca::Port::pointer>(obj);
      ports.insert(make_pair(portName, new CCAPortInstance(portName, portType,
							   properties, cport, CCAPortInstance::Provides)));
      preports.erase(portName);
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
    ports.insert(make_pair(portName,
                         new CCAPortInstance(portName, portType, properties,
                                             port, CCAPortInstance::Provides)));
  }
}

// should throw CCAException of type 'PortNotDefined'
void CCAComponentInstance::removeProvidesPort(const std::string& name)
{
    if (size < 1) {
        std::cerr << "CCAComponentInstance::removeProvidesPort: name="
                  << name << std::endl;
        std::map<std::string, CCAPortInstance*>::iterator iter =
            ports.find(name);
        if (iter == ports.end()) { // port can't be found
            throw CCAException("Port " + name + " is not defined.");
        }

        // check if port is in use???
        delete iter->second;
        ports.erase(iter);
    } else { // don't handle parallel ports for now
        std::cerr << "CCAComponentInstance::removeProvidesPort is not implemented: name="
                  << name << std::endl;
    }
}

sci::cca::TypeMap::pointer
CCAComponentInstance::getPortProperties(const std::string& portName)
{ 
  if (portName=="") {
    //return component property.
    return com_properties;
  }
  std::cerr << "getPortProperties not done, name=" << portName << std::endl;
  return sci::cca::TypeMap::pointer(0);
}

sci::cca::ComponentID::pointer CCAComponentInstance::getComponentID()
{
  return sci::cca::ComponentID::pointer(new ComponentID(framework, instanceName));
}

PortInstanceIterator* CCAComponentInstance::getPorts()
{
  return new Iterator(this);
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
