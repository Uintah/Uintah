/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

using namespace std;
using namespace SCIRun;

CCAComponentInstance::CCAComponentInstance(SCIRunFramework* framework,
					   const std::string& instanceName,
					   const std::string& typeName,
					   const sci::cca::TypeMap::pointer& /*properties*/,
					   const sci::cca::Component::pointer& component
)
  : ComponentInstance(framework, instanceName, typeName), component(component)
{
  mutex=new Mutex("getPort mutex");
}

CCAComponentInstance::~CCAComponentInstance()
{
  delete mutex;
}

PortInstance*
CCAComponentInstance::getPortInstance(const std::string& portname)
{
  map<string, CCAPortInstance*>::iterator iter = ports.find(portname);
  if(iter == ports.end())
    return 0;
  else
    return iter->second;
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
  if(!svc.isNull()){
    return svc;
  }
  map<string, CCAPortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end())
    return sci::cca::Port::pointer(0);
  CCAPortInstance* pr = iter->second;
  if(pr->porttype != CCAPortInstance::Uses)
    throw CCAException("Cannot call getPort on a Provides port");
  pr->incrementUseCount();
  if(pr->connections.size() != 1)
    return sci::cca::Port::pointer(0); 
  CCAPortInstance *pi=dynamic_cast<CCAPortInstance*> (pr->getPeer());
  return pi->port;
}

void CCAComponentInstance::releasePort(const std::string& name)
{
  if(framework->releaseFrameworkService(name, instanceName))
    return;

  map<string, CCAPortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end()){
    cerr << "Released an unknown port: " << name << '\n';
    throw CCAException("Released an unknown port: "+name);
  }

  CCAPortInstance* pr = iter->second;
  if(pr->porttype != CCAPortInstance::Uses)
    throw CCAException("Cannot call releasePort on a Provides port");

  if(!pr->decrementUseCount())
    throw CCAException("Port released without correspond get");
}

sci::cca::TypeMap::pointer CCAComponentInstance::createTypeMap()
{
  return sci::cca::TypeMap::pointer(new TypeMap);
}

void CCAComponentInstance::registerUsesPort(const std::string& portName,
					    const std::string& portType,
					    const sci::cca::TypeMap::pointer& properties)
{
  map<string, CCAPortInstance*>::iterator iter = ports.find(portName);
  if(iter != ports.end()){
    if(iter->second->porttype == CCAPortInstance::Provides)
      throw CCAException("name conflict between uses and provides ports");
    else {
      cerr << "registerUsesPort called twice, instance=" << instanceName << ", portName = " << portName << ", portType = " << portType << '\n';
      throw CCAException("registerUsesPort called twice");
    }
  }
  ports.insert(make_pair(portName, new CCAPortInstance(portName, portType, properties, CCAPortInstance::Uses)));
}

void CCAComponentInstance::unregisterUsesPort(const std::string& portName)
{
  map<string, CCAPortInstance*>::iterator iter = ports.find(portName);
  if(iter != ports.end()){
    if(iter->second->porttype == CCAPortInstance::Provides)
      throw CCAException("name conflict between uses and provides ports");
    else {
      ports.erase(portName);
    }
  }
  else{
    cerr<<"port name not found, unregisterUsesPort not done\n";
    throw CCAException("port name not found");
  }
}
void CCAComponentInstance::addProvidesPort(const sci::cca::Port::pointer& port,
					   const std::string& portName,
					   const std::string& portType,
					   const sci::cca::TypeMap::pointer& properties)
{
  map<string, CCAPortInstance*>::iterator iter = ports.find(portName);
  if(iter != ports.end()){
    if(iter->second->porttype == CCAPortInstance::Provides)
      throw CCAException("name conflict between uses and provides ports");
    else
      throw CCAException("addProvidesPort called twice");
  }
  ports.insert(make_pair(portName, new CCAPortInstance(portName, portType, properties, port, CCAPortInstance::Provides)));
}

void CCAComponentInstance::removeProvidesPort(const std::string& name)
{
  cerr << "removeProvidesPort not done, name=" << name << '\n';
}

sci::cca::TypeMap::pointer CCAComponentInstance::getPortProperties(const std::string& portName)
{
  cerr << "getPortProperties not done, name=" << portName << '\n';
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
