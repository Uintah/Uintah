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
 *  BridgeComponentInstance.cc:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September, 2003
 *
 */

#include <SCIRun/Bridge/BridgeComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/TypeMap.h>
#include <SCIRun/CCA/CCAException.h>
#include <iostream>
#include <Core/Thread/Mutex.h>
#include <Core/Exceptions/InternalError.h>

using namespace std;
using namespace SCIRun;

BridgeComponentInstance::BridgeComponentInstance(SCIRunFramework* framework,
					   const std::string& instanceName,
					   const std::string& typeName,
					   BridgeComponent* component
)
  : ComponentInstance(framework, instanceName, typeName), component(component)
{
  mutex=new Mutex("getPort mutex");
}

BridgeComponentInstance::~BridgeComponentInstance()
{
  delete mutex;
}

PortInstance*
BridgeComponentInstance::getPortInstance(const std::string& portname)
{
  map<string, PortInstance*>::iterator iter = ports.find(portname);
  if(iter == ports.end())
    return 0;
  else
    return iter->second;
}

sci::cca::Port::pointer BridgeComponentInstance::getCCAPort(const std::string& name)
{
  mutex->lock();
  sci::cca::Port::pointer svc = framework->getFrameworkService(name, instanceName);
  if(!svc.isNull()){
    return svc;
  }
  map<string, PortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end())
    return sci::cca::Port::pointer(0);
  CCAPortInstance* pr = dynamic_cast<CCAPortInstance*>(iter->second);
  if(pr == NULL)
    return sci::cca::Port::pointer(0);
  if(pr->porttype != CCAPortInstance::Uses)
    throw CCAException("Cannot call getPort on a Provides port");
  pr->incrementUseCount();
  if(pr->connections.size() != 1)
    return sci::cca::Port::pointer(0); 
  CCAPortInstance *pi=dynamic_cast<CCAPortInstance*> (pr->getPeer());
  mutex->unlock();
  return pi->port;
}

gov::cca::Port BridgeComponentInstance::getBabelPort(const std::string& name)
{
  mutex->lock();
  map<string, PortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end())
    return 0;
  BabelPortInstance* pr = dynamic_cast<BabelPortInstance*>(iter->second);
  if(pr == NULL)
    return 0;
  if(pr->porttype != BabelPortInstance::Uses) {
    throw InternalError("Cannot call getPort on a Provides port");
  }
  pr->incrementUseCount();
  if(pr->connections.size() != 1)
    return 0; 
  BabelPortInstance *pi=dynamic_cast<BabelPortInstance*> (pr->getPeer());
  mutex->unlock();
  return pi->port;
}

void BridgeComponentInstance::releasePort(const std::string& name, const modelT model)
{
  CCAPortInstance* cpr;
  BabelPortInstance* bpr;
  map<string, PortInstance*>::iterator iter;

  switch (model) {
  case CCA:
    if(framework->releaseFrameworkService(name, instanceName))
      return;
    
    iter = ports.find(name);
    if(iter == ports.end()){
      cerr << "Released an unknown port: " << name << '\n';
      throw CCAException("Released an unknown port: "+name);
    }
    cpr = dynamic_cast<CCAPortInstance*>(iter->second);
    if(cpr == NULL)
      throw CCAException("Trying to release a port of the wrong type");
    
    if(cpr->porttype != CCAPortInstance::Uses)
      throw CCAException("Cannot call releasePort on a Provides port");
    
    if(!cpr->decrementUseCount())
      throw CCAException("Port released without correspond get");
    break;

  case Babel:
    iter = ports.find(name);
    if(iter == ports.end()){
      cerr << "Released an unknown port: " << name << '\n';
      throw InternalError("Released an unknown port: "+name);
    }
    bpr = dynamic_cast<BabelPortInstance*>(iter->second);
    if(bpr == NULL)
      throw InternalError("Trying to release a port of the wrong type");
    
    if(bpr->porttype != BabelPortInstance::Uses)
      throw InternalError("Cannot call releasePort on a Provides port");
    
    if(!bpr->decrementUseCount())
      throw InternalError("Port released without correspond get");
    break;
  }
  return;
}

void BridgeComponentInstance::registerUsesPort(const std::string& portName,
					    const std::string& portType,
					    const modelT model)
{
  CCAPortInstance* cpr;
  BabelPortInstance* bpr;
  map<string, PortInstance*>::iterator iter;

  switch (model) {
  case CCA:	
    iter = ports.find(portName);
    if(iter != ports.end()){
      cpr = dynamic_cast<CCAPortInstance*>(iter->second);
      if(cpr == NULL)
	throw CCAException("Trying to register a port of the wrong type");
      if(cpr->porttype == CCAPortInstance::Provides)
	throw CCAException("name conflict between uses and provides ports");
      else {
	cerr << "registerUsesPort called twice, instance=" << instanceName << ", portName = " << portName << ", portType = " << portType << '\n';
	throw CCAException("registerUsesPort called twice");
      } 
    }
    ports.insert(make_pair(portName, new CCAPortInstance(portName, portType, sci::cca::TypeMap::pointer(0), CCAPortInstance::Uses)));
    break;

  case Babel:
    iter = ports.find(portName);
    if(iter != ports.end()){
      bpr = dynamic_cast<BabelPortInstance*>(iter->second);
      if(bpr == NULL)
	throw InternalError("Trying to register a port of the wrong type");
      if(bpr->porttype == BabelPortInstance::Provides)
	throw InternalError("name conflict between uses and provides ports");
      else {
	cerr << "registerUsesPort called twice, instance=" << instanceName << ", portName = " << portName << ", portType = " << portType << '\n';
	throw InternalError("registerUsesPort called twice");
      }     
    }
    ports.insert(make_pair(portName, new BabelPortInstance(portName, portType, 0, BabelPortInstance::Uses)));
    break;
  }
  return;
}

void BridgeComponentInstance::unregisterUsesPort(const std::string& portName, const modelT model)
{
  CCAPortInstance* cpr;
  BabelPortInstance* bpr;
  map<string, PortInstance*>::iterator iter;

  switch (model) {
  case CCA:  
    iter = ports.find(portName);
    if(iter != ports.end()){
      cpr = dynamic_cast<CCAPortInstance*>(iter->second);
      if(cpr == NULL)
	throw CCAException("Trying to unregister a port of the wrong type");
      if(cpr->porttype == CCAPortInstance::Provides)
	throw CCAException("name conflict between uses and provides ports");
      else {
	ports.erase(portName);
      }
    }
    else{
      cerr<<"port name not found, unregisterUsesPort not done\n";
      throw CCAException("port name not found");
    }
    break;

  case Babel:
    iter = ports.find(portName);
    if(iter != ports.end()){
      bpr = dynamic_cast<BabelPortInstance*>(iter->second);
      if(bpr == NULL)
	throw InternalError("Trying to unregister a port of the wrong type");
      if(bpr->porttype == BabelPortInstance::Provides)
	throw InternalError("name conflict between uses and provides ports");
      else {
	ports.erase(portName);
      }
    }
    else{
      cerr<<"port name not found, unregisterUsesPort not done\n";
      throw CCAException("port name not found");
    }
    break;
  }
  return;
}

void BridgeComponentInstance::addProvidesPort(void* port,
					   const std::string& portName,
					   const std::string& portType,
					   const modelT model)
{
  CCAPortInstance* cpr;
  BabelPortInstance* bpr;
  map<string, PortInstance*>::iterator iter;
  sci::cca::Port::pointer* ccaport;
  gov::cca::Port* babelport;

  switch (model) {
  case CCA:
    iter = ports.find(portName);
    if(iter != ports.end()){
      cpr = dynamic_cast<CCAPortInstance*>(iter->second);
      if(cpr == NULL)
	throw CCAException("port name conflicts with another one of a different type");
      if(cpr->porttype == CCAPortInstance::Uses)
	throw CCAException("name conflict between uses and provides ports");
      else
	throw CCAException("addProvidesPort called twice");
    }

    ccaport = reinterpret_cast<sci::cca::Port::pointer*>(port);
    if(!ccaport)
      throw CCAException("Wrong port model for addProvidesPort");
    ports.insert(make_pair(portName, new CCAPortInstance(portName, portType, sci::cca::TypeMap::pointer(0), *ccaport, CCAPortInstance::Provides)));
    break;

  case Babel:
    iter = ports.find(portName);
    if(iter != ports.end()){
      bpr = dynamic_cast<BabelPortInstance*>(iter->second);
      if(bpr == NULL)
	throw InternalError("port name conflicts with another one of a different type");
      if(bpr->porttype == BabelPortInstance::Uses)
	throw InternalError("name conflict between uses and provides ports");
      else
	throw InternalError("addProvidesPort called twice");
    }
    babelport = reinterpret_cast<gov::cca::Port*>(port);
    if(!babelport)
      throw InternalError("Wrong port model for addProvidesPort");
    ports.insert(make_pair(portName, new BabelPortInstance(portName, portType, 0, *babelport, BabelPortInstance::Provides)));
    break;
  }
  return;
}

void BridgeComponentInstance::removeProvidesPort(const std::string& name, const modelT model)
{
  switch (model) {
  case CCA:
    cerr << "removeProvidesPort not done, name=" << name << '\n';
    break;
  case Babel:
    cerr << "removeProvidesPort not done, name=" << name << '\n';
    break;
  }
  return;
}


sci::cca::ComponentID::pointer BridgeComponentInstance::getComponentID()
{
  return sci::cca::ComponentID::pointer(new ComponentID(framework, instanceName));
}

PortInstanceIterator* BridgeComponentInstance::getPorts()
{
  return new Iterator(this);
}

BridgeComponentInstance::Iterator::Iterator(BridgeComponentInstance* comp)
  :iter(comp->ports.begin()), comp(comp)
{
}

BridgeComponentInstance::Iterator::~Iterator()
{
}

PortInstance* BridgeComponentInstance::Iterator::get()
{
  return iter->second;
}

bool BridgeComponentInstance::Iterator::done()
{
  return iter == comp->ports.end();
}

void BridgeComponentInstance::Iterator::next()
{
  ++iter;
}
