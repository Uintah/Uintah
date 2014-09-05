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
#include <SCIRun/CCA/CCAPortInstance.h>
#include <SCIRun/CCA/CCAException.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

CCAComponentInstance::CCAComponentInstance(SCIRunFramework* framework,
					   const std::string& instanceName,
					   const std::string& typeName,
					   const gov::cca::TypeMap::pointer& properties,
					   const gov::cca::Component::pointer& component
)
  : ComponentInstance(framework, instanceName, typeName), component(component)
{
}

CCAComponentInstance::~CCAComponentInstance()
{
  cerr << "CCAComponentInstance destroyed...\n";
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

gov::cca::Port::pointer CCAComponentInstance::getPort(const std::string& name)
{
  gov::cca::Port::pointer svc = framework->getFrameworkService(name);
  if(!svc.isNull())
    return svc;

  map<string, CCAPortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end())
    return gov::cca::Port::pointer(0);

  CCAPortInstance* pr = iter->second;
  if(pr->porttype != CCAPortInstance::Uses)
    throw CCAException("Cannot call getPort on a Provides port");

  if(pr->connections.size() != 1)
    throw CCAException("More than 1 port connected, but getPort called");
  return pr->connections[0]->port;
}

gov::cca::Port::pointer CCAComponentInstance::getPortNonblocking(const std::string& name)
{
  cerr << "getPortNonblocking not done, name=" << name << '\n';
  return gov::cca::Port::pointer(0);
}

void CCAComponentInstance::releasePort(const std::string& name)
{
  cerr << "releasePort not done, name=" << name << '\n';
}

gov::cca::TypeMap::pointer CCAComponentInstance::createTypeMap()
{
  cerr << "createTypeMap not done\n";
  return gov::cca::TypeMap::pointer(0);
}

void CCAComponentInstance::registerUsesPort(const std::string& portName,
					    const std::string& portType,
					    const gov::cca::TypeMap::pointer& properties)
{
  map<string, CCAPortInstance*>::iterator iter = ports.find(portName);
  if(iter != ports.end()){
    if(iter->second->porttype == CCAPortInstance::Provides)
      throw CCAException("name conflict between uses and provides ports");
    else
      throw CCAException("registerUsesPort called twice");
  }
  ports.insert(make_pair(portName, new CCAPortInstance(portName, portType, properties, CCAPortInstance::Uses)));
}

void CCAComponentInstance::unregisterUsesPort(const std::string& name)
{
  cerr << "unregisterUsesPort not done, name=" << name << '\n';
}
void CCAComponentInstance::addProvidesPort(const gov::cca::Port::pointer& port,
					   const std::string& portName,
					   const std::string& portType,
					   const gov::cca::TypeMap::pointer& properties)
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

gov::cca::TypeMap::pointer CCAComponentInstance::getPortProperties(const std::string& portName)
{
  cerr << "getPortProperties not done, name=" << portName << '\n';
  return gov::cca::TypeMap::pointer(0);
}

gov::cca::ComponentID::pointer CCAComponentInstance::getComponentID()
{
  return gov::cca::ComponentID::pointer(new ComponentID(framework, instanceName));
}

