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
 *  BabelComponentInstance.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */

#include <SCIRun/Babel/BabelPortInstance.h>
#include <SCIRun/CCA/CCAPortInstance.h>
#include <SCIRun/CCA/CCAException.h>
#include <iostream>
#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Babel/BabelComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>

using namespace std;
using namespace SCIRun;


BabelCCAGoPort::BabelCCAGoPort(const gov::cca::ports::GoPort& port) 
{
  this->port=port;
}

int BabelCCAGoPort::go() 
{
  return port.go();
}

BabelCCAUIPort::BabelCCAUIPort(const gov::cca::ports::UIPort& port) 
{
  this->port=port;
}

int BabelCCAUIPort::ui() 
{
  return port.ui();
}


BabelComponentInstance::BabelComponentInstance(SCIRunFramework* framework,
					       const std::string& instanceName,
					       const std::string& typeName,
					       const gov::cca::TypeMap& properties  ,
					       const gov::cca::Component& component,
					       const framework::Services& svc)
  : ComponentInstance(framework, instanceName, typeName)
{
  this->component=component;
  this->svc=svc;
  BabelPortInstance *go=dynamic_cast<BabelPortInstance*> (getPortInstance("go"));
  if(go!=0){
    
    std::map<std::string, PortInstance*> *pports=
      (std::map<std::string, PortInstance*>* ) (this->svc.getData());
     
    sci::cca::ports::GoPort::pointer goPort(new BabelCCAGoPort(go->port));
    CCAPortInstance *piGo=new CCAPortInstance("sci.go","sci.cca.ports.GoPort",
					      sci::cca::TypeMap::pointer(0),
					      goPort,
					      CCAPortInstance::Provides);
    
    pports->insert(make_pair("sci.go", piGo));
  }

  BabelPortInstance *ui=dynamic_cast<BabelPortInstance*> (getPortInstance("ui"));
  if(ui!=0){
    
    std::map<std::string, PortInstance*> *pports=
      (std::map<std::string, PortInstance*>* ) (this->svc.getData());
     
    sci::cca::ports::UIPort::pointer uiPort(new BabelCCAUIPort(ui->port));
    CCAPortInstance *piUI=new CCAPortInstance("sci.ui","sci.cca.ports.UIPort",
					      sci::cca::TypeMap::pointer(0),
					      uiPort,
					      CCAPortInstance::Provides);
    
    pports->insert(make_pair("sci.ui", piUI));
  }
}

BabelComponentInstance::~BabelComponentInstance()
{
  cerr << "BabelComponentInstance destroyed...\n";
}

PortInstance*
BabelComponentInstance::getPortInstance(const std::string& portname)
{
  std::map<std::string, PortInstance*> *pports=
    (std::map<std::string, PortInstance*>*)svc.getData();

  map<string, PortInstance*>::iterator iter = pports->find(portname);
  if(iter == pports->end())
    return 0;
  else
    return iter->second;
}

gov::cca::Port BabelComponentInstance::getPort(const std::string& name)
{
  return svc.getPort(name);
}

gov::cca::Port
BabelComponentInstance::getPortNonblocking(const std::string& name)
{
  return svc.getPortNonblocking(name);
}

void BabelComponentInstance::releasePort(const std::string& name)
{
  return svc.releasePort(name);
}

gov::cca::TypeMap BabelComponentInstance::createTypeMap()
{
  return svc.createTypeMap();
}

void BabelComponentInstance::registerUsesPort(const std::string& portName,
					    const std::string& portType,
					    const gov::cca::TypeMap& properties)
{
  return svc.registerUsesPort(portName, portType, properties);
}

void BabelComponentInstance::unregisterUsesPort(const std::string& name)
{
  return svc.unregisterUsesPort(name);
}

void BabelComponentInstance::addProvidesPort(const gov::cca::Port& port,
					   const std::string& portName,
					   const std::string& portType,
					   const gov::cca::TypeMap& properties)
{
  return svc.addProvidesPort(port, portName, portType, properties);
}

void BabelComponentInstance::removeProvidesPort(const std::string& name)
{
   svc.removeProvidesPort(name);
   return;
}

gov::cca::TypeMap BabelComponentInstance::getPortProperties(const std::string& portName)
{
  return svc.getPortProperties(portName);
}

gov::cca::ComponentID BabelComponentInstance::getComponentID()
{
  return svc.getComponentID();
}

PortInstanceIterator* BabelComponentInstance::getPorts()
{
  return new Iterator(this);
}

BabelComponentInstance::Iterator::Iterator(BabelComponentInstance* comp)
{
  ports= (std::map<std::string, PortInstance*>*) (comp->svc.getData());
  iter=ports->begin();
}

BabelComponentInstance::Iterator::~Iterator()
{
}

PortInstance* BabelComponentInstance::Iterator::get()
{
  return iter->second;
}

bool BabelComponentInstance::Iterator::done()
{
  return iter == ports->end();
}

void BabelComponentInstance::Iterator::next()
{
  ++iter;
}
