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


BabelCCAGoPort::BabelCCAGoPort(const govcca::GoPort& port) 
{
  this->port=port;
}

int BabelCCAGoPort::go() 
{
  return port.go();
}

BabelCCAUIPort::BabelCCAUIPort(const govcca::UIPort& port) 
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
					       const govcca::TypeMap& properties  ,
					       const govcca::Component& component,
					       const framework::Services& svc)
  : ComponentInstance(framework, instanceName, typeName)
{
  this->component=component;
  this->svc=svc;
  BabelPortInstance *go=dynamic_cast<BabelPortInstance*> (getPortInstance("babel.go"));
  if(go!=0){
    
    std::map<std::string, PortInstance*> *pports=
      (std::map<std::string, PortInstance*>* ) (this->svc.getData());
     
    gov::cca::ports::GoPort::pointer goPort(new BabelCCAGoPort(go->port));
    CCAPortInstance *piGo=new CCAPortInstance("go","gov.cca.GoPort",
					      gov::cca::TypeMap::pointer(0),
					      goPort,
					      CCAPortInstance::Provides);
    
    pports->insert(make_pair("go", piGo));
  }

  BabelPortInstance *ui=dynamic_cast<BabelPortInstance*> (getPortInstance("babel.ui"));
  if(ui!=0){
    
    std::map<std::string, PortInstance*> *pports=
      (std::map<std::string, PortInstance*>* ) (this->svc.getData());
     
    gov::cca::ports::UIPort::pointer uiPort(new BabelCCAUIPort(ui->port));
    CCAPortInstance *piUI=new CCAPortInstance("ui","gov.cca.UIPort",
					      gov::cca::TypeMap::pointer(0),
					      uiPort,
					      CCAPortInstance::Provides);
    
    pports->insert(make_pair("ui", piUI));
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

govcca::Port BabelComponentInstance::getPort(const std::string& name)
{
  return svc.getPort(name);
}

govcca::Port
BabelComponentInstance::getPortNonblocking(const std::string& name)
{
  return svc.getPortNonblocking(name);
}

void BabelComponentInstance::releasePort(const std::string& name)
{
  return svc.releasePort(name);
}

govcca::TypeMap BabelComponentInstance::createTypeMap()
{
  return svc.createTypeMap();
}

void BabelComponentInstance::registerUsesPort(const std::string& portName,
					    const std::string& portType,
					    const govcca::TypeMap& properties)
{
  return svc.registerUsesPort(portName, portType, properties);
}

void BabelComponentInstance::unregisterUsesPort(const std::string& name)
{
  return svc.unregisterUsesPort(name);
}

void BabelComponentInstance::addProvidesPort(const govcca::Port& port,
					   const std::string& portName,
					   const std::string& portType,
					   const govcca::TypeMap& properties)
{
  return svc.addProvidesPort(port, portName, portType, properties);
}

void BabelComponentInstance::removeProvidesPort(const std::string& name)
{
   svc.removeProvidesPort(name);
   return;
}

govcca::TypeMap BabelComponentInstance::getPortProperties(const std::string& portName)
{
  return svc.getPortProperties(portName);
}

govcca::ComponentID BabelComponentInstance::getComponentID()
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
