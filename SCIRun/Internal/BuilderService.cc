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
 *  BuilderService.cc: Implementation of CCA BuilderService for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Internal/BuilderService.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <SCIRun/PortInstanceIterator.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAException.h>
#include <SCIRun/PortInstance.h>
#include <SCIRun/CCA/ComponentID.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/CCA/ConnectionID.h>
#include <iostream>
#include <string>
using namespace std;
using namespace SCIRun;

BuilderService::BuilderService(SCIRunFramework* framework,
			       const std::string& name)
  : InternalComponentInstance(framework, name, "internal:BuilderService")
{
    this->framework=framework;
}

BuilderService::~BuilderService()
{
}

gov::cca::ComponentID::pointer
BuilderService::createInstance(const std::string& instanceName,
			       const std::string& className,
			       const gov::cca::TypeMap::pointer& /*properties*/,
			       const std::string &url)
{
  cerr << "Need to do something with properties...\n";
  return framework->createComponentInstance(instanceName, className,url);
}

gov::cca::ConnectionID::pointer BuilderService::connect(const gov::cca::ComponentID::pointer& c1,
					       const string& port1,
					       const gov::cca::ComponentID::pointer& c2,
					       const string& port2)
{
 ComponentID* cid1 = dynamic_cast<ComponentID*>(c1.getPointer());
  ComponentID* cid2 = dynamic_cast<ComponentID*>(c2.getPointer());
  if(!cid1 || !cid2){
    throw CCAException("Cannot understand this ComponentID");
  }
  if(cid1->framework != framework || cid2->framework != framework){
    throw CCAException("Cannot connect components from different frameworks");
  }
  ComponentInstance* comp1=framework->lookupComponent(cid1->name);
  ComponentInstance* comp2=framework->lookupComponent(cid2->name);
  PortInstance* pr1=comp1->getPortInstance(port1);
  if(!pr1){
    throw CCAException("Unknown port");
  }
  PortInstance* pr2=comp2->getPortInstance(port2);
  if(!pr2){
    throw CCAException("Unknown port");
  }
  if(!pr1->connect(pr2)){
    throw CCAException("Cannot connect");
  }
  cerr << "connect should return a connection ID\n";
  return gov::cca::ConnectionID::pointer(new ConnectionID(c1, port1, c2, port2));
}

InternalComponentInstance* BuilderService::create(SCIRunFramework* framework,
						  const std::string& name)
{
  BuilderService* n = new BuilderService(framework, name);
  n->addReference();
  return n;
}

gov::cca::Port::pointer BuilderService::getService(const std::string&)
{
  return gov::cca::Port::pointer(this);
}

CIA::array1<gov::cca::ComponentID::pointer> BuilderService::getComponentIDs()
{
  cerr << "BuilderService::getComponentIDs not finished\n";
  CIA::array1<gov::cca::ComponentID::pointer> junk(0);
  return junk;
}

gov::cca::TypeMap::pointer BuilderService::getComponentProperties(const gov::cca::ComponentID::pointer& /*cid*/)
{
  cerr << "BuilderService::getComponentProperties not finished\n";
  return gov::cca::TypeMap::pointer(0);
}

void BuilderService::setComponentProperties(const gov::cca::ComponentID::pointer& /*cid*/,
					    const gov::cca::TypeMap::pointer& /*map*/)
{
  cerr << "BuilderService::setComponentProperties not finished\n";
}

gov::cca::ComponentID::pointer BuilderService::getDeserialization(const std::string& /*s*/)
{
  cerr << "BuilderService::getDeserialization not finished\n";
  return gov::cca::ComponentID::pointer(0);
}

gov::cca::ComponentID::pointer BuilderService::getComponentID(const std::string& /*componentInstanceName*/)
{
  cerr << "BuilderService::getComponentID not finished\n";
  return gov::cca::ComponentID::pointer(0);
}

void BuilderService::destroyInstance(const gov::cca::ComponentID::pointer& /*toDie*/,
				     float /*timeout*/)
{
  cerr << "BuilderService::destroyInstance not finished\n";
}

CIA::array1<std::string> BuilderService::getProvidedPortNames(const gov::cca::ComponentID::pointer& cid)
{
  CIA::array1<std::string> result;
  ComponentInstance *ci=framework->lookupComponent(cid->getInstanceName());
  for(PortInstanceIterator* iter = ci->getPorts(); !iter->done(); iter->next()){
    PortInstance* port = iter->get();
    if(port->portType() == PortInstance::To)
      result.push_back(port->getUniqueName());
  }
  return result;
}

CIA::array1<std::string> BuilderService::getUsedPortNames(const gov::cca::ComponentID::pointer& cid)
{
  CIA::array1<std::string> result;
  ComponentInstance *ci=framework->lookupComponent(cid->getInstanceName());
  for(PortInstanceIterator* iter = ci->getPorts(); !iter->done(); iter->next()){
    PortInstance* port = iter->get();
    if(port->portType() == PortInstance::From)
      result.push_back(port->getUniqueName());
  }
  return result;
}

gov::cca::TypeMap::pointer BuilderService::getPortProperties(const gov::cca::ComponentID::pointer& /*cid*/,
							     const std::string& /*portname*/)
{
  cerr << "BuilderService::getPortProperties not finished\n";
  return gov::cca::TypeMap::pointer(0);
}

void BuilderService::setPortProperties(const gov::cca::ComponentID::pointer& /*cid*/,
				       const std::string& /*portname*/,
				       const gov::cca::TypeMap::pointer& /*map*/)
{
  cerr << "BuilderService::setPortProperties not finished\n";
}

CIA::array1<gov::cca::ConnectionID::pointer> BuilderService::getConnectionIDs(const CIA::array1<gov::cca::ComponentID::pointer>& /*componentList*/)
{
  cerr << "BuilderService::getConnectionIDs not finished\n";
  CIA::array1<gov::cca::ConnectionID::pointer> junk(0);
  return junk;
}

gov::cca::TypeMap::pointer
BuilderService::getConnectionProperties(const gov::cca::ConnectionID::pointer& /*connID*/)
{
  cerr << "BuilderService::getConnectionProperties not finished\n";
  return gov::cca::TypeMap::pointer(0);
}

void BuilderService::setConnectionProperties(const gov::cca::ConnectionID::pointer& /*connID*/,
					     const gov::cca::TypeMap::pointer& /*map*/)
{
  cerr << "BuilderService::setConnectionProperties not finished\n";
}

void BuilderService::disconnect(const gov::cca::ConnectionID::pointer& connID,
				float /*timeout*/)
{
  ComponentID* userID=dynamic_cast<ComponentID*>(connID->getUser().getPointer());
  ComponentID* providerID=dynamic_cast<ComponentID*>(connID->getProvider().getPointer());

  ComponentInstance* user=framework->lookupComponent(userID->name);
  ComponentInstance* provider=framework->lookupComponent(providerID->name);

  PortInstance* userPort=user->getPortInstance(connID->getUserPortName());
  PortInstance* providerPort=provider->getPortInstance(connID->getProviderPortName());
  userPort->disconnect(providerPort);
  cerr << "BuilderService::disconnect: timeout or safty check needed "<<endl;
}

void BuilderService::disconnectAll(const gov::cca::ComponentID::pointer& /*id1*/,
				   const gov::cca::ComponentID::pointer& /*id2*/,
				   float /*timeout*/)
{
  cerr << "BuilderService::disconnectAll not finished\n";
}


CIA::array1<std::string>  BuilderService::getCompatiblePortList(
     const gov::cca::ComponentID::pointer& c1,
     const std::string& port1,
     const gov::cca::ComponentID::pointer& c2)
{
  ComponentID* cid1 = dynamic_cast<ComponentID*>(c1.getPointer());
  ComponentID* cid2 = dynamic_cast<ComponentID*>(c2.getPointer());
  if(!cid1 || !cid2)
    throw CCAException("Cannot understand this ComponentID");
  if(cid1->framework != framework || cid2->framework != framework){
    throw CCAException("Cannot connect components from different frameworks");
  }
  ComponentInstance* comp1=framework->lookupComponent(cid1->name);
  ComponentInstance* comp2=framework->lookupComponent(cid2->name);
  PortInstance* pr1=comp1->getPortInstance(port1);
  if(!pr1)
    throw CCAException("Unknown port");


  CIA::array1<std::string> availablePorts;
  for(PortInstanceIterator* iter = comp2->getPorts(); !iter->done();
      iter->next()){
    PortInstance* pr2 = iter->get();
    if(pr1->canConnectTo(pr2))
      availablePorts.push_back(pr2->getUniqueName());
  }  

  return availablePorts;
}

void BuilderService::registerFramework(const string &frameworkURL)
{
  Object::pointer obj=PIDL::objectFrom(frameworkURL);
  gov::cca::AbstractFramework::pointer remoteFramework=
    pidl_cast<gov::cca::AbstractFramework::pointer>(obj);
  gov::cca::Services::pointer bs = remoteFramework->getServices("external builder", 
								"builder main", 
								gov::cca::TypeMap::pointer(0));
  cerr << "got bs\n";
  gov::cca::ports::ComponentRepository::pointer reg =
    pidl_cast<gov::cca::ports::ComponentRepository::pointer>
    (bs->getPort("cca.ComponentRepository"));
  if(reg.isNull()){
    cerr << "Cannot get component registry, not building component menus\n";
    return;
  }
  
  //traverse Builder Components here...

  for(unsigned int i=0; i<servicesList.size();i++){
    gov::cca::ports::BuilderService::pointer builder 
      = pidl_cast<gov::cca::ports::BuilderService::pointer>
      (servicesList[i]->getPort("cca.BuilderService"));

    if(builder.isNull()){
      cerr << "Fatal Error: Cannot find builder service\n";
      return;
    } 
    
    gov::cca::ComponentID::pointer cid=servicesList[i]->getComponentID();
    cerr<<"try to connect..."<<endl;
    gov::cca::ConnectionID::pointer connID=builder->connect(cid, "builderPort",
							    cid, "builder");
    cerr<<"connection done"<<endl;
  

    gov::cca::Port::pointer p = servicesList[i]->getPort("builder");
    gov::cca::ports::BuilderPort::pointer bp = 
      pidl_cast<gov::cca::ports::BuilderPort::pointer>(p);
    if(bp.isNull()){
      cerr << "BuilderPort is not connected!\n";
    } 
    else{
      bp->buildRemotePackageMenus(reg, frameworkURL);
    }
    builder->disconnect(connID,0);
    servicesList[i]->releasePort("cca.BuilderService"); 
    servicesList[i]->releasePort("builder");
  }
  
}

void BuilderService::registerServices(const gov::cca::Services::pointer &svc)
{
  servicesList.push_back(svc);
}

gov::cca::AbstractFramework::pointer BuilderService::getFramework()
{
  return gov::cca::AbstractFramework::pointer(framework);
}



