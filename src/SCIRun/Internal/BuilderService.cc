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
#include <Core/CCA/PIDL/PIDL.h>
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

sci::cca::ComponentID::pointer
BuilderService::createInstance(const std::string& instanceName,
			       const std::string& className,
			       const sci::cca::TypeMap::pointer& properties)
{
  return framework->createComponentInstance(instanceName, className, properties);
}

sci::cca::ConnectionID::pointer BuilderService::connect(const sci::cca::ComponentID::pointer& c1,
					       const string& port1,
					       const sci::cca::ComponentID::pointer& c2,
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
  sci::cca::ConnectionID::pointer conn(new ConnectionID(c1, port1, c2, port2));
  framework->connIDs.push_back(conn);
  return conn;
}

InternalComponentInstance* BuilderService::create(SCIRunFramework* framework,
						  const std::string& name)
{
  BuilderService* n = new BuilderService(framework, name);
  n->addReference();
  return n;
}

sci::cca::Port::pointer BuilderService::getService(const std::string&)
{
  return sci::cca::Port::pointer(this);
}

SSIDL::array1<sci::cca::ComponentID::pointer> BuilderService::getComponentIDs()
{
  return framework->compIDs;
}

sci::cca::TypeMap::pointer BuilderService::getComponentProperties(const sci::cca::ComponentID::pointer& /*cid*/)
{
  cerr << "BuilderService::getComponentProperties not finished\n";
  return sci::cca::TypeMap::pointer(0);
}

void BuilderService::setComponentProperties(const sci::cca::ComponentID::pointer& /*cid*/,
					    const sci::cca::TypeMap::pointer& /*map*/)
{
  cerr << "BuilderService::setComponentProperties not finished\n";
}

sci::cca::ComponentID::pointer BuilderService::getDeserialization(const std::string& /*s*/)
{
  cerr << "BuilderService::getDeserialization not finished\n";
  return sci::cca::ComponentID::pointer(0);
}

sci::cca::ComponentID::pointer BuilderService::getComponentID(const std::string& componentInstanceName)
{
  sci::cca::ComponentID::pointer cid=framework->lookupComponentID(componentInstanceName);
  if(cid.isNull()) throw CCAException("ComponentID not found");
  return cid;
}

void BuilderService::destroyInstance(const sci::cca::ComponentID::pointer& toDie,
				     float timeout)
{
  framework->destroyComponentInstance(toDie, timeout);
  return ;
}

SSIDL::array1<std::string> BuilderService::getProvidedPortNames(const sci::cca::ComponentID::pointer& cid)
{
  SSIDL::array1<std::string> result;
  ComponentInstance *ci=framework->lookupComponent(cid->getInstanceName());
  cerr<<"Component: "<<cid->getInstanceName()<<endl;
  for(PortInstanceIterator* iter = ci->getPorts(); !iter->done(); iter->next()){
    PortInstance* port = iter->get();
    if(port->portType() == PortInstance::To)
      result.push_back(port->getUniqueName());
  }
  return result;
}

SSIDL::array1<std::string> BuilderService::getUsedPortNames(const sci::cca::ComponentID::pointer& cid)
{
  SSIDL::array1<std::string> result;
  ComponentInstance *ci=framework->lookupComponent(cid->getInstanceName());
  for(PortInstanceIterator* iter = ci->getPorts(); !iter->done(); iter->next()){
    PortInstance* port = iter->get();
    if(port->portType() == PortInstance::From)
      result.push_back(port->getUniqueName());
  }
  return result;
}

sci::cca::TypeMap::pointer BuilderService::getPortProperties(const sci::cca::ComponentID::pointer& /*cid*/,
							     const std::string& /*portname*/)
{
  cerr << "BuilderService::getPortProperties not finished\n";
  return sci::cca::TypeMap::pointer(0);
}

void BuilderService::setPortProperties(const sci::cca::ComponentID::pointer& /*cid*/,
				       const std::string& /*portname*/,
				       const sci::cca::TypeMap::pointer& /*map*/)
{
  cerr << "BuilderService::setPortProperties not finished\n";
}

SSIDL::array1<sci::cca::ConnectionID::pointer> BuilderService::getConnectionIDs(const SSIDL::array1<sci::cca::ComponentID::pointer>& componentList)
{
  SSIDL::array1<sci::cca::ConnectionID::pointer> conns;
  for(unsigned i=0; i<framework->connIDs.size(); i++){
    for(unsigned j=0; j<componentList.size(); j++){
      sci::cca::ComponentID::pointer cid1=framework->connIDs[i]->getUser();
      sci::cca::ComponentID::pointer cid2=framework->connIDs[i]->getProvider();
      if(cid1==componentList[j]||cid2==componentList[j]){
	conns.push_back(framework->connIDs[i]);
	break;
      }
    }
  }
  return conns;
}

sci::cca::TypeMap::pointer
BuilderService::getConnectionProperties(const sci::cca::ConnectionID::pointer& /*connID*/)
{
  cerr << "BuilderService::getConnectionProperties not finished\n";
  return sci::cca::TypeMap::pointer(0);
}

void BuilderService::setConnectionProperties(const sci::cca::ConnectionID::pointer& /*connID*/,
					     const sci::cca::TypeMap::pointer& /*map*/)
{
  cerr << "BuilderService::setConnectionProperties not finished\n";
}

void BuilderService::disconnect(const sci::cca::ConnectionID::pointer& connID,
				float /*timeout*/)
{
  ComponentID* userID=dynamic_cast<ComponentID*>(connID->getUser().getPointer());
  ComponentID* providerID=dynamic_cast<ComponentID*>(connID->getProvider().getPointer());

  ComponentInstance* user=framework->lookupComponent(userID->name);
  ComponentInstance* provider=framework->lookupComponent(providerID->name);

  PortInstance* userPort=user->getPortInstance(connID->getUserPortName());
  PortInstance* providerPort=provider->getPortInstance(connID->getProviderPortName());
  userPort->disconnect(providerPort);
  for(unsigned i=0; i<framework->connIDs.size();i++){
    if(framework->connIDs[i]==connID){
      framework->connIDs.erase(framework->connIDs.begin()+i);
      break;
    }
  }
  //cerr << "BuilderService::disconnect: timeout or safty check needed "<<endl;
}

void BuilderService::disconnectAll(const sci::cca::ComponentID::pointer& /*id1*/,
				   const sci::cca::ComponentID::pointer& /*id2*/,
				   float /*timeout*/)
{
  cerr << "BuilderService::disconnectAll not finished\n";
}


SSIDL::array1<std::string>  BuilderService::getCompatiblePortList(
     const sci::cca::ComponentID::pointer& c1,
     const std::string& port1,
     const sci::cca::ComponentID::pointer& c2)
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

  cerr<<"Component: "<<cid2->getInstanceName()<<endl;
  PortInstance* pr1=comp1->getPortInstance(port1);
  if(!pr1)
    throw CCAException("Unknown port");


  SSIDL::array1<std::string> availablePorts;
  for(PortInstanceIterator* iter = comp2->getPorts(); !iter->done();
      iter->next()){
    PortInstance* pr2 = iter->get();
    if(pr1->canConnectTo(pr2))
      availablePorts.push_back(pr2->getUniqueName());
  }  

  return availablePorts;
}

std::string 
BuilderService::getFrameworkURL(){
  return framework->getURL().getString();
}


int BuilderService::addLoader(const std::string &loaderName, const std::string &user, const std::string &domain, const std::string &loaderPath )
{
  cerr<<"BuiderService::addLoader() not implemented\n";
  string cmd="xterm -e ssh ";
  //  cmd+=user+"@"+domain+" "+loaderPath+" "+loaderName+" "+getFrameworkURL() +"&";

  cmd="xterm -e "+loaderPath+" "+loaderName+" "+getFrameworkURL() +"&";
  cout<<cmd<<endl;
  system(cmd.c_str());
  return 0;
}

int BuilderService::removeLoader(const std::string &loaderName)
{
  cerr<<"BuiderService::removeLoader() not implemented\n";
  return 0;
}


int addComponentClasses(const std::string &loaderName)
{
  cerr<<"BuiderService::addComponentClasses not implemented\n";
  return 0;
}

int removeComponentClasses(const std::string &loaderName)
{
  cerr<<"BuiderService::removeComponentClasses not implemented\n";
  return 0;
}



/*
void BuilderService::registerFramework(const string &frameworkURL)
{
  Object::pointer obj=PIDL::objectFrom(frameworkURL);
  sci::cca::AbstractFramework::pointer remoteFramework=
    pidl_cast<sci::cca::AbstractFramework::pointer>(obj);
  sci::cca::Services::pointer bs = remoteFramework->getServices("external builder", 
								"builder main", 
								sci::cca::TypeMap::pointer(0));
  cerr << "got bs\n";
  sci::cca::ports::ComponentRepository::pointer reg =
    pidl_cast<sci::cca::ports::ComponentRepository::pointer>
    (bs->getPort("cca.ComponentRepository"));
  if(reg.isNull()){
    cerr << "Cannot get component registry, not building component menus\n";
    return;
  }
  
  //traverse Builder Components here...

  for(unsigned int i=0; i<servicesList.size();i++){
    sci::cca::ports::BuilderService::pointer builder 
      = pidl_cast<sci::cca::ports::BuilderService::pointer>
      (servicesList[i]->getPort("cca.BuilderService"));

    if(builder.isNull()){
      cerr << "Fatal Error: Cannot find builder service\n";
      return;
    } 
    
    sci::cca::ComponentID::pointer cid=servicesList[i]->getComponentID();
    cerr<<"try to connect..."<<endl;
    sci::cca::ConnectionID::pointer connID=builder->connect(cid, "builderPort",
							    cid, "builder");
    cerr<<"connection done"<<endl;
  

    sci::cca::Port::pointer p = servicesList[i]->getPort("builder");
    sci::cca::ports::BuilderPort::pointer bp = 
      pidl_cast<sci::cca::ports::BuilderPort::pointer>(p);
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

void BuilderService::registerServices(const sci::cca::Services::pointer &svc)
{
  servicesList.push_back(svc);
}


sci::cca::AbstractFramework::pointer BuilderService::getFramework()
{
  return sci::cca::AbstractFramework::pointer(framework);
}*/



