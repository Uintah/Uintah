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
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAException.h>
#include <SCIRun/PortInstance.h>
#include <SCIRun/CCA/ComponentID.h>
#include <SCIRun/ComponentInstance.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

BuilderService::BuilderService(SCIRunFramework* framework,
			       const std::string& name)
  : InternalComponentInstance(framework, name, "internal:BuilderService")
{
}

BuilderService::~BuilderService()
{
}

gov::cca::ComponentID::pointer
BuilderService::createInstance(const std::string& instanceName,
			       const std::string& className,
			       const gov::cca::TypeMap::pointer& properties)
{
  cerr << "Need to do something with properties...\n";
  return framework->createComponentInstance(instanceName, className);
}

gov::cca::ConnectionID::pointer BuilderService::connect(const gov::cca::ComponentID::pointer& c1,
					       const string& port1,
					       const gov::cca::ComponentID::pointer& c2,
					       const string& port2)
{
  ComponentID* cid1 = dynamic_cast<ComponentID*>(c1.getPointer());
  ComponentID* cid2 = dynamic_cast<ComponentID*>(c2.getPointer());
  if(!cid1 || !cid2)
    throw CCAException("Cannot understand this ComponentID");
  if(cid1->framework != framework || cid2->framework != framework){
    throw CCAException("Cannot connect components from different frameworks");
  }
  ComponentInstance* comp1=framework->getComponent(cid1->name);
  ComponentInstance* comp2=framework->getComponent(cid2->name);
  PortInstance* pr1=comp1->getPortInstance(port1);
  if(!pr1)
    throw CCAException("Unknown port");
  PortInstance* pr2=comp2->getPortInstance(port2);
  if(!pr2)
    throw CCAException("Unknown port");
  if(!pr1->connect(pr2))
    throw CCAException("Cannot connect");
  cerr << "connect should return a connection ID\n";
  return gov::cca::ConnectionID::pointer(0);
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

gov::cca::TypeMap::pointer BuilderService::getComponentProperties(const gov::cca::ComponentID::pointer& cid)
{
  cerr << "BuilderService::getComponentProperties not finished\n";
  return gov::cca::TypeMap::pointer(0);
}

void BuilderService::setComponentProperties(const gov::cca::ComponentID::pointer& cid,
					const gov::cca::TypeMap::pointer& map)
{
  cerr << "BuilderService::setComponentProperties not finished\n";
}

gov::cca::ComponentID::pointer BuilderService::getDeserialization(const std::string& s)
{
  cerr << "BuilderService::getDeserialization not finished\n";
  return gov::cca::ComponentID::pointer(0);
}

gov::cca::ComponentID::pointer BuilderService::getComponentID(const std::string& componentInstanceName)
{
  cerr << "BuilderService::getComponentID not finished\n";
  return gov::cca::ComponentID::pointer(0);
}

void BuilderService::destroyInstance(const gov::cca::ComponentID::pointer& toDie,
				 float timeout)
{
  cerr << "BuilderService::destroyInstance not finished\n";
}

CIA::array1<std::string> BuilderService::getProvidedPortNames(const gov::cca::ComponentID::pointer& cid)
{
  cerr << "BuilderService::getProvidedPortNames not finished\n";
  CIA::array1<std::string> junk(0);
  return junk;
}

CIA::array1<std::string> BuilderService::getUsedPortNames(const gov::cca::ComponentID::pointer& cid)
{
  cerr << "BuilderService::getUsedPortNames not finished\n";
  CIA::array1<std::string> junk(0);
  return junk;
}

gov::cca::TypeMap::pointer BuilderService::getPortProperties(const gov::cca::ComponentID::pointer& cid,
							     const std::string& portname)
{
  cerr << "BuilderService::getPortProperties not finished\n";
  return gov::cca::TypeMap::pointer(0);
}

void BuilderService::setPortProperties(const gov::cca::ComponentID::pointer& cid,
				   const std::string& portname,
				   const gov::cca::TypeMap::pointer& map)
{
  cerr << "BuilderService::setPortProperties not finished\n";
}

CIA::array1<gov::cca::ConnectionID::pointer> BuilderService::getConnectionIDs(const CIA::array1<gov::cca::ComponentID::pointer>& componentList)
{
  cerr << "BuilderService::getConnectionIDs not finished\n";
  CIA::array1<gov::cca::ConnectionID::pointer> junk(0);
  return junk;
}

gov::cca::TypeMap::pointer
BuilderService::getConnectionProperties(const gov::cca::ConnectionID::pointer& connID)
{
  cerr << "BuilderService::getConnectionProperties not finished\n";
  return gov::cca::TypeMap::pointer(0);
}

void BuilderService::setConnectionProperties(const gov::cca::ConnectionID::pointer& connID,
					 const gov::cca::TypeMap::pointer& map)
{
  cerr << "BuilderService::setConnectionProperties not finished\n";
}

void BuilderService::disconnect(const gov::cca::ConnectionID::pointer& connID,
			    float timeout)
{
  cerr << "BuilderService::disconnect not finished\n";
}

void BuilderService::disconnectAll(const gov::cca::ComponentID::pointer& id1,
				   const gov::cca::ComponentID::pointer& id2,
				   float timeout)
{
  cerr << "BuilderService::disconnectAll not finished\n";
}

