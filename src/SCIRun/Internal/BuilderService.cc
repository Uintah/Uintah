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
  : InternalComponentInstance(framework, name)
{
}

BuilderService::~BuilderService()
{
}

gov::cca::ComponentID BuilderService::createComponentInstance(const std::string& name,
							      const std::string& type)
{
  return framework->createComponentInstance(name, type);
}

void BuilderService::connect(const gov::cca::ComponentID& c1,
			     const string& port1,
			     const gov::cca::ComponentID& c2,
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
}

InternalComponentInstance* BuilderService::create(SCIRunFramework* framework,
						  const std::string& name)
{
  BuilderService* n = new BuilderService(framework, name);
  n->_addReference();
  return n;
}

gov::cca::Port BuilderService::getService(const std::string&)
{
  return this;
}
