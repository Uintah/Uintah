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
 *  InternalServices.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalComponentDescription.h>
#include <SCIRun/Internal/BuilderService.h>
#include <SCIRun/Internal/ComponentEventService.h>
#include <SCIRun/Internal/ComponentRegistry.h>
#include <SCIRun/SCIRunFramework.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

InternalComponentModel::InternalComponentModel(SCIRunFramework* framework)
  : ComponentModel("internal"), framework(framework)
{
  addService(new InternalComponentDescription(this, "cca.builderService",
					      &BuilderService::create, true));
  addService(new InternalComponentDescription(this, "cca.componentRegistry",
					      &ComponentRegistry::create, true));
  addService(new InternalComponentDescription(this, "cca.componentEventService",
					      &ComponentEventService::create, true));
}

InternalComponentModel::~InternalComponentModel()
{
  for(map<string, InternalComponentDescription*>::iterator iter=services.begin();
      iter != services.end(); iter++)
    delete iter->second;
}

void InternalComponentModel::addService(InternalComponentDescription* svc)
{
  if(services.find(svc->serviceType) != services.end())
    cerr << "WARNING: duplicate service: " << svc->serviceType << '\n';
  services[svc->serviceType]=svc;
}

gov::cca::Port InternalComponentModel::getFrameworkService(const std::string& type)
{
  map<string, InternalComponentDescription*>::iterator iter=services.find(type);
  if(iter == services.end())
    return 0;
  InternalComponentDescription* cd = iter->second;
  InternalComponentInstance* ci;
  string cname = "internal: "+type;
  if(cd->isSingleton){
    if(!cd->singleton_instance){
      cd->singleton_instance = (*cd->create)(framework, cname);
      framework->registerComponent(cd->singleton_instance, cname);
    }
    ci = cd->singleton_instance;
  } else {
    ci = (*cd->create)(framework, cname);
    framework->registerComponent(ci, cname);
  }
  return ci->getService(type);
}

bool InternalComponentModel::haveComponent(const std::string& name)
{
  return false;
}

ComponentInstance* InternalComponentModel::createInstance(const std::string&,
							  const std::string&)
{
  return 0;
}

string InternalComponentModel::getName() const
{
  return "Internal";
}

void
InternalComponentModel::listAllComponentTypes(vector<ComponentDescription*>& list,
					      bool listInternal)
{
  if(listInternal){
    for(map<string, InternalComponentDescription*>::iterator iter = services.begin();
	iter != services.end(); iter++)
      list.push_back(iter->second);
  }
}
