/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <SCIRun/Dataflow/DataflowScheduler.h>
#include <SCIRun/SCIRunFramework.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

InternalComponentModel::InternalComponentModel(SCIRunFramework* framework)
  : ComponentModel("internal"), framework(framework)
{
  addService(new InternalComponentDescription(this, "cca.BuilderService",
					      &BuilderService::create, true));
  addService(new InternalComponentDescription(this, "cca.ComponentRepository",
					      &ComponentRegistry::create, true));
  addService(new InternalComponentDescription(this, "cca.ComponentEventService",
					      &ComponentEventService::create, true));
  addService(new InternalComponentDescription(this, "cca.DataflowScheduler",
                                              &DataflowScheduler::create, true));
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
    cerr << "WARNING: duplicate internal service: " << svc->serviceType << '\n';
  services[svc->serviceType]=svc;
}

sci::cca::Port::pointer
InternalComponentModel::getFrameworkService(const std::string& type,
					    const std::string& componentName)
{
  //cerr<<"getFrameworkService #"<<1<<endl;
  map<string, InternalComponentDescription*>::iterator iter=services.find(type);
  if(iter == services.end())
    return sci::cca::Port::pointer(0);
  //cerr<<"getFrameworkService #"<<2<<endl;
  InternalComponentDescription* cd = iter->second;
  InternalComponentInstance* ci;
  if(cd->isSingleton){
    //cerr<<"getFrameworkService #"<<3<<endl;
    string cname = "internal: "+type;
    if(!cd->singleton_instance){
      cd->singleton_instance = (*cd->create)(framework, cname);
      framework->registerComponent(cd->singleton_instance, cname);
    }
    //cerr<<"getFrameworkService #"<<4<<endl;
    ci = cd->singleton_instance;
  } else {
    //cerr<<"getFrameworkService #"<<5<<endl;
    string cname = "internal: "+type+" for "+componentName;
    ci = (*cd->create)(framework, cname);
    //cerr<<"getFrameworkService #"<<6<<endl;
    framework->registerComponent(ci, cname);
    //cerr<<"getFrameworkService #"<<7<<endl;
  }
  ci->incrementUseCount();
  //cerr<<"getFrameworkService #"<<8<<endl;
  sci::cca::Port::pointer ptr = ci->getService(type);
  //cerr<<"getFrameworkService #"<<9<<endl;
  ptr->addReference();
  //cerr<<"getFrameworkService #"<<10<<endl;
  return ptr;
}

bool
InternalComponentModel::releaseFrameworkService(const std::string& type,
						const std::string& componentName)
{
  map<string, InternalComponentDescription*>::iterator iter=services.find(type);
  if(iter == services.end())
    return false;

  InternalComponentDescription* cd = iter->second;
  InternalComponentInstance* ci;
  if(cd->isSingleton){
    ci=cd->singleton_instance;
  } else {
    string cname = "internal: "+type+" for "+componentName;
    ci = dynamic_cast<InternalComponentInstance*>(framework->lookupComponent(cname));
    if(!ci)
      throw InternalError("Cannot find Service component of type: "+type+" for component "+componentName);
  }
  if(!ci->decrementUseCount())
    throw InternalError("Service released without correspond get");
  return true;
}

bool InternalComponentModel::haveComponent(const std::string& /*name*/)
{
  return false;
}

ComponentInstance* InternalComponentModel::createInstance(const std::string&,
							  const std::string&)
{
  return 0;
}

bool InternalComponentModel::destroyInstance(ComponentInstance *ic)
{
  cerr<<"Warning: I don't know how to destroy a internal component instance!"<<endl;
  return true;
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
