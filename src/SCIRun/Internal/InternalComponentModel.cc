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
#include <SCIRun/Internal/ConnectionEventService.h>
#include <SCIRun/Internal/ComponentRegistry.h>
#include <SCIRun/Internal/FrameworkProperties.h>
#include <SCIRun/Internal/FrameworkProxyService.h>
#include <SCIRun/SCIRunFramework.h>

#ifdef BUILD_DATAFLOW
 #include <SCIRun/Dataflow/DataflowScheduler.h>
#endif
#include <iostream>

#ifndef DEBUG
 #define DEBUG 0
#endif

namespace SCIRun {

InternalComponentModel::InternalComponentModel(SCIRunFramework* framework)
  : ComponentModel("internal"), framework(framework)
{
    addService(new InternalComponentDescription(this, "cca.BuilderService",
        &BuilderService::create, true));
    addService(new InternalComponentDescription(this, "cca.ComponentRepository",
        &ComponentRegistry::create, true));
    addService(new InternalComponentDescription(this, "cca.ComponentEventService",
        &ComponentEventService::create, true));
    addService(new InternalComponentDescription(this, "cca.ConnectionEventService",
        &ConnectionEventService::create, true));
#ifdef BUILD_DATAFLOW
    addService(new InternalComponentDescription(this, "cca.DataflowScheduler",
        &DataflowScheduler::create, true));
#endif
    addService(new InternalComponentDescription(this, "cca.FrameworkProperties",
        &FrameworkProperties::create, true));
    addService(new InternalComponentDescription(this, "cca.FrameworkProxyService",
        &FrameworkProxyService::create, true));
}

InternalComponentModel::~InternalComponentModel()
{
    for (std::map<std::string, InternalComponentDescription*>::iterator
        iter=services.begin(); iter != services.end(); iter++) {
        delete iter->second;
    }
}

void InternalComponentModel::addService(InternalComponentDescription* svc)
{
    if (services.find(svc->serviceType) != services.end()) {
    std::cerr << "WARNING: duplicate internal service: " <<
        svc->serviceType << std::endl;
    }
    services[svc->serviceType] = svc;
}

sci::cca::Port::pointer
InternalComponentModel::getFrameworkService(const std::string& type,
                        const std::string& componentName)
{
    //std::cerr<<"getFrameworkService #"<<1<<std::endl;
    std::map<std::string, InternalComponentDescription*>::iterator iter =
    services.find(type);
    if (iter == services.end()) {
    return sci::cca::Port::pointer(0);
    }
    //std::cerr<<"getFrameworkService #"<<2<<std::endl;
    InternalComponentDescription* cd = iter->second;
    InternalComponentInstance* ci;
    if (cd->isSingleton) {
    //std::cerr<<"getFrameworkService #"<<3<<std::endl;
    std::string cname = "internal: "+type;
    if (!cd->singleton_instance) {
        cd->singleton_instance = (*cd->create)(framework, cname);
        framework->registerComponent(cd->singleton_instance, cname);
    }
    //std::cerr<<"getFrameworkService #"<<4<<std::endl;
    ci = cd->singleton_instance;
    } else {
    //std::cerr<<"getFrameworkService #"<<5<<std::endl;
    std::string cname = "internal: " + type + " for " + componentName;
    ci = (*cd->create)(framework, cname);
    //std::cerr<<"getFrameworkService #"<<6<<std::endl;
    framework->registerComponent(ci, cname);
    //std::cerr<<"getFrameworkService #"<<7<<std::endl;
    }
    ci->incrementUseCount();
    //std::cerr<<"getFrameworkService #"<<8<<std::endl;
    sci::cca::Port::pointer ptr = ci->getService(type);
    //std::cerr<<"getFrameworkService #"<<9<<std::endl;
    ptr->addReference();
    //std::cerr<<"getFrameworkService #"<<10<<std::endl;
    return ptr;
}

bool
InternalComponentModel::releaseFrameworkService(const std::string& type,
                                                const std::string& componentName)
{
    std::map<std::string, InternalComponentDescription*>::iterator iter =
    services.find(type);
    if (iter == services.end()) { return false; }
  
    InternalComponentDescription* cd = iter->second;
    InternalComponentInstance* ci;
    if (cd->isSingleton) {
        ci = cd->singleton_instance;
    } else {
        std::string cname = "internal: " + type + " for " + componentName;
        ci = dynamic_cast<InternalComponentInstance*>(
            framework->lookupComponent(cname));
        if (!ci) {
            throw InternalError("Cannot find Service component of type: " +
                type + " for component " + componentName, __FILE__, __LINE__);
        }
    }
    if (!ci->decrementUseCount()) {
        throw InternalError("Service released without correspond get",
            __FILE__, __LINE__);
    }
    return true;
}

bool InternalComponentModel::haveComponent(const std::string& /*name*/)
{
#if DEBUG
  std::cerr << "Error: InternalComponentModel does not implement haveComponent"
        << std::endl;
#endif
    return false;
}

void InternalComponentModel::destroyComponentList()
{
#if DEBUG
  std::cerr << "Error: InternalComponentModel does not implement destroyComponentList"
            << std::endl;
#endif
}

void InternalComponentModel::buildComponentList()
{
#if DEBUG
  std::cerr << "Error: InternalComponentModel does not implement buildComponentList"
            << std::endl;
#endif
}

ComponentInstance* InternalComponentModel::createInstance(const std::string&,
                                                          const std::string&)
{
    return 0;
}

bool InternalComponentModel::destroyInstance(ComponentInstance *ic)
{
#if DEBUG
    std::cerr << "Warning: I don't know how to destroy a internal component instance!" << std::endl;
#endif
    return true;
}

std::string InternalComponentModel::getName() const
{
    return "Internal";
}

void
InternalComponentModel::listAllComponentTypes(
    std::vector<ComponentDescription*>& list, bool listInternal)
{
    if (listInternal) {
        for (std::map<std::string, InternalComponentDescription*>::iterator
            iter = services.begin(); iter != services.end(); iter++) {
            list.push_back(iter->second);
        }
    }
}

} // end namespace SCIRun
