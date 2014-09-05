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
 *  InternalComponentModel.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <sci_defs/dataflow_defs.h>

#include <SCIRun/Internal/FrameworkInternalException.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceDescription.h>
#include <SCIRun/Internal/BuilderService.h>
#include <SCIRun/Internal/ComponentEventService.h>
#include <SCIRun/Internal/ConnectionEventService.h>
#include <SCIRun/Internal/ComponentRegistry.h>
#include <SCIRun/Internal/FrameworkProperties.h>
#include <SCIRun/Internal/FrameworkProxyService.h>
#include <SCIRun/SCIRunFramework.h>

#if BUILD_DATAFLOW
 #include <SCIRun/Dataflow/DataflowScheduler.h>
#endif
#include <iostream>

#ifndef DEBUG
 #define DEBUG 0
#endif

namespace SCIRun {

InternalComponentModel::InternalComponentModel(SCIRunFramework* framework)
  : ComponentModel("internal", framework),
    lock_frameworkServices("InternalComponentModel::frameworkServices lock")
{
    addService(new InternalFrameworkServiceDescription(this, "cca.BuilderService", &BuilderService::create));
    addService(new InternalFrameworkServiceDescription(this, "cca.ComponentRepository", &ComponentRegistry::create));
    addService(new InternalFrameworkServiceDescription(this, "cca.ComponentEventService", &ComponentEventService::create));
    addService(new InternalFrameworkServiceDescription(this, "cca.ConnectionEventService", &ConnectionEventService::create));
#if BUILD_DATAFLOW
    addService(new InternalFrameworkServiceDescription(this, "cca.DataflowScheduler", &DataflowScheduler::create));
#endif
    addService(new InternalFrameworkServiceDescription(this, "cca.FrameworkProperties", &FrameworkProperties::create));
    addService(new InternalFrameworkServiceDescription(this, "cca.FrameworkProxyService", &FrameworkProxyService::create));
}

InternalComponentModel::~InternalComponentModel()
{
  SCIRun::Guard g1(&lock_frameworkServices);
  for (FrameworkServicesMap::iterator iter=frameworkServices.begin(); iter != frameworkServices.end(); iter++) {
    delete iter->second;
  }
  framework = 0;
}

void InternalComponentModel::addService(InternalFrameworkServiceDescription* desc)
{
  SCIRun::Guard g1(&lock_frameworkServices);
  if (frameworkServices.find(desc->getType()) != frameworkServices.end())
    throw FrameworkInternalException("add duplicate service ["+desc->getType()+"]");
  frameworkServices[desc->getType()] = desc;
}

sci::cca::Port::pointer
InternalComponentModel::getFrameworkService(const std::string& type,
					    const std::string& componentName)
{
  InternalServiceInstance *service = 0;
  sci::cca::Port::pointer port(0);

  lock_frameworkServices.lock();
  FrameworkServicesMap::const_iterator fwkServiceDesc = frameworkServices.find(type);
  lock_frameworkServices.unlock();
  if ( fwkServiceDesc != frameworkServices.end() )
    service = fwkServiceDesc->second->get(framework);

  if ( service ) {
    service->incrementUseCount();
    port = service->getService(type);
  }

  return port;
}

#if 0
/////////////////////////////////////////////////////////////////////////////
//
//  Former contents of InternalComponentModel::getFrameworkService
//
//     //std::cerr<<"getFrameworkService #"<<1<<std::endl;
//     std::map<std::string, InternalComponentDescription*>::iterator iter =
//     services.find(type);
//     if (iter == services.end()) {
//     return sci::cca::Port::pointer(0);
//     }
//     //std::cerr<<"getFrameworkService #"<<2<<std::endl;
//     InternalComponentDescription* cd = iter->second;
//     InternalComponentInstance* ci;
//     if (cd->isSingleton) {
//     //std::cerr<<"getFrameworkService #"<<3<<std::endl;
//     std::string cname = "internal: "+type;
//     if (!cd->singleton_instance) {
//         cd->singleton_instance = (*cd->create)(framework, cname);
//         framework->registerComponent(cd->singleton_instance, cname);
//     }
//     //std::cerr<<"getFrameworkService #"<<4<<std::endl;
//     ci = cd->singleton_instance;
//     } else {
//     //std::cerr<<"getFrameworkService #"<<5<<std::endl;
//     std::string cname = "internal: " + type + " for " + componentName;
//     ci = (*cd->create)(framework, cname);
//     //std::cerr<<"getFrameworkService #"<<6<<std::endl;
//     framework->registerComponent(ci, cname);
//     //std::cerr<<"getFrameworkService #"<<7<<std::endl;
//     }
//     ci->incrementUseCount();
//     //std::cerr<<"getFrameworkService #"<<8<<std::endl;
//     sci::cca::Port::pointer ptr = ci->getService(type);
//     //std::cerr<<"getFrameworkService #"<<9<<std::endl;
//     ptr->addReference();
//     //std::cerr<<"getFrameworkService #"<<10<<std::endl;
//     return ptr;
// }
/////////////////////////////////////////////////////////////////////////////
#endif


bool
InternalComponentModel::releaseFrameworkService(const std::string& type,
						const std::string& componentName)
{
  lock_frameworkServices.lock();
  FrameworkServicesMap::iterator iter = frameworkServices.find(type);
  lock_frameworkServices.unlock();

  if (iter == frameworkServices.end()) {
    return false;
  }

  iter->second->release(framework);

  return true;
}

#if 0
/////////////////////////////////////////////////////////////////////////////
//   InternalComponentDescription* cd = iter->second;
//   InternalComponentInstance* ci;
//   if (cd->isSingleton) {
//     ci = cd->singleton_instance;
//   } else {
//     std::string cname = "internal: " + type + " for " + componentName;
//     ci = dynamic_cast<InternalComponentInstance*>(
//						  framework->lookupComponent(cname));
//     if (!ci) {
//       throw InternalError("Cannot find Service component of type: " +
//			  type + " for component " + componentName, __FILE__, __LINE__);
//     }
//   }
//   if (!ci->decrementUseCount()) {
//     throw InternalError("Service released without correspond get",
//			__FILE__, __LINE__);
//   }
//   return true;
// }
/////////////////////////////////////////////////////////////////////////////
#endif

bool InternalComponentModel::haveComponent(const std::string& /*name*/)
{
#if DEBUG
  std::cerr << "Warning: InternalComponentModel does not implement haveComponent"
	<< std::endl;
#endif
    return false;
}

void InternalComponentModel::destroyComponentList()
{
#if DEBUG
  std::cerr << "Warning: InternalComponentModel does not implement destroyComponentList"
	    << std::endl;
#endif
}

void InternalComponentModel::buildComponentList(const StringVector& files)
{
#if DEBUG
  std::cerr << "Warning: InternalComponentModel does not implement buildComponentList"
	    << std::endl;
#endif
}

void InternalComponentModel::setComponentDescription(const std::string& type, const std::string& library)
{
#if DEBUG
  std::cerr << "Warning: InternalComponentModel does not implement setComponentDescription"
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

const std::string InternalComponentModel::getName() const
{
    return "Internal";
}

void
InternalComponentModel::listAllComponentTypes(
   std::vector<ComponentDescription*>& list, bool listInternal)
{
  SCIRun::Guard g1(&lock_frameworkServices);
  if (listInternal) {
    for (FrameworkServicesMap::iterator iter = frameworkServices.begin(); iter != frameworkServices.end(); iter++) {
      list.push_back(iter->second);
    }
  }
}

} // end namespace SCIRun
