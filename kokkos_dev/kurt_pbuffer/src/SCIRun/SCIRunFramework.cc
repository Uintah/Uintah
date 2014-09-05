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
 *  SCIRunFramework.cc: An instance of the SCIRun framework
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <sci_defs/babel_defs.h>
#include <sci_defs/ruby_defs.h>
#include <sci_defs/vtk_defs.h>

#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/TypeMap.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/ComponentEvent.h>
#include <SCIRun/Internal/ComponentEventService.h>
#include <SCIRun/CCA/CCAComponentModel.h>

#if HAVE_BABEL 
 #if HAVE_RUBY
  #include <SCIRun/Bridge/BridgeComponentModel.h>
 #endif
 #include <SCIRun/Babel/BabelComponentModel.h>
#endif

#if HAVE_VTK
#include <SCIRun/Vtk/VtkComponentModel.h>
#endif

#include <SCIRun/Corba/CorbaComponentModel.h>
#include <SCIRun/Tao/TaoComponentModel.h>

#include <SCIRun/ComponentInstance.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/Util/NotFinished.h>

#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

#include "CCACommunicator.h"

namespace SCIRun {

SCIRunFramework::SCIRunFramework()
  //:d_slave_sema("Wait for a slave to regester Semaphore",0)
{
  models.push_back(internalServices = new InternalComponentModel(this));
  models.push_back(dflow = new SCIRunComponentModel(this));
  models.push_back(cca = new CCAComponentModel(this));
#if HAVE_BABEL 
 #if HAVE_RUBY
  models.push_back(new BridgeComponentModel(this));
 #endif
  models.push_back(babel = new BabelComponentModel(this));
#endif

#if HAVE_VTK
  models.push_back(vtk = new VtkComponentModel(this));
#endif

  models.push_back(corba = new CorbaComponentModel(this));
  models.push_back(tao = new TaoComponentModel(this));
}

SCIRunFramework::~SCIRunFramework()
{
  std::cerr << "~SCIRunFramework called!" << std::endl;
  abort();
}

sci::cca::Services::pointer
SCIRunFramework::getServices(const std::string& selfInstanceName,
    const std::string& selfClassName,
    const sci::cca::TypeMap::pointer& selfProperties)
{
  return cca->createServices(selfInstanceName, selfClassName, selfProperties);
}

sci::cca::ComponentID::pointer
SCIRunFramework::createComponentInstance(const std::string& name,
    const std::string& t,
    const sci::cca::TypeMap::pointer properties)
{
    std::string type = t;
    // See if the type is of the form:
    //   model:name
    // If so, extract the model and look up that component specifically.
    // Otherwise, look at all models for that component
    ComponentModel* mod = 0;
    unsigned int firstColon = type.find(':');
    if (firstColon < type.size()) {
        std::string modelName = type.substr(0, firstColon);
        type = type.substr(firstColon+1);
        // This is a linear search, but we don't expect to have
        // a ton of models, nor do we expect instantiation to
        // occur often
        for (std::vector<ComponentModel*>::iterator iter = models.begin();
                iter != models.end(); iter++) {
            ComponentModel* model = *iter;
            if (model->getPrefixName() == modelName) {
                mod = model;
                break;
            }
        }
    } else {
        int count = 0;
        for (std::vector<ComponentModel*>::iterator iter = models.begin();
                iter != models.end(); iter++) {
            ComponentModel* model = *iter;
            if (model->haveComponent(type)) {
                count++;
                mod = model;
            }
        }
        if (count > 1) {
            std::cerr << "More than one component model wants to build "
                      << type << std::endl;
            throw InternalError("Need CCA Exception here");
        }
    }

    if (!mod) {
        std::cerr << "No component model wants to build " << type << std::endl;
        return ComponentID::pointer(0);
    }
    ComponentInstance* ci;
    if (mod->getName() == "CCA") {
        ci = ((CCAComponentModel*)mod)->createInstance(name, type, properties);
    } else {
        ci = mod->createInstance(name, type);
    }
    if (!ci) {
        std::cerr << "Error: failed to create ComponentInstance" << std::endl;
        return ComponentID::pointer(0);
    }
    registerComponent(ci, name);
    sci::cca::ComponentID::pointer cid =
             ComponentID::pointer(new ComponentID(this, ci->instanceName));
    //emitComponentEvent(
    //    new ComponentEvent(sci::cca::ports::InstantiatePending, cid, properties)
    //); 

    //ComponentID::pointer()
    compIDs.push_back(cid);
    emitComponentEvent(
        new ComponentEvent(sci::cca::ports::ComponentInstantiated, cid, properties)
    ); 
    return compIDs[compIDs.size()-1];
}

void
SCIRunFramework::destroyComponentInstance(const sci::cca::ComponentID::pointer
                                               &cid, float timeout )
{
  //assuming no connections between this component and 
  //any other component

    // get component properties...
    emitComponentEvent(
        new ComponentEvent(sci::cca::ports::DestroyPending,
                           cid, sci::cca::TypeMap::pointer(0))
    ); 

  //#1 remove cid from compIDs
  for (unsigned i = 0; i<compIDs.size(); i++) {
    if (compIDs[i] == cid) {
      compIDs.erase(compIDs.begin()+i);
      break;
    }
  }
  
  //#2 unregister the component instance
  ComponentInstance *ci = unregisterComponent(cid->getInstanceName());
  
  //#3 find the associated component model
  std::string type = ci->className;
  // See if the type is of the form:
  //   model:name
  // If so, extract the model and look up that component specifically.
  // Otherwise, look at all models for that component
  ComponentModel* mod = 0;
  unsigned int firstColon = type.find(':');
  if (firstColon < type.size()) {
    std::string modelName = type.substr(0, firstColon);
    type = type.substr(firstColon+1);
    // This is a linear search, but we don't expect to have
    // a ton of models, nor do we expect instantiation to
    // occur often
    for (std::vector<ComponentModel*>::iterator iter = models.begin();
    iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if (model->getPrefixName() == modelName) {
        mod = model;
        break;
      }
    }
  } else {
    int count = 0;
    for (std::vector<ComponentModel*>::iterator iter = models.begin();
    iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if (model->haveComponent(type)) {
        count++;
        mod = model;
      }
    }
    if (count > 1) {
      std::cerr << "More than one component model wants to build " << type << '\n';
      throw InternalError("Need CCA Exception here");
    }
  }
  if (!mod) {
    std::cerr << "No component model matches component" << type << '\n';
    return;
  }
  //#4 destroy the component instance
  mod->destroyInstance(ci);

    emitComponentEvent(
        new ComponentEvent(sci::cca::ports::ComponentDestroyed,
                           cid, sci::cca::TypeMap::pointer(0))
    ); 

  return;
}


void
SCIRunFramework::registerComponent(ComponentInstance* ci, const std::string& name)
{
  std::string goodname = name;
  int count = 0;
  while(activeInstances.find(goodname) != activeInstances.end()) {
    std::ostringstream newname;
    newname << name << "_" << count++;
    goodname = newname.str();
  }
  ci->framework = this;
  ci->instanceName = goodname;
  activeInstances[ci->instanceName] = ci;
}

ComponentInstance*
SCIRunFramework::unregisterComponent(const std::string& instanceName)
{
  std::map<std::string, ComponentInstance*>::iterator found =
    activeInstances.find(instanceName);
  if (found != activeInstances.end()) {
    ComponentInstance *ci = found->second;
    activeInstances.erase(found);
    return ci;
  } else {
    std::cerr << "Error: component instance " << instanceName << " not found!"
              << std::endl;;
    return 0;
  }
}

ComponentInstance*
SCIRunFramework::lookupComponent(const std::string& name)
{
  std::map<std::string, ComponentInstance*>::iterator iter =
    activeInstances.find(name);
  if (iter == activeInstances.end()) {
    return 0;
  } else {
    return iter->second;
  }
}

sci::cca::ComponentID::pointer
SCIRunFramework::lookupComponentID(const std::string& componentInstanceName)
{
  for (unsigned i = 0; i<compIDs.size();i++) {
    if (componentInstanceName == compIDs[i]->getInstanceName()) {
      return compIDs[i];
    }
  }
  return sci::cca::ComponentID::pointer(0);
}
     


sci::cca::Port::pointer
SCIRunFramework::getFrameworkService(const std::string& type,
                     const std::string& componentName)
{
  return internalServices->getFrameworkService(type, componentName);
}

bool
SCIRunFramework::releaseFrameworkService(const std::string& type,
                     const std::string& componentName)
{
  return internalServices->releaseFrameworkService(type, componentName);
}

void
SCIRunFramework::listAllComponentTypes(
    std::vector<ComponentDescription*>& list, bool listInternal)
{
    for (std::vector<ComponentModel*>::iterator iter = models.begin();
            iter != models.end(); iter++) {
        (*iter)->listAllComponentTypes(list, listInternal);
    }
}

ComponentModel*
SCIRunFramework::lookupComponentModel(const std::string& name)
{
    for (std::vector<ComponentModel*>::iterator iter = models.begin();
            iter != models.end(); iter++) {
        if ((*iter)->getName() == name) {
            return *iter;
        }
    }
    return 0;
}

sci::cca::TypeMap::pointer
SCIRunFramework::createTypeMap()
{
    // return empty TypeMap according to CCA spec
    return sci::cca::TypeMap::pointer(new TypeMap);
}

void
SCIRunFramework::releaseServices(const sci::cca::Services::pointer& svc)
{
  // services handle (svc) no longer needed
  // delete this service from the framework
  cca->destroyServices(svc);
  // invalidate any ComponentIDs
  // and ConnectionIDs
}

void
SCIRunFramework::shutdownFramework()
{
    // throw CCAException if the framework has already been shutdown

    // cleanup models
    std::cerr << "SCIRunFramework::shutdownFramework not finished" << std::endl;
    for (std::vector<ComponentModel*>::iterator iter = models.begin();
            iter != models.end(); iter++) {
        delete *iter;
    }
    // shutdown proxy frameworks
}

sci::cca::AbstractFramework::pointer
SCIRunFramework::createEmptyFramework()
{
    std::cerr << "SCIRunFramework::createEmptyFramework not finished" << std::endl;
    return sci::cca::AbstractFramework::pointer(0);
}

int
SCIRunFramework::registerLoader(const ::std::string& loaderName, const SSIDL::array1<std::string>& slaveURLs)
{
std::cerr << "SCIRunFramework::registerLoader" << std::endl;
  resourceReference* rr = new resourceReference(loaderName, slaveURLs);
  rr->print();
  cca->addLoader(rr);
  
  //d_slave_sema.up(); //now wake UP
  return 0;
}

int
SCIRunFramework::unregisterLoader(const std::string &loaderName)
{
  cca->removeLoader(loaderName);
  return 0;
}

void
SCIRunFramework::emitComponentEvent(ComponentEvent* event)
{
  sci::cca::ports::ComponentEventService::pointer service =
    pidl_cast<sci::cca::ports::ComponentEventService::pointer>(
        getFrameworkService("cca.ComponentEventService", "")
    );
  if (service.isNull()) {
    std::cerr << "Error: could not find ComponentEventService" << std::endl;
  } else {
    ComponentEventService* ces =
        dynamic_cast<ComponentEventService*>(service.getPointer());
    sci::cca::ports::ComponentEvent::pointer ce = ComponentEvent::pointer(event);
      ces->emitComponentEvent(ce);
      releaseFrameworkService("cca.ComponentEventService", "");
  }
}

#if 0
// do not delete the following 2 methods
/* 
void SCIRunFramework::share(const sci::cca::Services::pointer &svc)
{
  Thread* t = new Thread(new CCACommunicator(this,svc), "SCIRun CCA Communicator");
  t->detach(); 
}

//used for remote creation of a CCA component
//return URL of the new component
std::string
SCIRunFramework::createComponent(const std::string& name, const std::string& t)
{
  std::string type=t;
  // See if the type is of the form:
  //   model:name
  // If so, extract the model and look up that component specifically.
  // Otherwise, look at all models for that component
  ComponentModel* mod=0;
  unsigned int firstColon = type.find(':');
  if(firstColon < type.size()){
    std::string modelName = type.substr(0, firstColon);
    type = type.substr(firstColon+1);
    // This is a linear search, but we don't expect to have
    // a ton of models, nor do we expect instantiation to
    // occur often
    for(std::vector<ComponentModel*>::iterator iter=models.begin();
    iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if(model->getPrefixName() == modelName){
    mod=model;
    break;
      }
    }
  } 
  else {
    int count=0;
    for(std::vector<ComponentModel*>::iterator iter=models.begin();
    iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if(model->haveComponent(type)){
    count++;
    mod=model;
      }
    }
    if(count > 1){
      std::cerr << "More than one component model wants to build " << type << '\n';
      throw InternalError("Need CCA Exception here");
    }
  }
  if(!mod){
    std::cerr << "No component model wants to build " << type << '\n';
    return "";
  }
  return  mod->createComponent(name, type);
}
*/
#endif


} // end namespace SCIRun









