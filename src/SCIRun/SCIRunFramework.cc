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
 *  SCIRunFramework.cc: An instance of the SCIRun framework
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Dataflow/SCIRunComponentModel.h>
#include <SCIRun/CCA/CCAComponentModel.h>
#if HAVE_BABEL
#include <SCIRun/Babel/BabelComponentModel.h>
#endif
#include <SCIRun/ComponentInstance.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <iostream>
#include <sstream>
#include <Core/Util/NotFinished.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>


#include "CCACommunicator.h"

using namespace std;
using namespace SCIRun;

SCIRunFramework::SCIRunFramework()
  //:d_slave_sema("Wait for a slave to regester Semaphore",0)
{
  models.push_back(internalServices=new InternalComponentModel(this));
  models.push_back(new SCIRunComponentModel(this));
  models.push_back(cca=new CCAComponentModel(this));
#if HAVE_BABEL
  models.push_back(babel=new BabelComponentModel(this));
#endif
}

SCIRunFramework::~SCIRunFramework()
{
  cerr << "~SCIRunFramewrok called!\n";
  abort();
  for(vector<ComponentModel*>::iterator iter=models.begin();
      iter != models.end(); iter++)
    delete *iter;
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
  string type=t;
  // See if the type is of the form:
  //   model:name
  // If so, extract the model and look up that component specifically.
  // Otherwise, look at all models for that component
  ComponentModel* mod=0;
  unsigned int firstColon = type.find(':');
  if(firstColon < type.size()){
    string modelName = type.substr(0, firstColon);
    type = type.substr(firstColon+1);
    // This is a linear search, but we don't expect to have
    // a ton of models, nor do we expect instantiation to
    // occur often
    for(vector<ComponentModel*>::iterator iter=models.begin();
	iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if(model->prefixName == modelName){
	mod=model;
	break;
      }
    }
  } else {
    int count=0;
    for(vector<ComponentModel*>::iterator iter=models.begin();
	iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if(model->haveComponent(type)){
	count++;
	mod=model;
      }
    }
    if(count > 1){
      cerr << "More than one component model wants to build " << type << '\n';
      throw InternalError("Need CCA Exception here");
    }
  }
  if(!mod){
    cerr << "No component model wants to build " << type << '\n';
    return ComponentID::pointer(0);
  }
  ComponentInstance* ci;
  if(mod->getName()=="CCA")
    ci = ((CCAComponentModel*)mod)->createInstance(name, type, properties);
  else{
    ci = mod->createInstance(name, type);
  }
  if(!ci){
    cerr<<"Error: failed to create ComponentInstance"<<endl;
    return ComponentID::pointer(0);
  }
  registerComponent(ci, name);
  compIDs.push_back(ComponentID::pointer(new ComponentID(this, ci->instanceName)));
  return compIDs[compIDs.size()-1];
}

void SCIRunFramework::destroyComponentInstance(const sci::cca::ComponentID::pointer &cid, float timeout )
{
  //assuming no connections between this component and 
  //any other component

  //#1 remove cid from compIDs
  for(unsigned i=0; i<compIDs.size(); i++){
    if(compIDs[i]==cid){
      compIDs.erase(compIDs.begin()+i);
      break;
    }
  }

  //#2 unregister the component instance
  ComponentInstance *ci=unregisterComponent(cid->getInstanceName());
  
  //#3 find the associated component model
  string type=ci->className;
  // See if the type is of the form:
  //   model:name
  // If so, extract the model and look up that component specifically.
  // Otherwise, look at all models for that component
  ComponentModel* mod=0;
  unsigned int firstColon = type.find(':');
  if(firstColon < type.size()){
    string modelName = type.substr(0, firstColon);
    type = type.substr(firstColon+1);
    // This is a linear search, but we don't expect to have
    // a ton of models, nor do we expect instantiation to
    // occur often
    for(vector<ComponentModel*>::iterator iter=models.begin();
	iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if(model->prefixName == modelName){
	mod=model;
	break;
      }
    }
  } 
  else {
    int count=0;
    for(vector<ComponentModel*>::iterator iter=models.begin();
	iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if(model->haveComponent(type)){
	count++;
	mod=model;
      }
    }
    if(count > 1){
      cerr << "More than one component model wants to build " << type << '\n';
      throw InternalError("Need CCA Exception here");
    }
  }
  if(!mod){
    cerr << "No component model matches component" << type << '\n';
    return;
  }
  //#4 destroy the component instance
  mod->destroyInstance(ci);
  return;
}


void SCIRunFramework::registerComponent(ComponentInstance* ci,
					const std::string& name)
{
  string goodname = name;
  int count=0;
  while(activeInstances.find(goodname) != activeInstances.end()){
    ostringstream newname;
    newname << name << "_" << count++;
    goodname=newname.str();
  }
  ci->framework=this;
  ci->instanceName = goodname;
  activeInstances[ci->instanceName] = ci;
  // Get the component event service and send a creation event
  cerr << "TODO: register a creation event for component " << name << '\n';
}

ComponentInstance * SCIRunFramework::unregisterComponent(const std::string& instanceName)
{
  std::map<std::string, ComponentInstance*>::iterator found=activeInstances.find(instanceName);
  if(found != activeInstances.end()){
    ComponentInstance *ci=found->second;
    activeInstances.erase(found);
    return ci;
  }
  else{
    cerr<<"Error: component instance "<<instanceName<<" not found!"<<endl;
    return 0;
  }
}

ComponentInstance*
SCIRunFramework::lookupComponent(const std::string& name)
{
  map<string, ComponentInstance*>::iterator iter = activeInstances.find(name);
  if(iter == activeInstances.end())
    return 0;
  else
    return iter->second;
}

sci::cca::ComponentID::pointer
SCIRunFramework::lookupComponentID(const std::string& componentInstanceName)
{
  for(unsigned i=0; i<compIDs.size();i++)
    if(componentInstanceName==compIDs[i]->getInstanceName()) 
      return compIDs[i];
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
SCIRunFramework::listAllComponentTypes(vector<ComponentDescription*>& list,
				       bool listInternal)
{
  for(vector<ComponentModel*>::iterator iter=models.begin();
      iter != models.end(); iter++)
    (*iter)->listAllComponentTypes(list, listInternal);
}

sci::cca::TypeMap::pointer SCIRunFramework::createTypeMap()
{
  cerr << "SCIRunFramework::createTypeMap not finished\n";
  return sci::cca::TypeMap::pointer(0);
}

void SCIRunFramework::releaseServices(const sci::cca::Services::pointer& svc)
{
  cerr << "SCIRunFramework::releaseServices not finished\n";
}

void SCIRunFramework::shutdownFramework()
{
  cerr << "SCIRunFramework::shutdownFramework not finished\n";
}

sci::cca::AbstractFramework::pointer SCIRunFramework::createEmptyFramework()
{
  cerr << "SCIRunFramework::createEmptyFramework not finished\n";
  return sci::cca::AbstractFramework::pointer(0);
}

int SCIRunFramework::registerLoader(const ::std::string& loaderName, const ::SSIDL::array1< ::std::string>& slaveURLs)
{
  resourceReference* rr = new resourceReference(loaderName, slaveURLs);
  rr->print();
  cca->addLoader(rr);
  
  //d_slave_sema.up(); //now wake UP
  return 0;
}

int SCIRunFramework::unregisterLoader(const std::string &loaderName)
{
  cca->removeLoader(loaderName);
  return 0;
}


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
  string type=t;
  // See if the type is of the form:
  //   model:name
  // If so, extract the model and look up that component specifically.
  // Otherwise, look at all models for that component
  ComponentModel* mod=0;
  unsigned int firstColon = type.find(':');
  if(firstColon < type.size()){
    string modelName = type.substr(0, firstColon);
    type = type.substr(firstColon+1);
    // This is a linear search, but we don't expect to have
    // a ton of models, nor do we expect instantiation to
    // occur often
    for(vector<ComponentModel*>::iterator iter=models.begin();
	iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if(model->prefixName == modelName){
	mod=model;
	break;
      }
    }
  } 
  else {
    int count=0;
    for(vector<ComponentModel*>::iterator iter=models.begin();
	iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if(model->haveComponent(type)){
	count++;
	mod=model;
      }
    }
    if(count > 1){
      cerr << "More than one component model wants to build " << type << '\n';
      throw InternalError("Need CCA Exception here");
    }
  }
  if(!mod){
    cerr << "No component model wants to build " << type << '\n';
    return "";
  }
  return  mod->createComponent(name, type);
}

*/











