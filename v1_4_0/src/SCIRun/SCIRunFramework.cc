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
#include <SCIRun/ComponentInstance.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <sstream>
using namespace std;
using namespace SCIRun;

SCIRunFramework::SCIRunFramework()
{
  models.push_back(internalServices=new InternalComponentModel(this));
  models.push_back(new SCIRunComponentModel(this));
  models.push_back(cca=new CCAComponentModel(this));
}

SCIRunFramework::~SCIRunFramework()
{
  cerr << "~SCIRunFramewrok called!\n";
  for(vector<ComponentModel*>::iterator iter=models.begin();
      iter != models.end(); iter++)
    delete *iter;
}

gov::cca::Services::pointer
SCIRunFramework::getServices(const std::string& selfInstanceName,
			     const std::string& selfClassName,
			     const gov::cca::TypeMap::pointer& selfProperties)
{
  return cca->createServices(selfInstanceName, selfClassName, selfProperties);
}

gov::cca::ComponentID::pointer
SCIRunFramework::createComponentInstance(const std::string& name,
					 const std::string& t)
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
    cerr << "!mod\n";
    return ComponentID::pointer(0);
  }
  ComponentInstance* ci = mod->createInstance(name, type);
  if(!ci){
    cerr << "!ci\n";
    return ComponentID::pointer(0);
  }
  registerComponent(ci, name);
  return ComponentID::pointer(new ComponentID(this, name));
}

void SCIRunFramework::registerComponent(ComponentInstance* ci,
					const std::string& name)
{
  string goodname = name;
  int count=0;
  while(activeInstances.find(goodname) != activeInstances.end()){
    ostringstream newname;
    newname << name << "_" << count++ << "\n";
    goodname=newname.str();
  }
  ci->framework=this;
  ci->instanceName = goodname;
  activeInstances[ci->instanceName] = ci;
  // Get the component event service and send a creation event
  cerr << "Should register a creation event for component " << name << '\n';
}

ComponentInstance*
SCIRunFramework::getComponent(const std::string& name)
{
  map<string, ComponentInstance*>::iterator iter = activeInstances.find(name);
  if(iter == activeInstances.end())
    return 0;
  else
    return iter->second;
}

gov::cca::Port::pointer
SCIRunFramework::getFrameworkService(const std::string& type)
{
  return internalServices->getFrameworkService(type);
}

void
SCIRunFramework::listAllComponentTypes(vector<ComponentDescription*>& list,
				       bool listInternal)
{
  for(vector<ComponentModel*>::iterator iter=models.begin();
      iter != models.end(); iter++)
    (*iter)->listAllComponentTypes(list, listInternal);
}

gov::cca::TypeMap::pointer SCIRunFramework::createTypeMap()
{
  cerr << "SCIRunFramework::createTypeMap not finished\n";
  return gov::cca::TypeMap::pointer(0);
}

void SCIRunFramework::releaseServices(const gov::cca::Services::pointer& svc)
{
  cerr << "SCIRunFramework::releaseServices not finished\n";
}

void SCIRunFramework::shutdownFramework()
{
  cerr << "SCIRunFramework::shutdownFramework not finished\n";
}

gov::cca::AbstractFramework::pointer SCIRunFramework::createEmptyFramework()
{
  cerr << "SCIRunFramework::createEmptyFramework not finished\n";
  return gov::cca::AbstractFramework::pointer(0);
}


