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
 *  ComponentRegistry.cc: Implementation of CCA ComponentRegistry for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Internal/ComponentRegistry.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/ComponentDescription.h>
#include <SCIRun/ComponentModel.h>
#include <SCIRun/CCA/CCAException.h>
#include <SCIRun/PortInstance.h>
#include <SCIRun/CCA/ComponentID.h>
#include <SCIRun/ComponentInstance.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

class ComponentDescriptionAdapter : public gov::cca::ComponentDescription_interface {
public:
  ComponentDescriptionAdapter(const ComponentDescription*);
  ~ComponentDescriptionAdapter();

  virtual string getType();
  virtual string getModelName();
private:
  const ComponentDescription* cd;
};

ComponentDescriptionAdapter::ComponentDescriptionAdapter(const ComponentDescription* cd)
  : cd(cd)
{
}

ComponentDescriptionAdapter::~ComponentDescriptionAdapter()
{
}

string ComponentDescriptionAdapter::getType()
{
  return cd->getType();
}

string ComponentDescriptionAdapter::getModelName()
{
  return cd->getModel()->getName();
}

ComponentRegistry::ComponentRegistry(SCIRunFramework* framework,
			       const std::string& name)
  : InternalComponentInstance(framework, name)
{
}

ComponentRegistry::~ComponentRegistry()
{
  cerr << "Registry destroyed...\n";
}

InternalComponentInstance* ComponentRegistry::create(SCIRunFramework* framework,
						  const std::string& name)
{
  ComponentRegistry* n = new ComponentRegistry(framework, name);
  n->_addReference();
  return n;
}

gov::cca::Port ComponentRegistry::getService(const std::string&)
{
  return this;
}

CIA::array1<gov::cca::ComponentDescription> ComponentRegistry::listAllComponentTypes()
{
  vector<ComponentDescription*> list;
  framework->listAllComponentTypes(list, false);
  CIA::array1<gov::cca::ComponentDescription> ccalist;
  for(vector<ComponentDescription*>::iterator iter = list.begin();
      iter != list.end(); iter++){
    ccalist.push_back(new ComponentDescriptionAdapter(*iter));
  }
  return ccalist;
}
