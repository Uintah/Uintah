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

class ComponentClassDescriptionAdapter : public sci::cca::ComponentClassDescription {
public:
  ComponentClassDescriptionAdapter(const ComponentDescription*);
  ~ComponentClassDescriptionAdapter();

  virtual string getComponentClassName();
  virtual string getComponentModelName();
  virtual string getLoaderName();
private:
  const ComponentDescription* cd;
};

ComponentClassDescriptionAdapter::ComponentClassDescriptionAdapter(const ComponentDescription* cd)
  : cd(cd)
{
}

ComponentClassDescriptionAdapter::~ComponentClassDescriptionAdapter()
{
}

string ComponentClassDescriptionAdapter::getComponentClassName()
{
  return cd->getType();
}

string ComponentClassDescriptionAdapter::getComponentModelName()
{
  return cd->getModel()->getName();
}

string ComponentClassDescriptionAdapter::getLoaderName()
{
  return cd->getLoaderName();
}


ComponentRegistry::ComponentRegistry(SCIRunFramework* framework,
				     const std::string& name)
  : InternalComponentInstance(framework, name, "internal:ComponentRegistry")
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
  n->addReference();
  return n;
}

sci::cca::Port::pointer ComponentRegistry::getService(const std::string&)
{
  return sci::cca::Port::pointer(this);
}

SSIDL::array1<sci::cca::ComponentClassDescription::pointer> ComponentRegistry::getAvailableComponentClasses()
{
  vector<ComponentDescription*> list;
  framework->listAllComponentTypes(list, false);
  SSIDL::array1<sci::cca::ComponentClassDescription::pointer> ccalist;
  for(vector<ComponentDescription*>::iterator iter = list.begin();
      iter != list.end(); iter++){
    ccalist.push_back(sci::cca::ComponentClassDescription::pointer(new ComponentClassDescriptionAdapter(*iter)));
  }
  return ccalist;
}
