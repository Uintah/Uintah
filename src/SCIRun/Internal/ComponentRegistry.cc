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

namespace SCIRun {

/** \class ComponentClassDescriptionAdapter
 *
 * An adaptor class that converts the SCIRun ComponentDescription interface
 * into the standard CCA ComponentClassDescription interface.
 * 
 */
class ComponentClassDescriptionAdapter
  : public sci::cca::ComponentClassDescription
{
public:
  ComponentClassDescriptionAdapter(const ComponentDescription*);
  ~ComponentClassDescriptionAdapter();

  /** Returns the instance name of the component class provided   */
  virtual std::string getComponentClassName();

  /** A nonstandard method.  This returns the name of the component model
      associated with the component class. */
  virtual std::string getComponentModelName();
  /** */
  virtual std::string getLoaderName();
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

std::string ComponentClassDescriptionAdapter::getComponentClassName()
{
  return cd->getType();
}

std::string ComponentClassDescriptionAdapter::getComponentModelName()
{
  return cd->getModel()->getName();
}

std::string ComponentClassDescriptionAdapter::getLoaderName()
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
  std::cerr << "Registry destroyed..." << std::endl;
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

SSIDL::array1<sci::cca::ComponentClassDescription::pointer>
ComponentRegistry::getAvailableComponentClasses()
{
  std::vector<ComponentDescription*> list;
  framework->listAllComponentTypes(list, false);
  SSIDL::array1<sci::cca::ComponentClassDescription::pointer> ccalist;

  for(std::vector<ComponentDescription*>::iterator iter = list.begin();
      iter != list.end(); iter++) {
    ccalist.push_back(sci::cca::ComponentClassDescription::pointer(
                          new ComponentClassDescriptionAdapter(*iter)));
  }
  return ccalist;
}

void ComponentRegistry::addComponentClass(const std::string& componentClassName)
{
    ComponentModel* cm = framework->lookupComponentModel(componentClassName);
    if (cm == 0) {
        throw sci::cca::CCAException::pointer(new CCAException("Unknown component class"));
    } else {
        cm->buildComponentList();
    }
}

} // end namespace SCIRun
