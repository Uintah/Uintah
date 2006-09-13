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
 *  TaoComponentModel.cc:
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   May 2005
 *
 */

#include <Framework/Tao/Component.h>
#include <Framework/Tao/TaoComponentModel.h>
#include <Framework/Tao/TaoComponentDescription.h>
#include <Framework/Tao/TaoComponentInstance.h>
#include <Framework/SCIRunFramework.h>
#include <Core/Util/soloader.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Framework/resourceReference.h>
#include <string>

#include <iostream>

namespace SCIRun {

const std::string TaoComponentModel::DEFAULT_XML_PATH("/CCA/Components/TAO/xml");


TaoComponentModel::TaoComponentModel(SCIRunFramework* framework,
                                     const StringVector& xmlPaths)
  : ComponentModel("tao", framework),
    lock_components("TaoComponentModel::components lock")
{
  buildComponentList(xmlPaths);
}

TaoComponentModel::~TaoComponentModel()
{
  destroyComponentList();
}

void TaoComponentModel::destroyComponentList()
{
  SCIRun::Guard g1(&lock_components);

  for (componentDB_type::iterator iter=components.begin();
       iter != components.end(); iter++) {
    delete iter->second;
  }
  components.clear();
}

void TaoComponentModel::buildComponentList(const StringVector& files)
{
  destroyComponentList();

  if (files.empty()) {
    StringVector xmlPaths_;
    getXMLPaths(framework, xmlPaths_);
    for (StringVector::iterator iter = xmlPaths_.begin(); iter != xmlPaths_.end(); iter++) {
      parseComponentModelXML(*iter, this);
    }
  } else {
    for (StringVector::const_iterator iter = files.begin(); iter != files.end(); iter++) {
      parseComponentModelXML(*iter, this);
    }
  }
}

void
TaoComponentModel::setComponentDescription(const std::string& type, const std::string& library)
{
  TaoComponentDescription* cd = new TaoComponentDescription(this, type, library);
  Guard g(&lock_components);
  componentDB_type::iterator iter = components.find(cd->getType());
  if (iter != components.end()) {
    std::cerr << "WARNING: Multiple definitions exist for " << cd->getType() << std::endl;
  } else {
    components[cd->getType()] = cd;
  }
}

sci::cca::TaoServices::pointer
TaoComponentModel::createServices(const std::string& instanceName,
                                  const std::string& className,
                                  const sci::cca::TypeMap::pointer& properties)
{
  TaoComponentInstance* ci =
    new TaoComponentInstance(framework, instanceName, className,
                             properties, 0);
  framework->registerComponent(ci, instanceName);
  ci->addReference();
  return sci::cca::TaoServices::pointer(ci);
}

bool TaoComponentModel::destroyServices(const sci::cca::TaoServices::pointer& svc)
{
  TaoComponentInstance *ci =
    dynamic_cast<TaoComponentInstance*>(svc.getPointer());
  if (ci == 0) {
    return false;
  }
  framework->unregisterComponent(ci->getInstanceName());
  ci->deleteReference();
  return true;
}

bool TaoComponentModel::haveComponent(const std::string& type)
{
  SCIRun::Guard g1(&lock_components);
#if DEBUG
  std::cerr << "Tao looking for component of type: " << type << std::endl;
#endif
  return components.find(type) != components.end();
}


ComponentInstance*
TaoComponentModel::createInstance(const std::string& name,
                                  const std::string& type,
                                  const sci::cca::TypeMap::pointer &tm)
{
  tao::Component *component;
  lock_components.lock();
  componentDB_type::iterator iter = components.find(type);
  if (iter == components.end()) { // could not find this component
    std::cerr << "Error: could not locate any Tao components." << std::endl;
    return 0;
  }
  lock_components.unlock();

  void* maker_v = getMakerAddress(type, *(iter->second));
  if (! maker_v) {
    // should probably throw exception here
    std::cerr << "TaoComponentModel::createInstance failed." << std::endl;
    return 0;
  }
  tao::Component* (*maker)() = (tao::Component* (*)())(maker_v);
  //  std::cerr << "about to create Tao component" << std::endl;
  component = (*maker)();

  TaoComponentInstance* ci =
    new TaoComponentInstance(framework, name, type, tm, component);
  ci->addReference();
  component->setServices(sci::cca::TaoServices::pointer(ci));

  return ci;
}

bool TaoComponentModel::destroyInstance(ComponentInstance *ci)
{
  TaoComponentInstance* cca_ci = dynamic_cast<TaoComponentInstance*>(ci);
  if (!cca_ci) {
    std::cerr << "error: in destroyInstance() cca_ci is 0" << std::endl;
    return false;
  }
  cca_ci->deleteReference();
  return true;
}

void
TaoComponentModel::listAllComponentTypes(
                                         std::vector<ComponentDescription*>& list, bool /*listInternal*/)
{
  SCIRun::Guard g1(&lock_components);
  for (componentDB_type::iterator iter=components.begin();
       iter != components.end(); iter++) {
    list.push_back(iter->second);
  }
}

} // end namespace SCIRun
