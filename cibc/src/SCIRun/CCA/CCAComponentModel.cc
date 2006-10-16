/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  CCAComponentModel.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/CCA/CCAComponentModel.h>
#include <SCIRun/CCA/CCAComponentDescription.h>
#include <SCIRun/CCA/CCAComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/resourceReference.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/Thread/Guard.h>


#ifndef DEBUG
#  define DEBUG 0
#endif

namespace SCIRun {

const std::string CCAComponentModel::DEFAULT_PATH("/CCA/Components");
const std::string CCAComponentModel::DEFAULT_XML_PATH("/CCA/Components/xml");


CCAComponentModel::CCAComponentModel(SCIRunFramework* framework,
				     const StringVector& xmlPaths)
  : ComponentModel("cca", framework),
    lock_components("CCAComponentModel::components lock"),
    lock_loaderList("CCAComponentModel::loaderList lock")
{
  // move to framework properties
  // Record the path containing DLLs for components.
  const char *dll_path = getenv("SIDL_DLL_PATH");
  if (dll_path != 0) {
    this->setSidlDLLPath(std::string(dll_path));
  } else {
    this->setSidlDLLPath(sci_getenv("SCIRUN_OBJDIR") + std::string("/lib"));
  }

  buildComponentList(xmlPaths);
}

CCAComponentModel::~CCAComponentModel()
{
  destroyComponentList();
}

void CCAComponentModel::destroyComponentList()
{
  SCIRun::Guard g1(&lock_components);
  for (componentDB_type::iterator iter = components.begin();
      iter != components.end(); iter++) {
    delete iter->second;
  }
  components.clear();
}

void CCAComponentModel::buildComponentList(const StringVector& files)
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
CCAComponentModel::setComponentDescription(const std::string& type, const std::string& library)
{
  // check for duplicates
  CCAComponentDescription* cd = new CCAComponentDescription(this, type, library);
  Guard g(&lock_components);
  componentDB_type::iterator iter = components.find(cd->getType());
  if (iter != components.end()) {
    std::cerr << "WARNING: Multiple definitions exist for " << cd->getType() << std::endl;
  } else {
    components[cd->getType()] = cd;
  }
}

sci::cca::Services::pointer
CCAComponentModel::createServices(const std::string& instanceName,
		  const std::string& className,
		  const sci::cca::TypeMap::pointer& properties)
{
#if DEBUG
  std::cout << "CCAComponentModel::createServices: " << instanceName << ", " << className << std::endl;
#endif
// get unique name?
  CCAComponentInstance* ci = new CCAComponentInstance(framework, instanceName, className, properties, sci::cca::Component::pointer(0));
  framework->registerComponent(ci, instanceName);
  ci->addReference();
  return sci::cca::Services::pointer(ci);
}

bool CCAComponentModel::destroyServices(const sci::cca::Services::pointer& svc)
{
    CCAComponentInstance *ci =
    dynamic_cast<CCAComponentInstance*>(svc.getPointer());
    if (ci == 0) {
	return false;
    }
    framework->unregisterComponent(ci->getInstanceName());
    ci->deleteReference();
    return true;
}

bool CCAComponentModel::haveComponent(const std::string& type)
{
  SCIRun::Guard g1(&lock_components);
#if DEBUG
  std::cout << "CCA looking for component of type: " << type << std::endl;
#endif
  return components.find(type) != components.end();
}


ComponentInstance*
CCAComponentModel::createInstance(const std::string& name,
				  const std::string& type,
				  const sci::cca::TypeMap::pointer& properties)

{
  std::string loaderName;
  if (! properties.isNull()) {
    // TODO: possible memory leak?
    properties->addReference();
    loaderName = properties->getString("LOADER NAME", "");
  }
#if DEBUG
  std::cout <<"creating cca component <" << name << "," << type << "> with loader:"
	    << loaderName << std::endl;
#endif
  sci::cca::Component::pointer component;
  if (loaderName.empty()) {  //local component
     Guard g1(&lock_components);
    componentDB_type::iterator iter = components.find(type);
    if (iter == components.end()) {
      std::cerr << "Error: could not locate any cca components.  Make sure paths set in environment variables \"SIDL_XML_PATH\" and \"SIDL_DLL_PATH\" are correct." << std::endl;
      return 0;
    }

    // Get the list of DLL paths to search for the appropriate component library
    StringVector possible_paths =
	splitPathString(this->getSidlDLLPath());
    LIBRARY_HANDLE handle;

    for (StringVector::iterator it = possible_paths.begin();
	 it != possible_paths.end(); it++) {
      std::string so_name = *it + "/" + iter->second->getLibrary();
      handle = GetLibraryHandle(so_name.c_str());
      if (handle) { break; }
    }

    if (!handle) {
      std::cerr << "Cannot load component " << type << std::endl;
      std::cerr << SOError() << std::endl;
      return 0;
    }

    std::string makername = "make_"+type;
    for (int i = 0; i < (int)makername.size(); i++) {
	if (makername[i] == '.') {
	    makername[i] = '_';
	}
    }
    void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
    if (!maker_v) {
      std::cerr <<"Cannot load component " << type << std::endl;
      std::cerr << SOError() << std::endl;
      return 0;
    }
    sci::cca::Component::pointer (*maker)() =
	(sci::cca::Component::pointer (*)())(maker_v);
    component = (*maker)();
 } else {
    //use loader to load the component
    resourceReference* loader = getLoader(loaderName);
    std::vector<int> nodes;
    nodes.push_back(0);
    Object::pointer comObj = loader->createInstance(name, type, nodes);
    component = pidl_cast<sci::cca::Component::pointer>(comObj);
    properties->putInt("np",loader->getSize() );
  }
  CCAComponentInstance* ci =
	new CCAComponentInstance(framework, name, type, properties, component);
  component->setServices(sci::cca::Services::pointer(ci));
  return ci;
}

bool CCAComponentModel::destroyInstance(ComponentInstance *ci)
{
  CCAComponentInstance* cca_ci = dynamic_cast<CCAComponentInstance*>(ci);
  if (!cca_ci) {
    std::cerr << "error: in destroyInstance() cca_ci is 0" << std::endl;
    return false;
  }
  cca_ci->releaseComponentCallback(sci::cca::Services::pointer(cca_ci));
  // addReference in CCAComponentModel::createInstance(..)
  cca_ci->getComponentProperties()->deleteReference();

  // Since a CCAComponentInstance is also a Services object,
  // it is also implicitly a PIDL Object.
  // Object::deleteReference will delete itself if it's reference
  // count is 0.
  //
  // TODO: using 'delete' here tends to be unstable (needs investigating,
  // possibly related to Bugzilla bug #2294).

  // might be the source of object deletion exceptions - remove for now
  //cca_ci->deleteReference();
  return true;
}

const std::string CCAComponentModel::getName() const
{
  return "CCA";
}

void CCAComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list, bool /*listInternal*/)
{
  lock_components.lock();
  for (componentDB_type::iterator iter = components.begin();
       iter != components.end(); iter++) {
    list.push_back(iter->second);
  }
  lock_components.unlock();


  Guard g2(&lock_loaderList);
  for (unsigned int i = 0; i < loaderList.size(); i++) {
    SSIDL::array1<std::string> typeList;
    loaderList[i]->listAllComponentTypes(typeList);
    //convert typeList to component description list
    //by attaching a loader (resourceReference) to it.
    for (unsigned int j = 0; j < typeList.size(); j++) {
      // library name needed?
      CCAComponentDescription* cd = new CCAComponentDescription(this, typeList[j]);
      cd->setLoaderName(loaderList[i]->getName());
      list.push_back(cd);
    }
  }
}

int CCAComponentModel::addLoader(resourceReference *rr)
{
  lock_loaderList.lock();
  loaderList.push_back(rr);
  lock_loaderList.unlock();
#if DEBUG
  std::cerr << "Loader " << rr->getName()
	    << " is added into cca component model" << std::endl;
#endif
  return 0;
}

int CCAComponentModel::removeLoader(const std::string &loaderName)
{
  resourceReference *rr = getLoader(loaderName);
  if (rr != 0) {
#if DEBUG
    std::cerr << "loader " << rr->getName() << " is removed from cca component model\n";
#endif
    delete rr;
  } else {
#if DEBUG
    std::cerr << "loader " << loaderName << " not found in cca component model\n";
#endif
  }
  return 0;
}

resourceReference *
CCAComponentModel::getLoader(std::string loaderName)
{
  resourceReference *rr = 0;
  lock_loaderList.lock();
  for (unsigned int i = 0; i < loaderList.size(); i++) {
    if (loaderList[i]->getName() == loaderName) {
      rr = loaderList[i];
      break;
    }
  }
  lock_loaderList.unlock();
  return rr;
}

} // end namespace SCIRun
