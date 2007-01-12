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
 *  BabelComponentModel.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */

#include <Framework/Babel/framework.hxx>
#include <sidl.hxx>

#include <Framework/Babel/BabelComponentModel.h>
#include <Framework/Babel/BabelComponentDescription.h>
#include <Framework/Babel/BabelComponentInstance.h>

#include <Framework/SCIRunFramework.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/soloader.h>
#include <Core/Util/NotFinished.h>
#include <Core/CCA/PIDL/PIDL.h>

extern "C" {
#  include <string.h>
#  include <stdlib.h>
}

#include <iostream>

namespace SCIRun {

const std::string BabelComponentModel::DEFAULT_XML_PATH("/CCA/Components/BabelTest/xml");


BabelComponentModel::BabelComponentModel(SCIRunFramework* framework,
                                         const StringVector& xmlPaths)
  : ComponentModel("babel", framework),
    lock_components("BabelComponentModel::components lock")
{
  buildComponentList(xmlPaths);
}

BabelComponentModel::~BabelComponentModel()
{
  destroyComponentList();
}

void BabelComponentModel::destroyComponentList()
{
  SCIRun::Guard g1(&lock_components);
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++) {
    delete iter->second;
  }
  components.clear();
}

void BabelComponentModel::buildComponentList(const StringVector& files)
{
  destroyComponentList();

  // update SIDL Loader
  std::string searchPath(::sidl::Loader::getSearchPath());
  bool emptySearchPath = searchPath.empty();
  if (files.empty()) {
    StringVector xmlPaths_;
    getXMLPaths(framework, xmlPaths_);

    for (StringVector::iterator iter = xmlPaths_.begin(); iter != xmlPaths_.end(); iter++) {
      if (parseComponentModelXML(*iter, this)) {
        std::string path(pathname(*iter));
        if (emptySearchPath || searchPath.find(path) == std::string::npos) {
          // This is the ';' separated path that sidl::Loader will search for
          // *.scl files. Babel *.scl files are necessary for mapping component
          // names with their DLLs.
          ::sidl::Loader::addSearchPath(path);
        }
      }
    }
  } else {
    for (StringVector::const_iterator iter = files.begin(); iter != files.end(); iter++) {
      if (parseComponentModelXML(*iter, this)) {
        std::string path(pathname(*iter));
        if (emptySearchPath || searchPath.find(path) == std::string::npos) {
          // This is the ';' separated path that sidl::Loader will search for
          // *.scl files. Babel *.scl files are necessary for mapping component
          // names with their DLLs.
          ::sidl::Loader::addSearchPath(path);
        }
      }
    }
  }
}

void
BabelComponentModel::setComponentDescription(const std::string& type, const std::string& library)
{
  BabelComponentDescription* cd = new BabelComponentDescription(this, type, library);
  Guard g(&lock_components);
  componentDB_type::iterator iter = components.find(cd->getType());
  if (iter != components.end()) {
    std::cerr << "WARNING: Multiple definitions exist for " << cd->getType() << std::endl;
  } else {
    components[cd->getType()] = cd;
  }
}

::gov::cca::Services
BabelComponentModel::createServices(const std::string& instanceName,
                                    const std::string& className,
                                    const ::gov::cca::TypeMap& properties)
{
  /*
    ::gov::cca::Component nullCom;
    ::gov::cca::Services svc;
    cerr<<"need associate svc with ci in createServices!"<<endl;
    BabelComponentInstance* ci = new BabelComponentInstance(framework,
    instanceName, className,
    properties,
    nullCom,
    svc);
    framework->registerComponent(ci, instanceName);

    //ci->addReference();

  */
  // is this supposed to be called by AbstractFramework.getServices???
  NOT_FINISHED("gov::cca::Services BabelComponentModel::createServices(const std::string& instanceName, const std::string& className, const ::gov::cca::TypeMap& properties)");
  ::gov::cca::Services svc;
  return svc;
}

bool BabelComponentModel::haveComponent(const std::string& type)
{
  SCIRun::Guard g1(&lock_components);
  return components.find(type) != components.end();
}

ComponentInstance* BabelComponentModel::createInstance(const std::string& name,
                                                       const std::string& type,
                                                       const sci::cca::TypeMap::pointer&)
{
#if DEBUG
  std::cerr << "BabelComponentModel::createInstance: attempt to create "
            << name << " type " << type << std::endl;
#endif
  Guard g1(&lock_components);
  componentDB_type::iterator iter = components.find(type);

  if (iter == components.end()) {
    std::cerr << "ERROR: Component " << type << " is not a registered component."
              << std::endl;
    return 0;
  }

#if 0
  /*
   *  std::string lastname=type.substr(type.find('.')+1);
   *  std::string so_name("lib/libBabel_Components_");
   *  so_name=so_name+lastname+".so";
   *  cerr<<"type="<<type<<" soname="<<so_name<<std::endl;
   *
   *  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
   *  if(!handle){
   *    cerr << "Cannot load component .so " << type << '\n';
   *    cerr << SOError() << '\n';
   *    return 0;
   *  }
   *
   *  std::string makername = "make_"+type;
   *  for(int i=0;i<(int)makername.size();i++)
   *    if(makername[i] == '.')
   *  makername[i]='_';
   *
   *  cerr<<"looking for symbol:"<< makername<<std::endl;
   *  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
   *  if(!maker_v){
   *    cerr <<"Cannot load component symbol " << type << '\n';
   *    cerr << SOError() << '\n';
   *    return 0;
   *  }
   *  gov::cca::Component (*maker)() = (gov::cca::Component (*)())(maker_v);
   *  cerr<<"about to create babel component"<<std::endl;
   *  component = (*maker)();
   */
#endif

  /*
   * sidl.Loader.findLibrary params:
   *  server-side binding: use "ior/impl" to find class implementation
   *  client-side binding: use language name
   *  Scope and Resolve from SCIRun/src/CCA/Components/BabelTest/xml/BabelTest.scl
   *
   * Note for *nix: make sure library path is in LD_LIBRARY_PATH
   */
  ::sidl::DLL library = ::sidl::Loader::findLibrary(type, "ior/impl", ::sidl::Scope_SCLSCOPE, ::sidl::Resolve_SCLRESOLVE);
#if DEBUG
  std::cerr << "sidl::Loader::getSearchPath=" << ::sidl::Loader::getSearchPath() << std::endl;
  // get default finder and report search path
  ::sidl::Finder f = ::sidl::Loader::getFinder();
  std::cerr << "sidl::Finder::getSearchPath=" << f.getSearchPath() << std::endl;
#endif
  if (library._is_nil()) {
    std::cerr << "Could not find library for type " << type << ". "
              << "Check your environment settings as described in the Babel and SCIRun2 usage instructions."
              << std::endl;
    return 0;
  }

  ::sidl::BaseClass sidl_class = library.createClass(type);
  // cast BaseClass instance to Component
  // babel_cast<>() introduced in UC++ bindings, returns nil pointer on bad cast
  ::gov::cca::Component component = ::sidl::babel_cast< ::gov::cca::Component>(sidl_class);
  if ( component._is_nil() ) {
    std::cerr << "Cannot load babel component of type " << type
              << ". Babel component not created." << std::endl;
    return 0;
  }
  ::framework::Services svc = ::framework::Services::_create();
  component.setServices(svc);
  ::gov::cca::TypeMap nullMap;

  BabelComponentInstance* ci =
    new BabelComponentInstance(framework, name, type, nullMap, component, svc);
  //ci->addReference();
  return ci;
}

bool BabelComponentModel::destroyInstance(ComponentInstance *ci)
{
  NOT_FINISHED("bool BabelComponentModel::destroyInstance(ComponentInstance *ci)");
  // make sure why ci->addReference() is called in createInstance(); -- removed (AK)

  // not sure if deleteReference() is appropriate here
  // TODO: need to support release component callback
  delete ci;
  return false;
}

void BabelComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list,
                                                bool /*listInternal*/)
{
  SCIRun::Guard g1(&lock_components);
  for (componentDB_type::iterator iter=components.begin();
       iter != components.end(); iter++) {
    list.push_back(iter->second);
  }
}

// TODO: This returns an empty string only - should this return a URI???
std::string BabelComponentModel::createComponent(const std::string& name, const std::string& type)
{
#if DEBUG
  std::cerr << "BabelComponentModel::createComponent: attempt to create " << name << " type " << type << std::endl;
#endif
  sci::cca::Component::pointer component;
  lock_components.lock();
  componentDB_type::iterator iter = components.find(type);
  lock_components.unlock();
  if (iter == components.end()) {
    return std::string();
  }

  std::string lastname=type.substr(type.find('.')+1);
  std::string so_name("lib/libBabel_Components_");
  so_name = so_name + lastname + ".so";
#if DEBUG
  std::cerr << "type=" << type << " soname=" << so_name << std::endl;
#endif
  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
  if (!handle) {
    std::cerr << "Cannot load component " << type << std::endl;
    std::cerr << SOError() << std::endl;
    return std::string();
  }

  std::string makername = "make_" + type;
  for (int i = 0; i < (int)makername.size(); i++) {
    if (makername[i] == '.') {
      makername[i] = '_';
    }
  }

  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if (!maker_v) {
    std::cerr << "Cannot load component " << type << std::endl;
    std::cerr << SOError() << std::endl;
    return std::string();
  }
#if 0
  /*  sci::cca::Component::pointer (*maker)() = (sci::cca::Component::pointer (*)())(maker_v);
      component = (*maker)();
      //need to make sure addReference() will not cause problem
      component->addReference();
      return component->getURL().getString();
  */
#endif
  return std::string();
}

} // end namespace SCIRun
