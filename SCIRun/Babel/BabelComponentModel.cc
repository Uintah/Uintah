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

#include <SCIRun/Babel/BabelComponentModel.h>
#include <SCIRun/Babel/BabelComponentDescription.h>
#include <SCIRun/Babel/BabelComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/SCIRunErrorHandler.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Core/XMLUtil/StrX.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <string>
#include "framework.hh"
#include "sidl.hh"

extern "C" {
#include <string.h>
#include <stdlib.h>
}

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

#include <iostream>

namespace SCIRun {

const std::string BabelComponentModel::DEFAULT_PATH =
   std::string("/CCA/Components/BabelTest/xml");


BabelComponentModel::BabelComponentModel(SCIRunFramework* framework)
  : ComponentModel("babel"), framework(framework),
    lock_components("BabelComponentModel::components lock")
{
  buildComponentList();
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

void BabelComponentModel::buildComponentList()
{
  // Initialize the XML4C system
  try {
    XMLPlatformUtils::Initialize();
  }
  catch (const XMLException& toCatch) {
    std::cerr << "Error during initialization! :" << std::endl
              << StrX(toCatch.getMessage()) << std::endl;
    return;
  }

  destroyComponentList();

  std::string search_path = sidl::Loader::getSearchPath();
  SSIDL::array1<std::string> sArray;
  sci::cca::TypeMap::pointer tm;
  sci::cca::ports::FrameworkProperties::pointer fwkProperties =
    pidl_cast<sci::cca::ports::FrameworkProperties::pointer>(
        framework->getFrameworkService("cca.FrameworkProperties", ""));
  if (fwkProperties.isNull()) {
    std::cerr << "Error: Cannot find framework properties" << std::cerr;
    //return sci_getenv("SCIRUN_SRCDIR") + DEFAULT_PATH;
  } else {
    tm = fwkProperties->getProperties();
    sArray = tm->getStringArray("sidl_xml_path", sArray);
    framework->releaseFrameworkService("cca.FrameworkProperties", "");
  }

  for (SSIDL::array1<std::string>::iterator it = sArray.begin(); it != sArray.end(); it++) {
    if (search_path.find(*it) == std::string::npos) {
       // This is the ';' separated path that sidl::Loader will search for
       // *.scl files. Babel *.scl files are necessary for mapping component
       // names with their DLLs.
       sidl::Loader::addSearchPath(*it);
    }

    Dir d(*it);
    std::cerr << "Babel Component Model: Looking at directory: " << *it << std::endl;
    std::vector<std::string> files;
    d.getFilenamesBySuffix(".xml", files);
    for(std::vector<std::string>::iterator iter = files.begin();
        iter != files.end(); iter++) {
      std::string& file = *iter;
      std::cerr << "Babel Component Model: Looking at file" << file << std::endl;
      readComponentDescription(*it+"/"+file);
    }
  }
}
void BabelComponentModel::readComponentDescription(const std::string& file)
{
  // Instantiate the DOM parser.
  SCIRunErrorHandler handler;
  XercesDOMParser parser;
  parser.setDoValidation(true);
  parser.setErrorHandler(&handler);

  try {
    std::cout << "Parsing file: " << file << std::endl;
    parser.parse(file.c_str());
  }
  catch (const XMLException& toCatch) {
    std::cerr << "Error during parsing: '" <<
      file << "' " << std::endl << "Exception message is:  " <<
      xmlto_string(toCatch.getMessage()) << std::endl;
    handler.foundError = true;
    return;
  }
  catch ( ... ) {
    std::cerr << "Unknown error occurred during parsing: '" << file << "' "
              << std::endl;
    handler.foundError = true;
    return;
  }
  // Get all the top-level document node
  DOMDocument* document = parser.getDocument();
  // Check that this document is actually describing CCA components
  DOMElement *metacomponentmodel = static_cast<DOMElement *>(
        document->getElementsByTagName(to_xml_ch_ptr("metacomponentmodel"))->item(0));
  std::string compModelName
    = to_char_ptr(metacomponentmodel->getAttribute(to_xml_ch_ptr("name")));
  //std::cout << "Component model name = " << compModelName << std::endl;
  if ( compModelName != std::string(this->prefixName) ) {
    return;
  }
  // Get a list of the library nodes.  Traverse the list and read component
  // elements at each list node.
  DOMNodeList* libraries
    = document->getElementsByTagName(to_xml_ch_ptr("library"));
  for (unsigned int i = 0; i < libraries->getLength(); i++) {
    DOMElement *library = static_cast<DOMElement *>(libraries->item(i));
    // Read the library name
    std::string library_name(to_char_ptr(library->getAttribute(to_xml_ch_ptr("name"))));
    std::cout << "Library name = ->" << library_name << "<-" << std::endl;
    // Get the list of components.
    DOMNodeList* comps
      = library->getElementsByTagName(to_xml_ch_ptr("component"));
    for (unsigned int j = 0; j < comps->getLength(); j++) {
      // Read the component name
      DOMElement *component = static_cast<DOMElement *>(comps->item(j));
      std::string
        component_name(to_char_ptr(component->getAttribute(to_xml_ch_ptr("name"))));
      //std::cout << "Component name = ->" << component_name << "<-" << std::endl;
      // Register this component
      BabelComponentDescription* cd = new BabelComponentDescription(this);
      cd->type = component_name;
      //cd->setLibrary(library_name.c_str()); // record the DLL name
      lock_components.lock();
      this->components[cd->type] = cd;
      lock_components.unlock();
    }
  }
}
gov::cca::Services
BabelComponentModel::createServices(const std::string& instanceName,
                                    const std::string& className,
                                    const gov::cca::TypeMap& properties)
{
  /*
  gov::cca::Component nullCom;
  gov::cca::Services svc;
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
  gov::cca::Services svc;
  std::cerr << "BabelComponentModel::createServices() is not implemented !"
            << std::endl;
  return svc;
}

bool BabelComponentModel::haveComponent(const std::string& type)
{
  SCIRun::Guard g1(&lock_components);
  std::cerr << "CCA(Babel) looking for babel component of type: " << type
            << std::endl;
  return components.find(type) != components.end();
}

ComponentInstance* BabelComponentModel::createInstance(const std::string &name, const std::string &type)
{
std::cerr << "BabelComponentModel::createInstance: attempt to create " << name << " type " << type << std::endl;
  gov::cca::Component component;
  if (true) { //local component 
   
    lock_components.lock();
    componentDB_type::iterator iter = components.find(type);
    
    if (iter == components.end()) {
      std::cerr << "ERROR: Component " << type << " is not a registered component."
                << std::endl;
      return 0;
    }
    lock_components.unlock();

#if 0
    /*
    std::string lastname=type.substr(type.find('.')+1);  
    std::string so_name("lib/libBabel_Components_");
    so_name=so_name+lastname+".so";
    cerr<<"type="<<type<<" soname="<<so_name<<std::endl;

    LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
    if(!handle){
      cerr << "Cannot load component .so " << type << '\n';
      cerr << SOError() << '\n';
      return 0;
    }

    std::string makername = "make_"+type;
    for(int i=0;i<(int)makername.size();i++)
      if(makername[i] == '.')
    makername[i]='_';
    
    cerr<<"looking for symbol:"<< makername<<std::endl;
    void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
    if(!maker_v){
      cerr <<"Cannot load component symbol " << type << '\n';
      cerr << SOError() << '\n';
      return 0;
    }
    gov::cca::Component (*maker)() = (gov::cca::Component (*)())(maker_v);
    cerr<<"about to create babel component"<<std::endl;
    component = (*maker)();
    */
#endif

    /*
     * sidl.Loader.findLibrary params:
     *  server-side binding: use "ior/impl" to find class implementation
     *  client-side binding: use language name
     *  Scope and Resolve from SCIRun/src/CCA/Components/BabelTest/xml/BabelTest.scl
     */
    sidl::DLL library = sidl::Loader::findLibrary(type, "ior/impl",
                  sidl::Scope_SCLSCOPE, sidl::Resolve_SCLRESOLVE);
    std::cout << "sidl::Loader::getSearchPath=" << sidl::Loader::getSearchPath() << std::endl;
    if (library._not_nil()) {
      std::cerr << "Found library for class " << type << std::endl;
    } else {
      std::cerr << "Could not find library for type " << type
                << std::endl;
      return 0;
    }

    sidl::BaseClass sidl_class = library.createClass(type);

    // end upgrade

    // jc--why is this assignment necessary??
    component = sidl_class;
    if ( component._not_nil() ) { 
      //std::cerr << "babel component of type " << type << " is loaded!"
      //          << std::endl;
    } else {
        std::cerr << "Cannot load babel component of type " << type
                  << ". Babel component not created." << std::endl;
      return 0;
    }
  } else { //remote component: need to be created by framework at url 
    std::cerr << "remote babel components creation is not done!" << std::endl;
    /*
    Object::pointer obj=PIDL::objectFrom(url);
    if(obj.isNull()){
      cerr<<"got null obj (framework) from "<<url<<std::endl;
      return 0;
    }

    sci::cca::AbstractFramework::pointer remoteFramework=
      pidl_cast<sci::cca::AbstractFramework::pointer>(obj);

    std::string comURL=remoteFramework->createComponent(name, type);
    //cerr<<"comURL="<<comURL<<std::endl;
    Object::pointer comObj=PIDL::objectFrom(comURL);
    if(comObj.isNull()){
      cerr<<"got null obj(Component) from "<<url<<std::endl;
      return 0;
    }
    component=pidl_cast<sci::cca::Component::pointer>(comObj);
    */
  }

  std::cerr << "about to create services" << std::endl;
  framework::Services svc = framework::Services::_create();
  std::cerr << "services created !" << std::endl;
  component.setServices(svc);
  std::cerr << "component.setService done!" << std::endl;
  gov::cca::Component nullMap;

  BabelComponentInstance* ci =
    new BabelComponentInstance(framework, name, type, nullMap, component, svc);
  std::cerr<<"comopnent instance ci is created!"<<std::endl;
  //ci->addReference();
  return ci;
}

bool BabelComponentModel::destroyInstance(ComponentInstance *ci)
{
  std::cerr<<"BabelComponentModel::destroyInstance() is not done"<<std::endl;
  //make sure why ci->addReference() is called in createInstace();
  delete ci;  
  return false;
}

std::string BabelComponentModel::getName() const
{
  return "babel";
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



std::string BabelComponentModel::createComponent(const std::string& name, const std::string& type)
{
std::cerr << "BabelComponentModel::createComponent: attempt to create " << name << " type " << type << std::endl;

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
  //cerr << "type=" <<type<<" soname=" <<so_name<< std::endl;

  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
  if (!handle) {
    std::cerr << "Cannot load component " << type << std::endl;
    std::cerr << SOError() << std::endl;
    return std::string();
  }

  std::string makername = "make_" + type;
  for (int i = 0;i < (int)makername.size(); i++) {
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
