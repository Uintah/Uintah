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

#include <SCIRun/Tao/Component.h>
#include <SCIRun/Tao/TaoComponentModel.h>
#include <SCIRun/Tao/TaoComponentDescription.h>
#include <SCIRun/Tao/TaoComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/SCIRunErrorHandler.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <SCIRun/resourceReference.h>
#include <string>

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

const std::string TaoComponentModel::DEFAULT_PATH =
    std::string("/CCA/Components/TAO/xml");


TaoComponentModel::TaoComponentModel(SCIRunFramework* framework)
  : ComponentModel("tao"), framework(framework)
{
  // Record the path containing DLLs for components.
  const char *dll_path = getenv("SIDL_DLL_PATH");
  if (dll_path != 0) {
    this->setSidlDLLPath(std::string(dll_path));
  } else {
    this->setSidlDLLPath(sci_getenv("SCIRUN_OBJDIR") + std::string("/lib"));
  }

  buildComponentList();
}

TaoComponentModel::~TaoComponentModel()
{
  destroyComponentList();
}

void TaoComponentModel::destroyComponentList()
{
  for (componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++) {
    delete iter->second;
  }
  components.clear();
}

void TaoComponentModel::buildComponentList()
{
  // Initialize the XML4C system
  try {
      XMLPlatformUtils::Initialize();
  }
  catch (const XMLException& toCatch) {
      std::cerr << "Error during initialization! :" <<
          std::endl << StrX(toCatch.getMessage()) << std::endl;
      return;
  }

  destroyComponentList();

  SSIDL::array1<std::string> sArray;
  sci::cca::TypeMap::pointer tm;
  sci::cca::ports::FrameworkProperties::pointer fwkProperties =
   pidl_cast<sci::cca::ports::FrameworkProperties::pointer>(
       framework->getFrameworkService("cca.FrameworkProperties", "")
   );
  if (fwkProperties.isNull()) {
      std::cerr << "Error: Cannot find framework properties" << std::cerr;
      //return sci_getenv("SCIRUN_SRCDIR") + DEFAULT_PATH;
  } else {
      tm = fwkProperties->getProperties();
      sArray = tm->getStringArray("sidl_xml_path", sArray);
  }
  framework->releaseFrameworkService("cca.FrameworkProperties", "");

  for (SSIDL::array1<std::string>::iterator it = sArray.begin(); it != sArray.end(); it++) {
    Dir d(*it);
    std::vector<std::string> files;
    d.getFilenamesBySuffix(".xml", files);
    for (std::vector<std::string>::iterator iter = files.begin();
      iter != files.end();
      iter++) {
      std::string& file = *iter;
      readComponentDescription(*it+"/"+file);
    }
  }
}

void TaoComponentModel::readComponentDescription(const std::string& file)
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
    handler.foundError=true;
    return;
  }
  catch ( ... ) {
    std::cerr << "Unknown error occurred during parsing: '" << file << "' "
              << std::endl;
    handler.foundError=true;
    return;
  }

  // Get all the top-level document node
  DOMDocument* document = parser.getDocument();

  // Check that this document is actually describing TAO components
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
      TaoComponentDescription* cd = new TaoComponentDescription(this, component_name);
      cd->setLibrary(library_name.c_str()); // record the DLL name
      this->components[cd->type] = cd;
    }
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
  std::cerr << "Tao looking for component of type: " << type << std::endl;
  return components.find(type) != components.end();
}


ComponentInstance*
TaoComponentModel::createInstance(const std::string& name,
                                  const std::string& type,
                                  const sci::cca::TypeMap::pointer &tm)
{
  tao::Component *component;
  componentDB_type::iterator iter = components.find(type);
  if (iter == components.end()) { // could not find this component
    return 0;
  }
  // Get the list of DLL paths to search for the appropriate component library
  std::vector<std::string> possible_paths = splitPathString(this->getSidlDLLPath());
  LIBRARY_HANDLE handle;

  for (std::vector<std::string>::iterator it = possible_paths.begin();
       it != possible_paths.end(); it++) {
    std::string so_name = *it + "/" + iter->second->getLibrary();
    handle = GetLibraryHandle(so_name.c_str());
    if (handle)  {  break; }
  }

  if ( !handle ) {
    std::cerr << "Could not find component DLL: " << iter->second->getLibrary()
              << " for type " << type << std::endl;
    std::cerr << SOError() << std::endl;
    return 0;
  }

  std::string makername = "make_"+type;
  for (int i = 0; i < static_cast<int>(makername.size()); i++) {
    if (makername[i] == '.') { makername[i]='_'; }
  }

  //  std::cerr << "looking for symbol:" << makername << std::endl;
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if (!maker_v) {
    //    std::cerr <<"Cannot load component symbol " << type << std::endl;
    std::cerr << SOError() << std::endl;
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

std::string TaoComponentModel::getName() const
{
  return "Tao";
}

void
TaoComponentModel::listAllComponentTypes(
    std::vector<ComponentDescription*>& list, bool /*listInternal*/)
{
  for (componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++) {
    list.push_back(iter->second);
  }
}

int TaoComponentModel::addLoader(resourceReference *rr)
{
  loaderList.push_back(rr);
  std::cerr<<"Loader "<<rr->getName()<<" is added into cca component model"<<std::endl;
  return 0;
}

int TaoComponentModel::removeLoader(const std::string &loaderName)
{
  resourceReference *rr=getLoader(loaderName);
  if (rr!=0) {
    std::cerr<<"loader "<<rr->getName()<<" is removed from cca component model\n";
    delete rr;
  } else {
    std::cerr<<"loader "<<loaderName<<" not found in cca component model\n";
  }
  return 0;
}

resourceReference *
TaoComponentModel::getLoader(std::string loaderName)
{
  resourceReference *rr=0;
  for (unsigned int i=0; i<loaderList.size(); i++) {
    if (loaderList[i]->getName()==loaderName) {
      rr=loaderList[i];
      break;
    }
  }
  return rr;
}

} // end namespace SCIRun
