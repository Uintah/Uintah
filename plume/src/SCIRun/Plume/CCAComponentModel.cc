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
 *  CCAComponentModel.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Plume/CCAComponentModel.h>
#include <SCIRun/Distributed/DistributedFramework.h>

#include <SCIRun/Distributed/ComponentDescription.h>
#include <SCIRun/Distributed/ComponentInfo.h>
//#include <SCIRun/SCIRunErrorHandler.h>

#include <Core/Thread/Guard.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Core/XMLUtil/StrX.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/CCA/PIDL/PIDL.h>
//#include <SCIRun/resourceReference.h>
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

#ifndef DEBUG
  #define DEBUG 0
#endif

namespace SCIRun {
  
  const std::string CCAComponentModel::DEFAULT_PATH = std::string("/CCA/Components/xml");
  
  
  CCAComponentModel::CCAComponentModel(DistributedFramework* framework)
    : framework(framework),
      lock_components("CCAComponentModel::components lock")
  {
    // move to framework properties
    // Record the path containing DLLs for components.
    const char *dll_path = getenv("SIDL_DLL_PATH");
    if (dll_path != 0) {
      this->setSidlDLLPath(std::string(dll_path));
    } else {
      this->setSidlDLLPath(sci_getenv("SCIRUN_OBJDIR") + std::string("/lib"));
    }
    
    buildComponentList();
  }
  
  CCAComponentModel::~CCAComponentModel()
  {
    destroyComponentList();
  }
  
  void CCAComponentModel::destroyComponentList()
  {
    SCIRun::Guard g1(&lock_components);
    for(componentDB_type::iterator iter=components.begin();
	iter != components.end(); iter++) {
      delete iter->second;
    }
    components.clear();
  }
  
  void CCAComponentModel::buildComponentList()
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
    
    SSIDL::array1<std::string> sArray;
    sci::cca::TypeMap::pointer tm;
    sci::cca::ports::FrameworkProperties::pointer fwkProperties =
      pidl_cast<sci::cca::ports::FrameworkProperties::pointer>
      (framework->getFrameworkService("cca.FrameworkProperties", ""));

    if (fwkProperties.isNull()) {
      std::cerr << "Error: Cannot find framework properties" ;
      //return sci_getenv("SCIRUN_SRCDIR") + DEFAULT_PATH;
    } else {
      tm = fwkProperties->getProperties();
      sArray = tm->getStringArray("sidl_xml_path", sArray);
    }
    framework->releaseFrameworkService("cca.FrameworkProperties", "");
  
    for (SSIDL::array1<std::string>::iterator it = sArray.begin(); it != sArray.end(); it++) {
      Dir d(*it);
      //std::cout << "CCA Component Model: Looking at directory: " << *it << std::endl;
      std::vector<std::string> files;
      d.getFilenamesBySuffix(".xml", files);
      
      for(std::vector<std::string>::iterator iter = files.begin();
	  iter != files.end(); iter++) {
	std::string& file = *iter;
	//std::cout << "CCA Component Model: Looking at file" << file << std::endl;
	readComponentDescription(*it+"/"+file);
      }
    }
  }
  
  void CCAComponentModel::readComponentDescription(const std::string& file)
  {
    // Instantiate the DOM parser.
    SCIRunErrorHandler handler;
    XercesDOMParser parser;
    parser.setDoValidation(true);
    parser.setErrorHandler(&handler);
    
    try {
      //std::cout << "Parsing file: " << file << std::endl;
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
    
    // Check that this document is actually describing CCA components
    DOMElement *metacomponentmodel = 
      static_cast<DOMElement *>(document->getElementsByTagName(to_xml_ch_ptr("metacomponentmodel"))->item(0));
    
    std::string compModelName =
      to_char_ptr(metacomponentmodel->getAttribute(to_xml_ch_ptr("name")));
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
      //std::cout << "Library name = ->" << library_name << "<-" << std::endl;
      
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
	CCAComponentDescription* cd = new CCAComponentDescription(this);
	cd->type = component_name;
	cd->setLibrary(library_name.c_str()); // record the DLL name
	
	lock_components.lock();
	this->components[cd->type] = cd;
	lock_components.unlock();
      }
    }
  }
  

  ComponentInfo*
  CCAComponentModel::createComponent(const std::string& name,
				     const std::string& type,
				     const sci::cca::TypeMap::pointer& properties)
    
  {
    sci::cca::Component::pointer component;
    
    SCIRun::Guard guard(&components_lock);
    
    componentDB_type::iterator iter = components.find(type);
    if (iter == components.end()) {
      std::cerr << "Error: could not locate any cca components.\n"
		<< " Make sure the paths set in environment variable \"SIDL_DLL_PATH\" are correct." 
		<< std::endl;
      return 0;
    }
    
    // Get the list of DLL paths to search for the appropriate component library
    std::vector<std::string> possible_paths = splitPathString(this->getSidlDLLPath());
    LIBRARY_HANDLE handle;
      
    for (std::vector<std::string>::iterator it = possible_paths.begin();
	 it != possible_paths.end(); it++) {
      std::string so_name = *it + "/" + iter->second->getLibrary();
      handle = GetLibraryHandle(so_name.c_str());
      if (handle) break; 
    }
    
    if(!handle) {
      std::cerr << "Cannot load component " << type << std::endl;
      std::cerr << SOError() << std::endl;
      return 0;
    }
    
    std::string makername = "make_"+type;
    for (int i = 0; i < (int)makername.size(); i++) {
      if (makername[i] == '.') {
	makername[i]='_';
      }
    }
    void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
    if(!maker_v) {
      std::cerr <<"Cannot load component " << type << std::endl;
      std::cerr << SOError() << std::endl;
      return 0;
    }
    sci::cca::Component::pointer (*maker)() = (sci::cca::Component::pointer (*)())(maker_v);

    // create the component
    component = (*maker)();

    // create ComponentInfo. this must be a class the derives from Distributed/ComponentInfo
    ComponentInfo *info =  new ComponentInfo(framework, name, type, properties, component);

    // CCA initialization of the component
    component->setServices(sci::cca::Services::pointer(info));

    // done
    return info;
  }

  void CCAComponentModel::destroyComponent(const ComponentInfo::pointer &info)
  {
    // TODO [yarden]: need to release services on the component via a callback [is this sr2 specific?]

    info->getComponent()->deleteReference();
  }
  
  
#if 0
  void CCAComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list, bool /*listInternal*/)
  {
    lock_components.lock();
    for (componentDB_type::iterator iter = components.begin();
	 iter != components.end(); iter++) {
      list.push_back(iter->second);
    }
    lock_components.unlock();
    
    lock_loaderList.lock(); 
    for (unsigned int i = 0; i < loaderList.size(); i++) {
      SSIDL::array1<std::string> typeList;
      loaderList[i]->listAllComponentTypes(typeList);
      //convert typeList to component description list
      //by attaching a loader (resourceReferenece) to it.
      for (unsigned int j = 0; j < typeList.size(); j++) {
	CCAComponentDescription* cd = new CCAComponentDescription(this);
	cd->type = typeList[j];
	cd->setLoaderName(loaderList[i]->getName());
	list.push_back(cd);
      }
    }
    lock_loaderList.unlock();
  }
#endif  

} // end namespace SCIRun
