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
#include <SCIRun/Core/CoreFrameworkImpl.h>

#include <SCIRun/Plume/CCAComponentClassDescriptionImpl.h>
#include <SCIRun/Plume/CCAComponentClassFactoryImpl.h>
//#include <SCIRun/Core/FrameworkPropertiesService.h>
#include <SCIRun/Core/CoreServicesImpl.h>
#include <SCIRun/Plume/SCIRunErrorHandler.h>

#include <Core/Thread/Guard.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Core/XMLUtil/StrX.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/CCA/PIDL/PIDL.h>
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

  using namespace sci::cca;
  
  const std::string CCAComponentModel::DEFAULT_PATH = std::string("/CCA/Components/xml");
  
  
  CCAComponentModel::CCAComponentModel(const CoreFramework::pointer &framework)
    : framework(framework),
      descriptions_lock("CCAComponentModel::components lock")
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
    SCIRun::Guard g1(&descriptions_lock);
    descriptions.clear();
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
    sci::cca::TypeMap::pointer properties;

#if 0
    FrameworkPropertiesService::pointer service =
      pidl_cast<FrameworkPropertiesService::pointer>(framework->getFrameworkService("cca.FrameworkProperties"));

    if (service.isNull()) {
      std::cerr << "Error: Can not find framework properties\n" ;
      //return sci_getenv("SCIRUN_SRCDIR") + DEFAULT_PATH;
    } else {
      properties = service->getProperties();
      sArray = properties->getStringArray("sidl_xml_path", sArray);
    }
    framework->releaseFrameworkService(service);
#else
    sArray.push_back("/local/home/yarden/projects/plume/src/CCA/Core/xml");
#endif

    for (SSIDL::array1<std::string>::iterator item = sArray.begin(); item != sArray.end(); item++) {
      Dir dir(*item);
      //std::cout << "CCA Component Model: Looking at directory: " << *item << std::endl;
      std::vector<std::string> files;
      dir.getFilenamesBySuffix(".xml", files);
      
      for(std::vector<std::string>::iterator iter = files.begin();
	  iter != files.end(); iter++) {
	std::string& file = *iter;
	//std::cout << "CCA Component Model: Looking at file" << file << std::endl;
	readComponentClassDescription(*item+"/"+file);
      }
    }
  }
  
  void CCAComponentModel::readComponentClassDescription(const std::string& file)
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
      std::cerr << "Unknown error occurred during parsing: '" << file << "'\n";
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
    
    // Get a list of the library nodes.  Traverse the list and read component
    // elements at each list node.
    DOMNodeList* libraries
      = document->getElementsByTagName(to_xml_ch_ptr("library"));
    
    for (unsigned int i = 0; i < libraries->getLength(); i++) {
      DOMElement *library = static_cast<DOMElement *>(libraries->item(i));
      // Read the library name
      std::string library_name(to_char_ptr(library->getAttribute(to_xml_ch_ptr("name"))));
      std::cout << "Library [" << library_name << "]\n";
      
      // Get the list of components.
      DOMNodeList* comps = library->getElementsByTagName(to_xml_ch_ptr("component"));
      for (unsigned int j = 0; j < comps->getLength(); j++) {
	// Read the component name
	DOMElement *component = static_cast<DOMElement *>(comps->item(j));
	std::string component_name(to_char_ptr(component->getAttribute(to_xml_ch_ptr("name"))));
	std::cout << "\tComponent [" << component_name << "]\n";
	
	// Register this component
	CCAComponentClassDescription::pointer cd = new CCAComponentClassDescriptionImpl(component_name, library_name );
	
	descriptions_lock.lock();
	this->descriptions[cd->getComponentClassName()] = cd;
	descriptions_lock.unlock();

	// add a factory 
	framework->addComponentClassFactory( new CCAComponentClassFactoryImpl(cd, this) );
      }
    }
  }
  
  
  CoreServies::pointer
  CCAComponentModel::createComponent(const std::string &name,
				     const std::string &type,
				     const std::string &library,
				     const sci::cca::TypeMap::pointer &properties)
    
  {
    Component::pointer component;
    
    // Get the list of DLL paths to search for the appropriate component library
    std::vector<std::string> possible_paths = splitPathString(getSidlDLLPath());
    LIBRARY_HANDLE handle;
      
    for (std::vector<std::string>::iterator it = possible_paths.begin();
	 it != possible_paths.end(); it++) {
      std::string so_name = *it + "/" + library;
      handle = GetLibraryHandle(so_name.c_str());
      if (handle) break; 
    }
    
    if(!handle) {
      std::cerr << "Can not load component " << type << std::endl;
      std::cerr << SOError() << std::endl;
      return ComponentInfo::pointer(0);
    }
    
    std::string makername = "make_"+type;
    for (int i = 0; i < (int)makername.size(); i++) {
      if (makername[i] == '.') {
	makername[i]='_';
      }
    }
    void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
    if(!maker_v) {
      std::cerr <<"Can not load component " << type << std::endl;
      std::cerr << SOError() << std::endl;
      return CoreServices::pointer(0);
    }
    sci::cca::Component::pointer (*maker)() = (Component::pointer (*)())(maker_v);

    // create the component
    component = (*maker)();

    // create ComponentInfo. this must be a class the derives from Distributed/ComponentInfo
    CoreServices::pointer services;
    try {
       services =  new CoreServicesImpl(framework, name, type, properties, component);
    } catch (...) {
      std::cerr << "component crashed in constructor\n";
      services = 0;
    }

    // done
    return services;
  }

  void CCAComponentModel::destroyComponent(const ComponentInfo::pointer &info)
  {
    // Note [yarden]: called by the ComponentInfo destructor. Thus do not delete or decrement the info counter
    // TODO [yarden]: need to release services on the component via a callback [is this sr2 specific?]
  }
  
  
  std::vector<std::string>
  CCAComponentModel::splitPathString(const std::string &path)
  {
    std::vector<std::string> ans;
    if (path == "" ) {
      return ans;
    }
    
    // Split the PATH string into a list of paths.  Key on ';' token.
    std::string::size_type start = 0;
    std::string::size_type end = path.find(';', start);
    while (end != path.npos) {
      std::string substring = path.substr(start, end - start);
      ans.push_back(substring);
      start = end + 1;
      end = path.find(';', start);
    }
    // grab the remaining path
    std::string substring = path.substr(start, end - start);
    ans.push_back(substring);
    
    return ans;  
  }

} // end namespace SCIRun
