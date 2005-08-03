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
 *  CorbaComponentModel.cc: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <SCIRun/Corba/CorbaComponentModel.h>
#include <SCIRun/Corba/CorbaComponentDescription.h>
#include <SCIRun/Corba/CorbaComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/SCIRunErrorHandler.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Core/XMLUtil/StrX.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/Util/sci_system.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

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
#include <SCIRun/Corba/Port.h>
#include <SCIRun/Corba/Component.h>
#include <SCIRun/Corba/Services.h>

extern "C" {
#include <string.h>
}

namespace SCIRun {


const std::string CorbaComponentModel::DEFAULT_PATH =
    std::string("/CCA/Components/CORBA/xml");


CorbaComponentModel::CorbaComponentModel(SCIRunFramework* framework)
  : ComponentModel("corba"), framework(framework)
{
  // move to framework properties
  // Record the path containing DLLs for components.
  const char *dll_path = getenv("SIDL_DLL_PATH");
  if (dll_path != 0) {
    this->setSidlDLLPath(std::string(dll_path));
  } else {
    this->setSidlDLLPath(sci_getenv("SCIRUN_OBJDIR") + std::string("/lib"));
  }

  // Set the default DTD or Schema name for xml validation
  //  this->setGrammarFileName( std::string("metacomponentmodel.dtd") );
  
  buildComponentList();
}

CorbaComponentModel::~CorbaComponentModel()
{
  destroyComponentList();
}

void CorbaComponentModel::destroyComponentList()
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++) {
    delete iter->second;
  }
  components.clear();
}

void CorbaComponentModel::buildComponentList()
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
        std::cerr << "CORBA Component Model: Looking at directory: " << *it << std::endl;
        std::vector<std::string> files;
        d.getFilenamesBySuffix(".xml", files);

        for(std::vector<std::string>::iterator iter = files.begin();
            iter != files.end(); iter++) {
          std::string& file = *iter;
          std::cerr << "CORBA Component Model: Looking at file" << file << std::endl;
          readComponentDescription(*it+"/"+file);
        }
  }
}

void CorbaComponentModel::readComponentDescription(const std::string& file)
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

  // Check that this document is actually describing CORBA components
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
      CorbaComponentDescription* cd = new CorbaComponentDescription(this, component_name);
      cd->setExecPath(library_name.c_str()); // record the executable's full path
      this->components[cd->type] = cd;
     }
  }
}

bool CorbaComponentModel::haveComponent(const std::string& type)
{
  return components.find(type) != components.end();
}

ComponentInstance*
CorbaComponentModel::createInstance(const std::string& name,
                                    const std::string& type,
                                    const sci::cca::TypeMap::pointer &tm)
{
    corba::Component *component;

    componentDB_type::iterator iter = components.find(type);
    if (iter == components.end()) { // could not find this component
        return 0;
    }

    std::string exec_name = iter->second->getExecPath();

    // If the component library does not exist
    // do not create the component instance.
    struct stat buf;
    if (LSTAT(exec_name.c_str(), &buf) < 0) {
        if (errno == ENOENT) {
            throw InternalError("File " + exec_name + " does not exist.", __FILE__, __LINE__);
        } else {
            throw InternalError("LSTAT on " + exec_name + " failed.", __FILE__, __LINE__);
        }
    }

  component = new corba::Component();
  corba::Services *svc=new corba::Services(component);
  component->setServices(svc);

  sci::cca::CorbaServices::pointer services(svc);
  services->addReference();
  std::string svc_url=services->getURL().getString();

  /*    
  pid_t child_id=fork();
  if(child_id==0){
    //this is child process
    execl(exec_name.c_str(), exec_name.c_str(), svc_url.c_str(), NULL);
    exit(0);
  }else{
    std::cout<<"**** main process is still running ****"<<std::endl;
    //this is parent process
    //do nothing
  }
  */
  string cmdline=exec_name+" "+svc_url.c_str()+"&";
  const int status = sci_system(cmdline.c_str());
  if (status != 0) { // failed
    throw InternalError("Corba service " + cmdline + " is not available.", __FILE__, __LINE__);
  }

  services->check();

  //TODO: do we really need a "component" here?
  CorbaComponentInstance* ci =
      new CorbaComponentInstance(framework, name, type, tm, component);
  return ci;
}

bool CorbaComponentModel::destroyInstance(ComponentInstance *ci)
{
  //TODO: pre-deletion clearance.
  delete ci;  
  return true;
}

std::string CorbaComponentModel::getName() const
{
    return "Corba";
}

void CorbaComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list, bool /*listInternal*/)
{
  for (componentDB_type::iterator iter=components.begin(); iter != components.end(); iter++) {
    list.push_back(iter->second);
  }
}

} // end namespace SCIRun
