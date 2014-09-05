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
 *  VtkComponentModel.cc: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */


#include <SCIRun/Vtk/VtkComponentModel.h>
#include <SCIRun/Vtk/VtkComponentDescription.h>
#include <SCIRun/Vtk/VtkComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/SCIRunErrorHandler.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Util/soloader.h>
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
#include <SCIRun/Vtk/Port.h>
#include <SCIRun/Vtk/Component.h>

extern "C" {
#include <string.h>
}

using namespace SCIRun;

VtkComponentModel::VtkComponentModel(SCIRunFramework* framework)
  : ComponentModel("vtk"), framework(framework)
{
  // Record the path to XML descriptions of components.  The environment
  // variable SIDL_XML_PATH should be set, otherwise use a default.
  const char *component_path = getenv("SIDL_XML_PATH");
  if (component_path != 0)
    {
    this->setSidlXMLPath( std::string(component_path) );
    }
  else
    {
    this->setSidlXMLPath("../src/CCA/Components/VTK/xml");
    }

  // Record the path containing DLLs for components.
  this->setSidlDLLPath( std::string( getenv("SIDL_DLL_PATH") ));

  // Set the default DTD or Schema name for xml validation
  //  this->setGrammarFileName( std::string("metacomponentmodel.dtd") );
  
  buildComponentList();
}

VtkComponentModel::~VtkComponentModel()
{
  destroyComponentList();
}

void VtkComponentModel::destroyComponentList()
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++)
    {
    delete iter->second;
    }
  components.clear();
}

void VtkComponentModel::buildComponentList()
{
  // Initialize the XML4C system
  try
    {
    XMLPlatformUtils::Initialize();
    }
  catch (const XMLException& toCatch)
    {
    std::cerr << "Error during initialization! :" << std::endl
              << StrX(toCatch.getMessage()) << std::endl;
    return;
    }
  
  destroyComponentList();

  std::string component_path(this->getSidlXMLPath());

  std::vector< std::string > paths = this->splitPathString(component_path);

  for (std::vector< std::string>::const_iterator it = paths.begin();
       it != paths.end(); ++it)
    {
    Dir d(*it);
    std::cerr << "Looking at directory: " << *it << std::endl;
    std::vector<std::string> files;
    d.getFilenamesBySuffix(".xml", files);

    for(std::vector<std::string>::iterator iter = files.begin();
        iter != files.end(); iter++)
      {
      std::string& file = *iter;
      readComponentDescription(*it+"/"+file);
      }
    }
}

void VtkComponentModel::readComponentDescription(const std::string& file)
{
  // Instantiate the DOM parser.
  SCIRunErrorHandler handler;
  XercesDOMParser parser;
  parser.setDoValidation(true);
  parser.setErrorHandler(&handler);
  
  try
    {
    std::cout << "Parsing file: " << file << std::endl;
    parser.parse(file.c_str());
    }
  catch (const XMLException& toCatch)
    {
    std::cerr << "Error during parsing: '" <<
      file << "' " << std::endl << "Exception message is:  " <<
      xmlto_string(toCatch.getMessage()) << std::endl;
    handler.foundError=true;
    return;
    }
  catch ( ... )
    {
    std::cerr << "Unknown error occurred during parsing: '" << file << "' "
              << std::endl;
    handler.foundError=true;
    return;
    }

  // Get all the top-level document node
  DOMDocument* document = parser.getDocument();

  // Check that this document is actually describing VTK components
  DOMElement *metacomponentmodel = static_cast<DOMElement *>(
    document->getElementsByTagName(to_xml_ch_ptr("metacomponentmodel"))->item(0));

  std::string compModelName
    = to_char_ptr(metacomponentmodel->getAttribute(to_xml_ch_ptr("name")));
  std::cout << "Component model name = " << compModelName << std::endl;

  if ( compModelName != std::string(this->prefixName) )
    {
    return;
    }
  
  // Get a list of the library nodes.  Traverse the list and read component
  // elements at each list node.
  DOMNodeList* libraries
    = document->getElementsByTagName(to_xml_ch_ptr("library"));

  for (unsigned int i = 0; i < libraries->getLength(); i++)
    {
    DOMElement *library = static_cast<DOMElement *>(libraries->item(i));
    // Read the library name
    std::string library_name(to_char_ptr(library->getAttribute(to_xml_ch_ptr("name"))));
    std::cout << "Library name = ->" << library_name << "<-" << std::endl;

    // Get the list of components.
    DOMNodeList* comps
      = library->getElementsByTagName(to_xml_ch_ptr("component"));
    for (unsigned int j = 0; j < comps->getLength(); j++)
      {
      // Read the component name
      DOMElement *component = static_cast<DOMElement *>(comps->item(j));
      std::string
        component_name(to_char_ptr(component->getAttribute(to_xml_ch_ptr("name"))));
      std::cout << "Component name = ->" << component_name << "<-" << std::endl;

      // Register this component
      VtkComponentDescription* cd = new VtkComponentDescription(this, component_name);
      cd->setLibrary(library_name.c_str()); // record the DLL name
      this->components[cd->type] = cd;
      }
    }
}

bool VtkComponentModel::haveComponent(const std::string& type)
{
  return components.find(type) != components.end();
}

std::vector<std::string>
VtkComponentModel::splitPathString(const std::string &path)
{
  std::vector<std::string> ans;

  if (path == "" )
    {
    return ans;
    }

  // Split the PATH string into a list of paths.  Key on ';' token.
  std::string::size_type start = 0;
  std::string::size_type end   = path.find(';', start);
  while ( end != path.npos )
    {
    std::string substring = path.substr(start, end - start);
    ans.push_back(substring);
    start = end + 1;
    end   = path.find(';', start);
    }
  // grab the remaining path
  std::string substring = path.substr(start, end - start);
  ans.push_back(substring);

  return ans;  
}

ComponentInstance* VtkComponentModel::createInstance(const std::string& name,
						     const std::string& type)
{
  vtk::Component *component;

  componentDB_type::iterator iter = components.find(type);
  if(iter == components.end())
    { // Could not find this component
    return 0;
    }

  // Get the list of DLL paths to search for the appropriate component library
  std::vector<std::string> possible_paths = this->splitPathString(this->getSidlDLLPath());
  LIBRARY_HANDLE handle;

  for (std::vector<std::string>::iterator it = possible_paths.begin();
       it != possible_paths.end(); it++)
    {
    std::string so_name = *it + "/" + iter->second->getLibrary();
    handle = GetLibraryHandle(so_name.c_str());
    if (handle)  {  break;   }
    }
   
  if ( !handle )
    {
    std::cerr << "Could not find component DLL: " << iter->second->getLibrary()
              << " for type " << type << std::endl;
    std::cerr << SOError() << std::endl;
    return 0;
    }
  
  std::string makername = "make_"+type;
  for(int i = 0; i < static_cast<int>(makername.size()); i++)
    {
    if (makername[i] == '.') { makername[i]='_'; }
    }
  
  //  std::cerr << "looking for symbol:" << makername << std::endl;
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if(!maker_v)
    {
    //    std::cerr <<"Cannot load component symbol " << type << std::endl;
    std::cerr << SOError() << std::endl;
    return 0;
    }
  vtk::Component* (*maker)() = (vtk::Component* (*)())(maker_v);
  //  std::cerr << "about to create Vtk component" << std::endl;
  component = (*maker)();
  
  VtkComponentInstance* ci = new VtkComponentInstance(framework, name, type,
						      component);
  return ci;
}

bool VtkComponentModel::destroyInstance(ComponentInstance *ci)
{
  //TODO: pre-deletion clearance.
  delete ci;  
  return true;
}

std::string VtkComponentModel::getName() const
{
  return "Vtk";
}

void VtkComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list,
					      bool /*listInternal*/)
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    list.push_back(iter->second);
  }
}

