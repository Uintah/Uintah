/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
    this->setSidlXMLPath("../src/CCA/Components/VtkTest/xml");
    }

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

  while(component_path != "")
    {
    unsigned int firstColon = component_path.find(';');
    std::string dir;
    if(firstColon < component_path.size())
      {
      dir=component_path.substr(0, firstColon);
      component_path = component_path.substr(firstColon+1);
      }
    else
      {
      dir = component_path;
      component_path="";
      }

    Dir d(dir);
    std::cerr << "Looking at directory: " << dir << std::endl;
    std::vector<std::string> files;
    d.getFilenamesBySuffix(".xml", files);

    for(std::vector<std::string>::iterator iter = files.begin();
        iter != files.end(); iter++)
      {
      std::string& file = *iter;
      readComponentDescription(dir+"/"+file);
      }
    }
}

void VtkComponentModel::readComponentDescription(const std::string& file)
{
  // Instantiate the DOM parser.
  XercesDOMParser parser;
  parser.setDoValidation(false);
  
  SCIRunErrorHandler handler;
  parser.setErrorHandler(&handler);
  
  try
    {
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
  
  DOMDocument* doc = parser.getDocument();
  DOMNodeList* list = doc->getElementsByTagName(to_xml_ch_ptr("component"));

  int nlist = list->getLength();
  if (nlist == 0)
    {
    std::cerr << "WARNING: file " << file << " has no components!" << std::endl;
    }
  for (int i=0; i < nlist; i++)
    {
    DOMNode* d = list->item(i);
    
    // Is this component a Vtk component?
    DOMNode* model= d->getAttributes()->getNamedItem(to_xml_ch_ptr("model"));
    if (model != 0)
      {  
      if ( strcmp(to_char_ptr(model->getNodeValue()), this->prefixName.c_str()) != 0 )
        {// not a vtk component, ignore it
        continue;
        }
      }
    else
      { // No model, ignore this component
      std::cerr << "ERROR: Component has no model in file " << file << std::endl;
      continue;
      }
 
    DOMNode* name = d->getAttributes()->getNamedItem(to_xml_ch_ptr("name"));
    std::string type;

    if (name == 0)
      {
      std::cout << "ERROR: Component has no name." << std::endl;
      type = "unknown type"; 
      }
    else
      {
      type = to_char_ptr(name->getNodeValue());
      }
    
    VtkComponentDescription* cd = new VtkComponentDescription(this, type);
  
    componentDB_type::iterator iter = components.find(type);
    if(iter != components.end())
      {
      std::cerr << "WARNING: Component multiply defined: " << type << std::endl;
      }
    else
      {
      std::cerr << "Added Vtk component of type: " << type << std::endl;
      components[cd->type]=cd;
      }
    }
}

bool VtkComponentModel::haveComponent(const std::string& type)
{
  return components.find(type) != components.end();
}

ComponentInstance* VtkComponentModel::createInstance(const std::string& name,
						     const std::string& type)
{
  
  vtk::Component *component;
  //only local components supported at this time.

  componentDB_type::iterator iter = components.find(type);
  if(iter == components.end())
    return 0;
  
  std::string lastname=type.substr(type.find('.')+1);  
  std::string so_name("lib/libCCA_Components_VtkTest_");
  so_name=so_name+lastname+".so";
  std::cerr << "type=" << type << " soname=" << so_name << std::endl;

  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
  if(!handle){
    std::cerr << "Cannot load component .so " << type << std::endl;
    std::cerr << SOError() << std::endl;
    return 0;
  }

  std::string makername = "make_"+type;
  for(int i=0;i<(int)makername.size();i++)
    if(makername[i] == '.')
      makername[i]='_';
  
  std::cerr << "looking for symbol:" << makername << std::endl;
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if(!maker_v)
    {
    std::cerr <<"Cannot load component symbol " << type << std::endl;
    std::cerr << SOError() << std::endl;
    return 0;
    }
  vtk::Component* (*maker)() = (vtk::Component* (*)())(maker_v);
  std::cerr << "about to create Vtk component" << std::endl;
  component = (*maker)();

  
  VtkComponentInstance* ci = new VtkComponentInstance(framework, name, type,
						      component);
  std::cerr << "comopnent instance ci is created!" << std::endl;
  
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

