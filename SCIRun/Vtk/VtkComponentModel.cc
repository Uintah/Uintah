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
using namespace std;
using namespace SCIRun;

VtkComponentModel::VtkComponentModel(SCIRunFramework* framework)
  : ComponentModel("vtk"), framework(framework)
{
  buildComponentList();
}

VtkComponentModel::~VtkComponentModel()
{
  destroyComponentList();
}

void VtkComponentModel::destroyComponentList()
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    delete iter->second;
  }
  components.clear();
}

void VtkComponentModel::buildComponentList()
{
  // Initialize the XML4C system
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& toCatch) {
    std::cerr << "Error during initialization! :\n"
	 << StrX(toCatch.getMessage()) << endl;
    return;
  }

  destroyComponentList();
  string component_path = "../src/CCA/Components/VtkTest/xml";
  while(component_path != ""){
    unsigned int firstColon = component_path.find(':');
    string dir;
    if(firstColon < component_path.size()){
      dir=component_path.substr(0, firstColon);
      component_path = component_path.substr(firstColon+1);
    } else {
      dir = component_path;
      component_path="";
    }
    Dir d(dir);
    cerr << "Looking at directory: " << dir << '\n';
    vector<string> files;
    d.getFilenamesBySuffix(".xml", files);
    for(vector<string>::iterator iter = files.begin();
	iter != files.end(); iter++){
      string& file = *iter;
      readComponentDescription(dir+"/"+file);
    }
  }
}

void VtkComponentModel::readComponentDescription(const std::string& file)
{
  cerr << "Reading file " << file << '\n';
  // Instantiate the DOM parser.
  XercesDOMParser parser;
  parser.setDoValidation(false);
  
  SCIRunErrorHandler handler;
  parser.setErrorHandler(&handler);
  
  try {
    parser.parse(file.c_str());
  }  catch (const XMLException& toCatch) {
    std::cerr << "Error during parsing: '" <<
      file << "'\nException message is:  " <<
      xmlto_string(toCatch.getMessage()) << '\n';
    handler.foundError=true;
    return;
  }
  
  DOMDocument* doc = parser.getDocument();
  DOMNodeList* list = doc->getElementsByTagName(to_xml_ch_ptr("component"));
  int nlist = list->getLength();
  if(nlist == 0){
    cerr << "WARNING: file " << file << " has no components!\n";
  }
  for (int i=0;i<nlist;i++){
    DOMNode* d = list->item(i);
    DOMNode* name = d->getAttributes()->getNamedItem(to_xml_ch_ptr("name"));
    string type;
    if (name==0) {
      cout << "ERROR: Component has no name." << endl;
      type = "unknown type"; 
    } else {
      type = to_char_ptr(name->getNodeValue());
    }

    VtkComponentDescription* cd = new VtkComponentDescription(this, type);
  
    componentDB_type::iterator iter = components.find(type);
    if(iter != components.end()){
      cerr << "WARNING: Component multiply defined: " << type << '\n';
    } else {
      cerr << "Added Vtk component of type: " << type << '\n';
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
  
  string lastname=type.substr(type.find('.')+1);  
  string so_name("lib/libCCA_Components_VtkTest_");
  so_name=so_name+lastname+".so";
  cerr<<"type="<<type<<" soname="<<so_name<<endl;

  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
  if(!handle){
    cerr << "Cannot load component .so " << type << '\n';
    cerr << SOError() << '\n';
    return 0;
  }

  string makername = "make_"+type;
  for(int i=0;i<(int)makername.size();i++)
    if(makername[i] == '.')
      makername[i]='_';
  
  cerr<<"looking for symbol:"<< makername<<endl;
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if(!maker_v){
    cerr <<"Cannot load component symbol " << type << '\n';
    cerr << SOError() << '\n';
    return 0;
  }
  vtk::Component* (*maker)() = (vtk::Component* (*)())(maker_v);
  cerr<<"about to create Vtk component"<<endl;
  component = (*maker)();

  
  VtkComponentInstance* ci = new VtkComponentInstance(framework, name, type,
						      component);
  cerr<<"comopnent instance ci is created!"<<endl;
  
  return ci;
}

bool VtkComponentModel::destroyInstance(ComponentInstance *ci)
{
  //TODO: pre-deletion clearance.
  delete ci;  
  return true;
}

string VtkComponentModel::getName() const
{
  return "Vtk";
}

void VtkComponentModel::listAllComponentTypes(vector<ComponentDescription*>& list,
					      bool /*listInternal*/)
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    list.push_back(iter->second);
  }
}

