/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#include <SCIRun/SCIRunErrorHandler.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Util/soloader.h>

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

#include <iostream>
using namespace std;
using namespace SCIRun;

CCAComponentModel::CCAComponentModel(SCIRunFramework* framework)
  : ComponentModel("cca"), framework(framework)
{
  buildComponentList();
}

CCAComponentModel::~CCAComponentModel()
{
  destroyComponentList();
}

void CCAComponentModel::destroyComponentList()
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    delete iter->second;
  }
  components.clear();
}

void CCAComponentModel::buildComponentList()
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
  string component_path = "../src/CCA/Components/Builder";
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
    d.getFilenamesBySuffix(".cca", files);
    for(vector<string>::iterator iter = files.begin();
	iter != files.end(); iter++){
      string& file = *iter;
      readComponentDescription(dir+"/"+file);
    }
  }
}

void CCAComponentModel::readComponentDescription(const std::string& file)
{
  cerr << "Reading file " << file << '\n';
  // Instantiate the DOM parser.
  DOMParser parser;
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
  
  DOM_Document doc = parser.getDocument();
  DOM_NodeList list = doc.getElementsByTagName("component");
  int nlist = list.getLength();
  if(nlist == 0){
    cerr << "WARNING: file " << file << " has no components!\n";
  }
  for (int i=0;i<nlist;i++){
    DOM_Node d = list.item(i);
    CCAComponentDescription* cd = new CCAComponentDescription(this);
    DOM_Node name = d.getAttributes().getNamedItem("name");
    if (name==0) {
      cout << "ERROR: Component has no name." << endl;
      cd->type = "unknown type";
    } else {
      cd->type = name.getNodeValue().transcode();
    }
  
    for (DOM_Node child = d.getFirstChild();
	 child!=0;
	 child = child.getNextSibling()) {
      DOMString childname = child.getNodeName();
    }
    componentDB_type::iterator iter = components.find(cd->type);
    if(iter != components.end()){
      cerr << "WARNING: Component multiply defined: " << cd->type << '\n';
    } else {
      cerr << "Added CCA component of type: " << cd->type << '\n';
      components[cd->type]=cd;
    }
  }
}

gov::cca::Services::pointer
CCAComponentModel::createServices(const std::string& instanceName,
				  const std::string& className,
				  const gov::cca::TypeMap::pointer& properties)
{
  CCAComponentInstance* ci = new CCAComponentInstance(framework,
						      instanceName, className,
						      properties,
						      gov::cca::Component::pointer(0));
  framework->registerComponent(ci, instanceName);
  ci->addReference();
  return gov::cca::Services::pointer(ci);
}

bool CCAComponentModel::haveComponent(const std::string& type)
{
  cerr << "CCA looking for component of type: " << type << '\n';
  return components.find(type) != components.end();
}

ComponentInstance* CCAComponentModel::createInstance(const std::string& name,
						     const std::string& type)
{
  componentDB_type::iterator iter = components.find(type);
  if(iter == components.end())
    return 0;
  //ComponentDescription* cd = iter->second;

  LIBRARY_HANDLE handle = GetLibraryHandle("lib/libCCA_Components_Builder.so");
  if(!handle){
    cerr << "Cannot load component " << type << '\n';
    cerr << SOError() << '\n';
    return 0;
  }

  string makername = "make_"+type;
  for(int i=0;i<(int)makername.size();i++)
    if(makername[i] == '.')
      makername[i]='_';
  
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if(!maker_v){
    cerr << "Cannot load component " << type << '\n';
    cerr << SOError() << '\n';
    return 0;
  }
  gov::cca::Component::pointer (*maker)() = (gov::cca::Component::pointer (*)())(maker_v);
  gov::cca::Component::pointer component = (*maker)();
  CCAComponentInstance* ci = new CCAComponentInstance(framework, name, type,
						      gov::cca::TypeMap::pointer(0),
						      component);
  component->setServices(gov::cca::Services::pointer(ci));
  ci->addReference();
  return ci;
}

string CCAComponentModel::getName() const
{
  return "CCA";
}

void CCAComponentModel::listAllComponentTypes(vector<ComponentDescription*>& list,
						 bool listInternal)
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    list.push_back(iter->second);
  }
}
