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
#include <Core/CCA/Component/PIDL/PIDL.h>
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
  string component_path = "../src/CCA/Components/xml";
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
    CCAComponentDescription* cd = new CCAComponentDescription(this);
    DOMNode* name = d->getAttributes()->getNamedItem(to_xml_ch_ptr("name"));
    if (name==0) {
      cout << "ERROR: Component has no name." << endl;
      cd->type = "unknown type";
    } else {
      cd->type = to_char_ptr(name->getNodeValue());
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
  //cerr << "CCA looking for component of type: " << type << '\n';
  return components.find(type) != components.end();
}

ComponentInstance* CCAComponentModel::createInstance(const std::string& name,
						     const std::string& type)
{
  std::string url=""; 
  gov::cca::Component::pointer component;
  if(url==""){  //local component 
    componentDB_type::iterator iter = components.find(type);
    if(iter == components.end())
      return 0;
    //ComponentDescription* cd = iter->second;

    string lastname=type.substr(type.find('.')+1);  
    string so_name("lib/libCCA_Components_");
    so_name=so_name+lastname+".so";
    cerr<<"type="<<type<<" soname="<<so_name<<endl;

    LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
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
      cerr <<"Cannot load component " << type << '\n';
      cerr << SOError() << '\n';
      return 0;
    }
    gov::cca::Component::pointer (*maker)() = (gov::cca::Component::pointer (*)())(maker_v);
    component = (*maker)();
  }
  else{ //remote component: need to be created by framework at url 

    Object::pointer obj=PIDL::objectFrom(url);
    if(obj.isNull()){
      cerr<<"got null obj (framework) from "<<url<<endl;
      return 0;
    }

    gov::cca::AbstractFramework::pointer remoteFramework=
      pidl_cast<gov::cca::AbstractFramework::pointer>(obj);

    std::string comURL; //=remoteFramework->createComponent(name, type);
    //cerr<<"comURL="<<comURL<<endl;
    Object::pointer comObj=PIDL::objectFrom(comURL);
    if(comObj.isNull()){
      cerr<<"got null obj(Component) from "<<url<<endl;
      return 0;
    }
    component=pidl_cast<gov::cca::Component::pointer>(comObj);

  }
  CCAComponentInstance* ci = new CCAComponentInstance(framework, name, type,
						      gov::cca::TypeMap::pointer(0),
						      component);
  ci->addReference(); //what is this for?
  component->setServices(gov::cca::Services::pointer(ci));
  return ci;
}

bool CCAComponentModel::destroyInstance(ComponentInstance *ci)
{
  CCAComponentInstance* cca_ci = dynamic_cast<CCAComponentInstance*>(ci);
  if(!cca_ci){
	cerr<<"error: in destroyInstance() cca_ci is 0"<<endl;  	
    return false;
  }
  cca_ci->deleteReference();
  return true;	
}

string CCAComponentModel::getName() const
{
  return "CCA";
}

void CCAComponentModel::listAllComponentTypes(vector<ComponentDescription*>& list,
					      bool /*listInternal*/)
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    list.push_back(iter->second);
  }
}



std::string CCAComponentModel::createComponent(const std::string& name,
						      const std::string& type)
						     
{
  
  gov::cca::Component::pointer component;
  componentDB_type::iterator iter = components.find(type);
  if(iter == components.end())
    return "";
    //ComponentDescription* cd = iter->second;
  
  string lastname=type.substr(type.find('.')+1);  
  string so_name("lib/libCCA_Components_");
  so_name=so_name+lastname+".so";
  //cerr<<"type="<<type<<" soname="<<so_name<<endl;
  
  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
  if(!handle){
    cerr << "Cannot load component " << type << '\n';
    cerr << SOError() << '\n';
    return "";
  }

  string makername = "make_"+type;
  for(int i=0;i<(int)makername.size();i++)
    if(makername[i] == '.')
      makername[i]='_';
  
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if(!maker_v){
    cerr << "Cannot load component " << type << '\n';
    cerr << SOError() << '\n';
    return "";
  }
  gov::cca::Component::pointer (*maker)() = (gov::cca::Component::pointer (*)())(maker_v);
  component = (*maker)();
  //need to make sure addReference() will not cause problem
  component->addReference();
  return component->getURL().getString();
}
