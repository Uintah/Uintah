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
#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Util/soloader.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <string>
#include "framework.hh"
#include "SIDL.hh"

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

BabelComponentModel::BabelComponentModel(SCIRunFramework* framework)
  : ComponentModel("babel"), framework(framework)
{
  buildComponentList();
}

BabelComponentModel::~BabelComponentModel()
{
  destroyComponentList();
}

void BabelComponentModel::destroyComponentList()
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    delete iter->second;
  }
  components.clear();
}

void BabelComponentModel::buildComponentList()
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
  string component_path = "../src/CCA/Components/BabelTest/xml";
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

void BabelComponentModel::readComponentDescription(const std::string& file)
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
    BabelComponentDescription* cd = new BabelComponentDescription(this);
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
      cerr << "Added Babel component of type: " << cd->type << '\n';
      components[cd->type]=cd;
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
  gov::cca::Services svc;
  cerr<<"BabelComponentModel::createServices() is not implemented !"<<endl;
  return svc;
}

bool BabelComponentModel::haveComponent(const std::string& type)
{
  cerr << "CCA(Babel) looking for babel component of type: " << type << '\n';
  return components.find(type) != components.end();
}

ComponentInstance* BabelComponentModel::createInstance(const std::string& name,
						       const std::string& type)
{
  
  gov::cca::Component component;
  if(true){  //local component 
    componentDB_type::iterator iter = components.find(type);
    if(iter == components.end())
      return 0;
    /*
    string lastname=type.substr(type.find('.')+1);  
    string so_name("lib/libBabel_Components_");
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
    gov::cca::Component (*maker)() = (gov::cca::Component (*)())(maker_v);
    cerr<<"about to create babel component"<<endl;
    component = (*maker)();
    */
    SIDL::BaseClass sidl_class = SIDL::Loader::createClass(type);
    component = sidl_class;
    if ( component._not_nil() ) { 
      cerr<<"babel component of type "<<type<< " is loaded!"<<endl;
    }
    else
            cerr<<"Cannot load babel component of type "<<type<<endl;
    cerr<<"babel component created!"<<endl;
 	
    }
  else{ //remote component: need to be created by framework at url 
    cerr<<"remote babel components creation is not done!"<<endl;
    /*
    Object::pointer obj=PIDL::objectFrom(url);
    if(obj.isNull()){
      cerr<<"got null obj (framework) from "<<url<<endl;
      return 0;
    }

    sci::cca::AbstractFramework::pointer remoteFramework=
      pidl_cast<sci::cca::AbstractFramework::pointer>(obj);

    std::string comURL=remoteFramework->createComponent(name, type);
    //cerr<<"comURL="<<comURL<<endl;
    Object::pointer comObj=PIDL::objectFrom(comURL);
    if(comObj.isNull()){
      cerr<<"got null obj(Component) from "<<url<<endl;
      return 0;
    }
    component=pidl_cast<sci::cca::Component::pointer>(comObj);
    */
  }


  cerr<<"about to create services"<<endl;
  framework::Services svc=framework::Services::_create();
  cerr<<"services created !"<<endl;
  component.setServices(svc);
  cerr<<"component.setService done!"<<endl;
  gov::cca::Component nullMap;

  BabelComponentInstance* ci = new BabelComponentInstance(framework, name, type,
							  nullMap, 
							  component,
							  svc);
  cerr<<"comopnent instance ci is created!"<<endl;
  //ci->addReference();
  return ci;
}

bool BabelComponentModel::destroyInstance(ComponentInstance *ci)
{
  cerr<<"BabelComponentModel::destroyInstance() is not done"<<endl;
  //make sure why ci->addReference() is called in createInstace();
  delete ci;  
  return false;
}

string BabelComponentModel::getName() const
{
  return "Babel";
}

void BabelComponentModel::listAllComponentTypes(vector<ComponentDescription*>& list,
					      bool /*listInternal*/)
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    list.push_back(iter->second);
  }
}



std::string BabelComponentModel::createComponent(const std::string& name,
						      const std::string& type)
						     
{
  
  sci::cca::Component::pointer component;
  componentDB_type::iterator iter = components.find(type);
  if(iter == components.end())
    return "";
  
  string lastname=type.substr(type.find('.')+1);  
  string so_name("lib/libBabel_Components_");
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
  /*  sci::cca::Component::pointer (*maker)() = (sci::cca::Component::pointer (*)())(maker_v);
  component = (*maker)();
  //need to make sure addReference() will not cause problem
  component->addReference();
  return component->getURL().getString();
  */
  return ""; 
}
