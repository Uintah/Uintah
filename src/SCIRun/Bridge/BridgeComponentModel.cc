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
 *  BridgeComponentModel.cc:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#include <SCIRun/Bridge/BridgeComponentModel.h>
#include <SCIRun/Bridge/BridgeComponentDescription.h>
#include <SCIRun/Bridge/BridgeComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/SCIRunErrorHandler.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Util/soloader.h>
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
using namespace std;
using namespace SCIRun;

BridgeComponentModel::BridgeComponentModel(SCIRunFramework* framework)
  : ComponentModel("cca"), framework(framework)
{
  buildComponentList();
}

BridgeComponentModel::~BridgeComponentModel()
{
  destroyComponentList();
}

void BridgeComponentModel::destroyComponentList()
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    delete iter->second;
  }
  components.clear();
}

void BridgeComponentModel::buildComponentList()
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
    vector<string> files;
    d.getFilenamesBySuffix(".bridge", files);
    for(vector<string>::iterator iter = files.begin();
	iter != files.end(); iter++){
      string& file = *iter;
      readComponentDescription(dir+"/"+file);
    }
  }
}

void BridgeComponentModel::readComponentDescription(const std::string& file)
{
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
    //should use correct Loader pointer below.
    BridgeComponentDescription* cd = new BridgeComponentDescription(this);
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
      cerr << "Added Bridge component of type: " << cd->type << '\n';
      components[cd->type]=cd;
    }
  }
}

BridgeServices*
BridgeComponentModel::createServices(const std::string& instanceName,
				  const std::string& className)
{
  BridgeComponentInstance* ci = new BridgeComponentInstance(framework,
						      instanceName, className,
						      NULL);
  framework->registerComponent(ci, instanceName);
  return ci;
}

bool BridgeComponentModel::haveComponent(const std::string& type)
{
  cerr << "Bridge looking for component of type: " << type << '\n';
  return components.find(type) != components.end();
}



ComponentInstance* BridgeComponentModel::createInstance(const std::string& name,
						     const std::string& type)

{
  std::string loaderName="";
  cerr<<"creating component <"<<name<<","<<type<<"> with loader:"<<loaderName<<endl;
  BridgeComponent* component;
  if(loaderName==""){  //local component
    componentDB_type::iterator iter = components.find(type);
    if(iter == components.end())
      return 0;
    //ComponentDescription* cd = iter->second;

    string lastname=type.substr(type.find('.')+1);  
    string so_name("lib/libCCA_Components_");
    so_name=so_name+lastname+".so";
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
    BridgeComponent* (*maker)() = (BridgeComponent* (*)())(maker_v);
    component = (*maker)();
  }
  else{     
    //No way to remotely load bridge components for now
    return NULL;
    /*//use loader to load the component
    resourceReference* loader=getLoader(loaderName);
    std::vector<int> nodes;
    nodes.push_back(0);
    Object::pointer comObj=loader->createInstance(name, type, nodes);
    component=pidl_cast<sci::cca::Component::pointer>(comObj);
    */
  }
  BridgeComponentInstance* ci = new BridgeComponentInstance(framework, name, type, component);
  component->setServices(ci);
  return ci;
}

bool BridgeComponentModel::destroyInstance(ComponentInstance *ci)
{
  BridgeComponentInstance* cca_ci = dynamic_cast<BridgeComponentInstance*>(ci);
  if(!cca_ci){
	cerr<<"error: in destroyInstance() cca_ci is 0"<<endl;  	
    return false;
  }
  return true;	
}

string BridgeComponentModel::getName() const
{
  return "Bridge";
}

void BridgeComponentModel::listAllComponentTypes(vector<ComponentDescription*>& list,
					      bool /*listInternal*/)
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    list.push_back(iter->second);
  }

  for(unsigned int i=0; i<loaderList.size(); i++){
    ::SSIDL::array1<std::string> typeList;
    loaderList[i]->listAllComponentTypes(typeList);
    //convert typeList to component description list
    //by attaching a loader (resourceReferenece) to it.
    for(unsigned int j=0; j<typeList.size(); j++){
      BridgeComponentDescription* cd = new BridgeComponentDescription(this);
      cd->type=typeList[j];
      cd->setLoaderName(loaderList[i]->getName());
      list.push_back(cd);
    }
  }

}

int BridgeComponentModel::addLoader(resourceReference *rr){
  loaderList.push_back(rr);
  cerr<<"Loader "<<rr->getName()<<" is added into cca component model"<<endl;
  return 0;
}

int BridgeComponentModel::removeLoader(const std::string &loaderName)
{
  resourceReference *rr=getLoader(loaderName);
  if(rr!=0){
    cerr<<"loader "<<rr->getName()<<" is removed from cca component model\n";
    delete rr;
  }
  else{
    cerr<<"loader "<<loaderName<<" not found in cca component model\n";
  }
  return 0;
}

resourceReference *
BridgeComponentModel::getLoader(std::string loaderName){
  resourceReference *rr=0;
  for(unsigned int i=0; i<loaderList.size(); i++){
    if(loaderList[i]->getName()==loaderName){
      rr=loaderList[i];
      break;
    }
  }
  return rr;
}
