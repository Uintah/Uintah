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
 *  SCIRunLoader.cc: An instance of the SCIRun CCA Component Loader
 *
 *  Written by:
 *   Keming Zhang & Kosta
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>
#include <SCIRun/SCIRunLoader.h>
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
#include <Core/CCA/PIDL/PIDL.h>
#include <SCIRun/resourceReference.h>
#include <Core/CCA/spec/cca_sidl.h>
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
#include <sstream>


using namespace std;
using namespace SCIRun;

SCIRunLoader::SCIRunLoader(const string &loaderName,  const string & frameworkURL)
{
  /*  Object::pointer obj=PIDL::objectFrom(frameworkURL);
  if(obj.isNull()){
    cerr<<"Cannot get framework from url="<<frameworkURL<<endl;
    return;
  }
  sci::cca::AbstractFramework::pointer framework=pidl_cast<sci::cca::AbstractFramework::pointer>(obj);
  
  SSIDL::array1< std::string> URLs;
  URLs.push_back(this->getURL().getString());
  framework->registerLoader(loaderName, URLs);
  */
}

int SCIRunLoader::createPInstance(const string& componentName, const string& componentType, 
				  const sci::cca::TypeMap::pointer& properties,SSIDL::array1<std::string> &componentURLs) {

  //TODO: assume type is always good?

  cerr<<"SCIRunLoader::getRefCount()="<<getRefCount()<<endl;

  
  string lastname=componentType.substr(componentType.find('.')+1);  
  string so_name("lib/libCCA_Components_");
  so_name=so_name+lastname+".so";
  cerr<<"componentType="<<componentType<<" soname="<<so_name<<endl;
    
  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
  if(!handle){
    cerr << "Cannot load component " << componentType << '\n';
    cerr << SOError() << '\n';
    return 1;
  }
  string makername = "make_"+componentType;
  for(int i=0;i<(int)makername.size();i++)
    if(makername[i] == '.')
      makername[i]='_';
  
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if(!maker_v){
    cerr <<"Cannot load component " << componentType << '\n';
    cerr << SOError() << '\n';
    return 1;
  }
  sci::cca::Component::pointer (*maker)() = (sci::cca::Component::pointer (*)())(maker_v);
  sci::cca::Component::pointer component = (*maker)();
  //TODO: need keep a reference in the loader's creatation recored.
  component->addReference();

  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);

  cerr << "SCIRunLoader::createInstance..., rank/size="<<mpi_rank<<"/"<<mpi_size<<"\n";
  componentURLs.resize(1);
  componentURLs[0] = component->getURL().getString();
  //componentURLs.push_back(component->getURL().getString());
  cerr << "Done, rank/size="<<mpi_rank<<"/"<<mpi_size<<" and URL="<<component->getURL().getString()<<"\n";
  return 0;
}



int SCIRunLoader::createInstance(const string& componentName, const string& componentType, 
				 const sci::cca::TypeMap::pointer& properties,std::string &componentURL) {

  //TODO: assume type is always good?

  cerr<<"SCIRunLoader::getRefCount()="<<getRefCount()<<endl;

  
  string lastname=componentType.substr(componentType.find('.')+1);  
  string so_name("lib/libCCA_Components_");
  so_name=so_name+lastname+".so";
  cerr<<"componentType="<<componentType<<" soname="<<so_name<<endl;
    
  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
  if(!handle){
    cerr << "Cannot load component " << componentType << '\n';
    cerr << SOError() << '\n';
    return 1;
  }
  string makername = "make_"+componentType;
  for(int i=0;i<(int)makername.size();i++)
    if(makername[i] == '.')
      makername[i]='_';
  
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if(!maker_v){
    cerr <<"Cannot load component " << componentType << '\n';
    cerr << SOError() << '\n';
    return 1;
  }
  sci::cca::Component::pointer (*maker)() = (sci::cca::Component::pointer (*)())(maker_v);
  sci::cca::Component::pointer component = (*maker)();
  //TODO: need keep a reference in the loader's creatation recored.
  component->addReference();

  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);

  //componentURLs.resize(1);
  //componentURLs[0] = component->getURL().getString();
  //componentURLs.push_back(component->getURL().getString());
  componentURL=component->getURL().getString();
  cerr << "SCIRunLoader::createInstance Done, rank/size="<<mpi_rank<<"/"<<mpi_size<<"\n";
  return 0;
}


int SCIRunLoader::destroyInstance(const string& componentName, float time)
{
  cerr<<"destroyInstance not implemneted\n";
  return 0;
}

int SCIRunLoader::getAllComponentTypes(::SSIDL::array1< ::std::string>& componentTypes)
{
  cerr<<"listAllComponents() is called\n";
  buildComponentList();
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    componentTypes.push_back(iter->second->getType());
  }
}

int SCIRunLoader::shutdown(float timeout)
{
  cerr<<"shutdown() is called: should unregisterLoader()\n";
  return 0;
}


SCIRunLoader::~SCIRunLoader()
{
  cerr << "SCIRun  Loader exits.\n";
  //abort();
}

/*
std::string CCAComponentModel::createComponent(const std::string& name,
						      const std::string& type)
						     
{
  
  sci::cca::Component::pointer component;
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
  sci::cca::Component::pointer (*maker)() = (sci::cca::Component::pointer (*)())(maker_v);
  component = (*maker)();
  //need to make sure addReference() will not cause problem
  component->addReference();
  return component->getURL().getString();
}
*/


void SCIRunLoader::buildComponentList()
{
  // Initialize the XML4C system
  try {
    XMLPlatformUtils::Initialize();
  }catch (const XMLException& toCatch) {
    std::cerr << "Error during initialization! :\n"
	      << StrX(toCatch.getMessage()) << endl;
    return;
  }

  destroyComponentList();
  string component_path ="../src/CCA/Components/xml";
// "../src/CCA/Components/xml";
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



void SCIRunLoader::readComponentDescription(const std::string& file)
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
    CCAComponentDescription* cd = new CCAComponentDescription(0); //TODO: assign null as CCA component model
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

void SCIRunLoader::destroyComponentList()
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    delete iter->second;
  }
  components.clear();
}
