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
#include <Core/Util/Environment.h>
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

namespace SCIRun {

const std::string CCAComponentModel::DEFAULT_PATH =
    std::string("/CCA/Components/xml");


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
      iter != components.end(); iter++) {
    delete iter->second;
  }
  components.clear();
}

void CCAComponentModel::buildComponentList()
{
    // Initialize the XML4C system
    try {
        XMLPlatformUtils::Initialize();
    }
    catch (const XMLException& toCatch) {
        std::cerr << "Error during initialization! :" <<
            std::endl << StrX(toCatch.getMessage()) << std::endl;
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
        std::vector<std::string> files;
        d.getFilenamesBySuffix(".cca", files);
        for(std::vector<std::string>::iterator iter = files.begin();
            iter != files.end();
            iter++) {
            std::string& file = *iter;
            readComponentDescription(*it+"/"+file);
        }
    }
}

void CCAComponentModel::readComponentDescription(const std::string& file)
{
  // Instantiate the DOM parser.
  XercesDOMParser parser;
  parser.setDoValidation(false);
  
  SCIRunErrorHandler handler;
  parser.setErrorHandler(&handler);
  
  try {
    parser.parse(file.c_str());
 } catch (const XMLException& toCatch) {
    std::cerr << "Error during parsing: '" <<
      file << "'\nException message is:  " <<
      xmlto_string(toCatch.getMessage()) << std::endl;
    handler.foundError=true;
    return;
 }
  
  DOMDocument* doc = parser.getDocument();
  DOMNodeList* list = doc->getElementsByTagName(to_xml_ch_ptr("component"));
  int nlist = list->getLength();
  if(nlist == 0) {
    std::cerr << "WARNING: file " << file << " has no components!" << std::endl;
  }
  for (int i=0;i<nlist;i++) {
    DOMNode* d = list->item(i);

    //should use correct Loader pointer below.
    CCAComponentDescription* cd = new CCAComponentDescription(this);

    // Is this component a cca component?
    DOMNode* model= d->getAttributes()->getNamedItem(to_xml_ch_ptr("model"));
    if (model != 0) {  
      if ( strcmp(to_char_ptr(model->getNodeValue()), this->prefixName.c_str()) != 0 ) {
        // not a cca component, ignore it
        continue;
      }
    } else { // No model, ignore this component
      std::cerr << "ERROR: Component has no model in file << " << file << std::endl;
      continue;
    }
 
    DOMNode* name = d->getAttributes()->getNamedItem(to_xml_ch_ptr("name"));
    if (name==0) {
      std::cout << "ERROR: Component has no name." << std::endl;
      cd->type = "unknown type";
    } else {
      cd->type = to_char_ptr(name->getNodeValue());
    }
    
    componentDB_type::iterator iter = components.find(cd->type);
    if(iter != components.end()) {
      std::cerr << "WARNING: Component multiply defined: " << cd->type << std::endl;
    } else {
      //std::cerr << "Added CCA component of type: " << cd->type << std::endl;
      components[cd->type]=cd;
    }
  }
}

sci::cca::Services::pointer
CCAComponentModel::createServices(const std::string& instanceName,
                  const std::string& className,
                  const sci::cca::TypeMap::pointer& properties)
{
  CCAComponentInstance* ci = new CCAComponentInstance(framework,
                              instanceName, className,
                              properties,
                              sci::cca::Component::pointer(0));
  framework->registerComponent(ci, instanceName);
  ci->addReference();
  return sci::cca::Services::pointer(ci);
}

bool CCAComponentModel::destroyServices(const sci::cca::Services::pointer& svc)
{
    CCAComponentInstance *ci =
    dynamic_cast<CCAComponentInstance*>(svc.getPointer());
    if (ci == 0) {
        return false;
    }
    framework->unregisterComponent(ci->instanceName);
    ci->deleteReference();
    return true;
}

bool CCAComponentModel::haveComponent(const std::string& type)
{
  std::cerr << "CCA looking for component of type: " << type << std::endl;
  return components.find(type) != components.end();
}



ComponentInstance* CCAComponentModel::createInstance(const std::string& name,
                             const std::string& type,
                             const sci::cca::TypeMap::pointer& properties)

{
  std::string loaderName="";
  if(!properties.isNull()){
    properties->addReference();
    loaderName=properties->getString("LOADER NAME","");
  }
  std::cerr<<"creating cca component <" <<
      name << "," << type << "> with loader:"
           << loaderName << std::endl;
  sci::cca::Component::pointer component;
  if (loaderName=="") {  //local component
    componentDB_type::iterator iter = components.find(type);
    if(iter == components.end()) {
      std::cerr << "Error: could not locate any cca components.  Make sure the paths set in environment variable \"SIDL_XML_PATH\" are correct." << std::endl;
      return 0;
    }
    //ComponentDescription* cd = iter->second;
    
    std::string lastname=type.substr(type.find('.')+1);  
    std::string so_name("lib/libCCA_Components_");
    so_name=so_name+lastname+".so";
    LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
    if(!handle) {
      std::cerr << "Cannot load component " << type << std::endl;
      std::cerr << SOError() << std::endl;
      return 0;
    }
    
    std::string makername = "make_"+type;
    for(int i=0;i<(int)makername.size();i++)
      if(makername[i] == '.')
    makername[i]='_';
    
    void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
    if(!maker_v) {
      std::cerr <<"Cannot load component " << type << std::endl;
      std::cerr << SOError() << std::endl;
      return 0;
    }
    sci::cca::Component::pointer (*maker)() = (sci::cca::Component::pointer (*)())(maker_v);
    component = (*maker)();
 } else { 
    //use loader to load the component
    resourceReference* loader=getLoader(loaderName);
    std::vector<int> nodes;
    nodes.push_back(0);
    Object::pointer comObj=loader->createInstance(name, type, nodes);
    component=pidl_cast<sci::cca::Component::pointer>(comObj);
    properties->putInt("np",loader->getSize() );
  }
  CCAComponentInstance* ci = new CCAComponentInstance(framework, name, type,
                                                      properties, //sci::cca::TypeMap::pointer(0),
                                                      component);
  //  ci->addReference(); //not necessary, will be removed
  component->setServices(sci::cca::Services::pointer(ci));
  return ci;
}

bool CCAComponentModel::destroyInstance(ComponentInstance *ci)
{
  CCAComponentInstance* cca_ci = dynamic_cast<CCAComponentInstance*>(ci);
  if(!cca_ci) {
    std::cerr << "error: in destroyInstance() cca_ci is 0" << std::endl;
    return false;
  }
  cca_ci->deleteReference();
  return true;  
}

std::string CCAComponentModel::getName() const
{
  return "CCA";
}

void CCAComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list,
                          bool /*listInternal*/)
{
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++)
    {
    list.push_back(iter->second);
    }
  
  for(unsigned int i=0; i<loaderList.size(); i++)
    {
    ::SSIDL::array1<std::string> typeList;
    loaderList[i]->listAllComponentTypes(typeList);
    //convert typeList to component description list
    //by attaching a loader (resourceReferenece) to it.
    for(unsigned int j=0; j<typeList.size(); j++)
      {
      CCAComponentDescription* cd = new CCAComponentDescription(this);
      cd->type=typeList[j];
      cd->setLoaderName(loaderList[i]->getName());
      list.push_back(cd);
      }
    }  
}

int CCAComponentModel::addLoader(resourceReference *rr)
{
  loaderList.push_back(rr);
  std::cerr<<"Loader "<<rr->getName()<<" is added into cca component model"<<std::endl;
  return 0;
}

int CCAComponentModel::removeLoader(const std::string &loaderName)
{
  resourceReference *rr=getLoader(loaderName);
  if(rr!=0)
    {
    std::cerr<<"loader "<<rr->getName()<<" is removed from cca component model\n";
    delete rr;
    }
  else
    {
    std::cerr<<"loader "<<loaderName<<" not found in cca component model\n";
    }
  return 0;
}

resourceReference *
CCAComponentModel::getLoader(std::string loaderName)
{
  resourceReference *rr=0;
  for(unsigned int i=0; i<loaderList.size(); i++)
    {
    if(loaderList[i]->getName()==loaderName)
      {
      rr=loaderList[i];
      break;
      }
    }
  return rr;
}

} // end namespace SCIRun
