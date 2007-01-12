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
 *  SCIRunLoader.cc: An instance of the SCIRun CCA Component Loader
 *
 *  Written by:
 *   Keming Zhang & Kosta
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#include <sci_defs/mpi_defs.h> // For MPIPP_H
#include <Framework/SCIRunLoader.h>
#include <Framework/CCA/CCAComponentModel.h>
#include <Framework/CCA/CCAComponentDescription.h>
#include <Framework/CCA/CCAComponentInstance.h>
#include <Framework/SCIRunFramework.h>
#include <Framework/resourceReference.h>

#include <Core/Containers/StringUtil.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Environment.h>
#include <Core/Util/soloader.h>
#include <Core/Thread/Mutex.h>
#include <Core/CCA/PIDL/PIDL.h>

#include <Core/CCA/spec/cca_sidl.h>

#include <iostream>
#include <sstream>

#include <sci_mpi.h>
using namespace std;

namespace SCIRun {

SCIRunLoader::SCIRunLoader(const std::string &loaderName,
			   const std::string & frameworkURL)
 :lock_components("SCIRunLoader::components lock")
{
  create_sci_environment(0,0);
}

//<<<<<<< SCIRunLoader.cc
int SCIRunLoader::createPInstance(const string& componentName, const string& componentType, const sci::cca::TypeMap::pointer& properties, SSIDL::array1<std::string> &componentURLs) {

#if 0
  //////////////////////////
  //create a group here
//   SSIDL::array1<int> nodeSet;
//   nodeSet=properties->getIntArray("nodes", nodeSet);
//   int mpi_size;
//   int mpi_rank;
//   MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
//   MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
//   MPI_Group world_group;
//   MPI_Comm_group(MPI_COMM_WORLD, &world_group);
//   bool skip=true;

//   //count how many nodes are involved
//   int ns=0;
//   for(unsigned int i=0;i<nodeSet.size(); i+=2){
//     ns+=nodeSet[i+1]-nodeSet[i]+1;
//   }
//   cerr<<"### ns = "<<ns<<endl;
//   int *ranks=new int[ns];
//   int index=0;
//   for(unsigned int i=0;i<nodeSet.size(); i+=2){
//     for(int k=nodeSet[i]; k<=nodeSet[i+1]; k++){
//       ranks[index++]=k;
//       cerr<<" rank[ " << index-1 << "="<<k<<endl;
//       if(k==mpi_rank) skip=false;
//     }
//   }
//   ////////////////////////////////////////////////
//   // Note that the subgroup has to be created in
//   // the world_group, ie, every world_group has
//   // to participate in the creatation process:
//   //   MPI_Group_incl and MPI_Comm_create.
//   MPI_Group group;
//   MPI_Group_incl(world_group,ns,ranks,&group );
//   delete ranks;
//   MPI_Comm MPI_COMM_COM;
//   MPI_Comm_create(MPI_COMM_WORLD, group, &MPI_COMM_COM);
//   /////////////////////////////////
//   if(skip){
//     componentURLs.resize(1);
//     componentURLs[0] = "";
//     return 0;
//   }
#endif

  //TODO: assume type is always good?
  std::string lastname=componentType.substr(componentType.find('.')+1);
  std::string so_name("libCCA_Components_");
  so_name = so_name + lastname + ".so";

  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
  if (!handle) {
    std::cerr << "Cannot load component " << componentType << std::endl;
    std::cerr << SOError() << std::endl;
    return 1;
  }
  std::string makername = "make_"+componentType;
  for (int i = 0; i < (int) makername.size(); i++) {
    if(makername[i] == '.') {
      makername[i] = '_';
    }
  }
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if (!maker_v) {
    std::cerr <<"Cannot load component " << componentType << std::endl;
    std::cerr << SOError() << std::endl;
    return 1;
  }
  sci::cca::Component::pointer (*maker)() = (sci::cca::Component::pointer (*)())(maker_v);
  sci::cca::Component::pointer component = (*maker)();
  //TODO: need keep a reference in the loader's creatation record.
  component->addReference();
//  component->setCommunicator((int)(&MPI_COMM_COM) );

  componentURLs.resize(1);
  componentURLs[0] = component->getURL().getString();
  return 0;
}



int SCIRunLoader::createInstance(const std::string& componentName, const std::string& componentType, const sci::cca::TypeMap::pointer& properties, std::string &componentURL) {

  //TODO: assume type is always good?

  std::cerr << "SCIRunLoader::getReferenceCount()=" << getReferenceCount() << std::endl;


  std::string lastname=componentType.substr(componentType.find('.')+1);
  std::string so_name("lib/libCCA_Components_");
  so_name = so_name + lastname + ".so";
  std::cerr << "componentType=" << componentType << " soname=" << so_name << std::endl;

  LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
  if (!handle) {
    std::cerr << "Cannot load component " << componentType << std::endl;
    std::cerr << SOError() << std::endl;
    return 1;
  }
  std::string makername = "make_"+componentType;
  for(int i = 0; i < (int) makername.size(); i++)
    if (makername[i] == '.')
      makername[i]='_';
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if (!maker_v) {
    std::cerr <<"Cannot load component " << componentType << std::endl;
    std::cerr << SOError() << std::endl;
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
  std::cerr << "SCIRunLoader::createInstance Done, rank/size="<<mpi_rank<<"/"<<mpi_size<<"\n";
  return 0;
}


int SCIRunLoader::destroyInstance(const std::string& componentName, float time)
{
  std::cerr<<"destroyInstance not implemented\n";
  return 0;
}

int SCIRunLoader::getAllComponentTypes(::SSIDL::array1< ::std::string>& componentTypes)
{
  std::cerr<<"listAllComponents() is called\n";
  buildComponentList();

  lock_components.lock();
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    componentTypes.push_back(iter->second->getType());
  }
  lock_components.unlock();
  return 0;
}

int SCIRunLoader::shutdown(float timeout)
{
  std::cerr<<"shutdown() is called: should unregisterLoader()\n";
  return 0;
}


SCIRunLoader::~SCIRunLoader()
{
  std::cerr << "SCIRun  Loader exits.\n";
  //abort();
}

#if 0
//
// std::string CCAComponentModel::createComponent(const std::string& name,
// 						      const std::string& type)
// {
//   sci::cca::Component::pointer component;
//   componentDB_type::iterator iter = components.find(type);
//   if(iter == components.end())
//     return "";
//     //ComponentDescription* cd = iter->second;

//   std::string lastname=type.substr(type.find('.')+1);
//   std::string so_name("lib/libCCA_Components_");
//   so_name=so_name+lastname+".so";
//   //std::cerr<<"type="<<type<<" soname="<<so_name<<std::endl;

//   LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
//   if(!handle){
//     std::cerr << "Cannot load component " << type << std::endl;
//     std::cerr << SOError() << std::endl;
//     return "";
//   }

//   std::string makername = "make_"+type;
//   for(int i=0;i<(int)makername.size();i++)
//     if(makername[i] == '.')
//       makername[i]='_';

//   void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
//   if(!maker_v){
//     std::cerr << "Cannot load component " << type << std::endl;
//     std::cerr << SOError() << std::endl;
//     return "";
//   }
//   sci::cca::Component::pointer (*maker)() = (sci::cca::Component::pointer (*)())(maker_v);
//   component = (*maker)();
//   //need to make sure addReference() will not cause problem
//   component->addReference();
//   return component->getURL().getString();
// }
//
#endif


//>>>>>>> 1.13
void SCIRunLoader::buildComponentList()
{
  destroyComponentList();

  std::string component_path = sci_getenv("SCIRUN_SRCDIR") + CCAComponentModel::DEFAULT_XML_PATH;
  std::vector<std::string> sArray;
  sArray.push_back(component_path);

  for (StringVector::const_iterator it = sArray.begin(); it != sArray.end(); it++) {
    Dir d(*it);
    std::vector<std::string> files;
    d.getFilenamesBySuffix(".xml", files);

    // copied from parseComponentModelXML (see ComponentModel.h)
    for(std::vector<std::string>::iterator iter = files.begin();
	iter != files.end(); iter++) {
      std::string xmlfile = *iter;
      readComponentDescriptions(*it + "/" + xmlfile);
    }
  }
}

/*
 * Essentially copied from parseComponentModelXML (see src/SCIRun/ComponentModel.h).
 * Only CCA components are supported.
 */
void SCIRunLoader::readComponentDescriptions(const std::string& file)
{
  static bool initialized = false;

  if (!initialized) {
    // check that libxml version in use is compatible with version
    // the software has been compiled against
    LIBXML_TEST_VERSION;
    initialized = true;
  }

  // create a parser context
  xmlParserCtxtPtr ctxt = xmlNewParserCtxt();
  if (ctxt == 0) {
    std::cerr << "ERROR: Failed to allocate parser context." << std::endl;
    return;
  }
  // parse the file, activating the DTD validation option
  xmlDocPtr doc = xmlCtxtReadFile(ctxt, file.c_str(), 0, (XML_PARSE_DTDATTR |
							  XML_PARSE_DTDVALID |
							  XML_PARSE_PEDANTIC));
  // check if parsing suceeded
  if (doc == 0) {
    std::cerr << "ERROR: Failed to parse " << file << std::endl;
    return;
  }

  // check if validation suceeded
  if (ctxt->valid == 0) {
    std::cerr << "ERROR: Failed to validate " << file << std::endl;
    return;
  }

  // this code does NOT check for includes!
  xmlNode* node = doc->children;
  for (; node != 0; node = node->next) {
    if (node->type == XML_ELEMENT_NODE &&
	std::string(to_char_ptr(node->name)) == std::string("metacomponentmodel")) {
      xmlAttrPtr nameAttr = get_attribute_by_name(node, "name");

      if (std::string(to_char_ptr(nameAttr->children->content)) == "cca") {
	xmlNode* libNode = node->children;
	for (;libNode != 0; libNode = libNode->next) {
	  if (libNode->type == XML_ELEMENT_NODE &&
	      std::string(to_char_ptr(libNode->name)) == std::string("library")) {
	    xmlAttrPtr nameAttrLib = get_attribute_by_name(libNode, "name");
	    if (nameAttrLib != 0) {
	      std::string component_type;
	      std::string library_name(to_char_ptr(nameAttrLib->children->content));
#if DEBUG
	      std::cerr << "Library name = ->" << library_name << "<-" << std::endl;
#endif
	      xmlNode* componentNode = libNode->children;
	      for (; componentNode != 0; componentNode = componentNode->next) {
		if (componentNode->type == XML_ELEMENT_NODE &&
		    std::string(to_char_ptr(componentNode->name)) == std::string("component")) {
		  xmlAttrPtr nameAttrComp = get_attribute_by_name(componentNode, "name");
		  if (nameAttrComp == 0) {
		    std::cerr << "ERROR: Component has no name." << std::endl;
		    component_type = "unknown type";
		  }
		  component_type = std::string(to_char_ptr(nameAttrComp->children->content));
#if DEBUG
		  std::cerr << "Component name = ->" << component_type << "<-" << std::endl;
#endif
		  setComponentDescription(component_type, library_name);
		}
	      }
	    }
	  }
	}
      }
    }
  }
  xmlFreeDoc(doc);
  // free up the parser context
  xmlFreeParserCtxt(ctxt);
  xmlCleanupParser();
}

void SCIRunLoader::setComponentDescription(const std::string& component_type, const std::string& library_name)
{
  CCAComponentDescription* cd = new CCAComponentDescription(0, component_type, library_name);
  Guard g(&lock_components);
  componentDB_type::iterator iter = components.find(cd->getType());
  if (iter != components.end()) {
    std::cerr << "WARNING: Multiple definitions exist for " << cd->getType() << std::endl;
  } else {
    //std::cout << "Added CCA component of type: " << cd->getType() << std::endl;
    components[cd->getType()] = cd;
  }
}

void SCIRunLoader::destroyComponentList()
{
  lock_components.lock();
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    delete iter->second;
  }
  components.clear();
  lock_components.unlock();
}

} // end namespace SCIRun
