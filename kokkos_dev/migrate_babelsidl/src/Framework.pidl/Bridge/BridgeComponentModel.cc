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
 *  BridgeComponentModel.cc:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#include <Framework/Bridge/BridgeComponentModel.h>
#include <Framework/Bridge/BridgeComponentDescription.h>
#include <Framework/Bridge/BridgeComponentInstance.h>
#include <Framework/SCIRunFramework.h>

#include <Dataflow/Network/Network.h>
#include <Core/OS/Dir.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Framework/resourceReference.h>
#include <string>

#include <iostream>

#ifndef DEBUG
#define DEBUG 0
#endif

using namespace SCIRun;

const std::string BridgeComponentModel::DEFAULT_XML_PATH("/src/CCA/Components/xml");

int BridgeComponent::bridgeID(0);

BridgeComponentModel::BridgeComponentModel(SCIRunFramework* framework)
  : ComponentModel("bridge", framework),
    lock_components("BridgeComponentModel::components lock")
{
  buildComponentList();
}

BridgeComponentModel::~BridgeComponentModel()
{
  destroyComponentList();
}

void BridgeComponentModel::destroyComponentList()
{
  SCIRun::Guard g1(&lock_components);

  for (componentDB_type::iterator iter = components.begin();
       iter != components.end(); iter++) {
    delete iter->second;
  }
  components.clear();
}

// Find and parse '*.bridge' files (contain references to saved bridges).
void BridgeComponentModel::buildComponentList(const StringVector& files)
{
  destroyComponentList();
  std::string component_path = sci_getenv("SCIRUN_SRCDIR") + DEFAULT_XML_PATH;
  while (component_path != "") {
    unsigned int firstColon = component_path.find(':');
    std::string dir;
    if (firstColon < component_path.size()) {
      dir = component_path.substr(0, firstColon);
      component_path = component_path.substr(firstColon+1);
    } else {
      dir = component_path;
      component_path = "";
    }
    Dir d(dir);
    std::vector<std::string> files;
    d.getFilenamesBySuffix(".bridge", files);
    for (std::vector<std::string>::iterator iter = files.begin();
	 iter != files.end(); iter++) {
      std::string file = *iter;
      readComponentDescriptions(dir + "/" + file);
    }
  }
}

// XML (no DTD):
// <component name="some name"/>
void BridgeComponentModel::readComponentDescriptions(const std::string& file)
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

  // no DTD, so don't validate
  // parse the file, activating the DTD validation option
  xmlDocPtr doc = xmlCtxtReadFile(ctxt, file.c_str(), 0, XML_PARSE_RECOVER);
  // check if parsing suceeded
  if (doc == 0) {
    std::cerr << "ERROR: Failed to parse " << file << std::endl;
    // free up the parser context
    xmlFreeParserCtxt(ctxt);
    xmlCleanupParser();
    return;
  }
  // this code does NOT check for includes!
  xmlNode* node = doc->children;
  for (; node != 0; node = node->next) {
    if (node->type == XML_ELEMENT_NODE &&
      std::string(to_char_ptr(node->name)) == std::string("component")) {
      xmlAttrPtr nameAttrComp = get_attribute_by_name(node, "name");
      if (nameAttrComp == 0) {
	// error, but keep parsing
	std::cerr << "ERROR: Component with missing name." << std::endl;
	continue;
      }
      std::string component_type(to_char_ptr(nameAttrComp->children->content));

#if DEBUG
      std::cerr << "Component name = ->" << component_type << "<-" << std::endl;

#endif
      setComponentDescription(component_type);
    }
  }
  xmlFreeDoc(doc);
  // free up the parser context
  xmlFreeParserCtxt(ctxt);
  xmlCleanupParser();
  return;
}

void
BridgeComponentModel::setComponentDescription(const std::string& type, const std::string& library)
{
  // check for duplicates
  BridgeComponentDescription* cd = new BridgeComponentDescription(this, type);
  Guard g(&lock_components);
  componentDB_type::iterator iter = components.find(cd->getType());
  if (iter != components.end()) {
    std::cerr << "WARNING: Multiple definitions exist for " << cd->getType() << std::endl;
  } else {
    components[cd->getType()] = cd;
#if DEBUG
      std::cerr << "Added Bridge component of type: " << cd->getType() << std::endl;
#endif
  }
}

BridgeServices*
BridgeComponentModel::createServices(const std::string& instanceName,
				     const std::string& className,
				     const sci::cca::TypeMap::pointer& tm)
{
  BridgeComponentInstance* ci = new BridgeComponentInstance(framework, instanceName, className, tm, 0);
  framework->registerComponent(ci, instanceName);
  return ci;
}

bool BridgeComponentModel::haveComponent(const std::string& type)
{
  SCIRun::Guard g1(&lock_components);
#if DEBUG
  std::cerr << "Bridge looking for component of type: " << type << '\n';
#endif
  return components.find(type) != components.end();
}



ComponentInstance*
BridgeComponentModel::createInstance(const std::string& name,
				     const std::string& t,
				     const sci::cca::TypeMap::pointer &tm)

{
  std::string type(t);
  std::string loaderName;
#if DEBUG
  std::cerr << "creating component <" << name << "," << type << "> with loader:" << loaderName << std::endl;
#endif
  BridgeComponent* component;
  if (loaderName == "") {  //local component
    lock_components.lock();
    componentDB_type::iterator iter = components.find(type);
    lock_components.unlock();
    std::string so_name;
    if (iter == components.end()) {
      //on the fly building of bridges (don't have specific .cca files)
      type = type.substr(type.find(":")+1); //removing bridge:
      std::string lastname = type.substr(type.find('.')+1);
      so_name = "on-the-fly-libs/"+lastname+".so";
    } else {
      std::string lastname = type.substr(type.find('.')+1);
      so_name = "lib/libCCA_Components_"+lastname+".so";
    }
    LIBRARY_HANDLE handle = GetLibraryHandle(so_name.c_str());
    if (!handle) {
      std::cerr << "Cannot load component " << type << '\n';
      std::cerr << SOError() << '\n';
      return 0;
    }

    std::string makername = "make_"+type;
    for (int i = 0;i < (int)makername.size();i++)
      if (makername[i] == '.')
	makername[i] = '_';

    void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
    if (!maker_v) {
      std::cerr << "Cannot load component " << type << '\n';
      std::cerr << SOError() << '\n';
      return 0;
    }
    BridgeComponent* (*maker)() = (BridgeComponent* (*)())(maker_v);
    component = (*maker)();
  }
  else{
    //No way to remotely load bridge components for now
    return NULL;
  }
  BridgeComponentInstance* ci = new BridgeComponentInstance(framework, name, type, tm, component);
  component->setServices(ci);
  return ci;
}

bool BridgeComponentModel::destroyInstance(ComponentInstance *ci)
{
  BridgeComponentInstance* cca_ci = dynamic_cast<BridgeComponentInstance*>(ci);
  if (! cca_ci) {
    std::cerr << "error: in destroyInstance() cca_ci is 0" << std::endl;
    return false;
  }
  return true;
}

void BridgeComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list,
						 bool /*listInternal*/)
{
  SCIRun::Guard g1(&lock_components);
  for (componentDB_type::iterator iter = components.begin();
       iter != components.end(); iter++) {
    list.push_back(iter->second);
  }
}
