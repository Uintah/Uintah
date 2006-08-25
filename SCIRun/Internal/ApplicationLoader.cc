/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation

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

#include <SCIRun/Internal/ApplicationLoader.h>
#include <SCIRun/Internal/BuilderService.h>
#include <SCIRun/StandAlone/sr2_version.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAException.h>

#include <Core/Util/Environment.h>
#include <Core/XMLUtil/XMLUtil.h>

#include <libxml/catalog.h>
#include <fstream>
#include <iostream>

namespace SCIRun {

const std::string ApplicationLoader::APP_EXT("xml");

ApplicationLoader::ApplicationLoader(SCIRunFramework* framework)
  : InternalFrameworkServiceInstance(framework, "internal:ApplicationLoaderService")
{
}

InternalFrameworkServiceInstance*
ApplicationLoader::create(SCIRunFramework* framework)
{
  ApplicationLoader* n = new ApplicationLoader(framework);
  return n;
}

sci::cca::ComponentID::pointer
ApplicationLoader::createInstance(const std::string& instanceName,
                                  const std::string& className,
                                  const sci::cca::TypeMap::pointer& properties)
{
  if (instanceName.size()) {
    if (framework->lookupComponent(instanceName) != 0) {
      throw CCAExceptionPtr(new CCAException("Component instance name " + instanceName + " is not unique"));
    }
    return framework->createComponentInstance(instanceName, className, properties);
  }
  return framework->createComponentInstance(framework->getUniqueName(className), className, properties);
}

void ApplicationLoader::setFileName(const std::string& fn)
{
  // throw exception if file name is empty?
  if (! fn.empty()) {
    fileName = fn;
  }
}

void ApplicationLoader::loadFile(const std::string& fn)
{
  fileName = fn;
  loadFile();
}

// options: get builder as argument or use GUIService to get any and all builders?
// OR call ApplicationLoader from GUIService?
void ApplicationLoader::loadFile()
{
  if (fileName.empty()) {
    throw CCAExceptionPtr(new CCAException("Empty file name"));
  }

  LIBXML_TEST_VERSION;

  xmlParserCtxtPtr ctxt; /* the parser context */
  xmlDocPtr doc; /* the resulting document tree */

  string srcDir = string(sci_getenv("SCIRUN_SRCDIR")) + string("/SCIRun/XML/application.dtd");

  xmlInitializeCatalog();
  xmlCatalogAdd(BAD_CAST "public", BAD_CAST "-//SCIRun2/Application DTD", BAD_CAST srcDir.c_str());

  /* create a parser context */
  ctxt = xmlNewParserCtxt();
  if (ctxt == 0) {
    // throw exception
    throw CCAExceptionPtr(new CCAException("Failed to allocate parser context"));
  }
  /* parse the file, activating the DTD validation option */
  doc = xmlCtxtReadFile(ctxt, fileName.c_str(), 0, (XML_PARSE_DTDATTR | XML_PARSE_DTDVALID | XML_PARSE_PEDANTIC));
  /* check if parsing suceeded */
  if (doc == 0) {
    throw CCAExceptionPtr(new CCAException("Failed to parse " + fileName));
  } else {
    /* check if validation suceeded */
    if (ctxt->valid == 0) {
      xmlFreeDoc(doc);
      throw CCAExceptionPtr(new CCAException("Failed to validate " + fileName));
    }
  }

  sci::cca::Port::pointer bsp =
    framework->getFrameworkService("cca.BuilderService", "cca.ApplicationLoaderService");
  if (bsp.isNull()) {
    xmlFreeDoc(doc);
    /* free up the parser context */
    xmlFreeParserCtxt(ctxt);
    throw CCAExceptionPtr(new CCAException("BuilderService not available"));
  }
  sci::cca::ports::BuilderService::pointer bs =
    pidl_cast<sci::cca::ports::BuilderService::pointer>(bsp);
  if (bs.isNull()) {
    xmlFreeDoc(doc);
    /* free up the parser context */
    xmlFreeParserCtxt(ctxt);
    throw CCAExceptionPtr(new CCAException("BuilderService not available"));
  }

  // iterate over nodes in XML tree and create instances, connections
  // position info -> GUI Service???

  framework->releaseFrameworkService("cca.BuilderService", "cca.ApplicationLoaderService");

  xmlFreeDoc(doc);
  /* free up the parser context */
  xmlFreeParserCtxt(ctxt);
}

void ApplicationLoader::saveFile(const std::string& fn)
{
  fileName = fn;
  return saveFile();
}

// component ids, names from builder service, component positions from GUI builder
void ApplicationLoader::saveFile()
{
  if (fileName.empty()) {
    throw CCAExceptionPtr(new CCAException("Empty file name"));
  }
  xmlNodePtr rootNode = 0; /* node pointers */
  xmlDtdPtr dtd = 0;        /* DTD pointer */

  LIBXML_TEST_VERSION;

  /*
   * Creates a new document, a node and set it as a root node
   */
  xmlDoc = xmlNewDoc(BAD_CAST "1.0");
  rootNode = xmlNewNode(0, BAD_CAST "application");
  nodeStack.push(rootNode);
  xmlDocSetRootElement(xmlDoc, rootNode);

  /*
   * Creates a DTD declaration.
   */
  string dtdstr = string("application.dtd");

  dtd = xmlCreateIntSubset(xmlDoc, BAD_CAST "application",
			   BAD_CAST "-//SCIRun2/Application DTD",
			   BAD_CAST dtdstr.c_str());
  /*
   * xmlNewChild() creates a new node, which is "attached" as child node
   * of rootNode node.
   */
  xmlNewProp(rootNode, BAD_CAST "version", BAD_CAST SR2_VERSION);

  sci::cca::Port::pointer bsp =
    framework->getFrameworkService("cca.BuilderService", "cca.ApplicationLoaderService");
  if (bsp.isNull()) {
    xmlFreeDoc(xmlDoc);
    throw CCAExceptionPtr(new CCAException("BuilderService not available"));
  }
  sci::cca::ports::BuilderService::pointer bs =
    pidl_cast<sci::cca::ports::BuilderService::pointer>(bsp);
  if (bs.isNull()) {
    xmlFreeDoc(xmlDoc);
    throw CCAExceptionPtr(new CCAException("BuilderService not available"));
  }

  // get componentIDs, names, GUI positions -> GUI Service???
  ComponentIDPtrList cidArray = bs->getComponentIDs();
  ConnectionIDPtrList connIDArray = bs->getConnectionIDs(cidArray);
  for (ComponentIDPtrList::iterator cidIter = cidArray.begin();
       cidIter != cidArray.end(); cidIter++) {
    sci::cca::TypeMap::pointer props = bs->getComponentProperties(*cidIter);
    // Is this a framework service (an internal component)?
    // If so, do not save.
    // Builder information is not stored either, but it would be a good idea
    // to associate components with their Builders if a Builder was used.
    bool isInternalComponent = props->getBool("internal component", false);
    if (! isInternalComponent) {
      xmlNode *node = writeComponentNode(*cidIter, props, &rootNode);
      xmlAddChild(rootNode, node);
    }
  }

  for (ConnectionIDPtrList::iterator connIDIter = connIDArray.begin();
       connIDIter != connIDArray.end(); connIDIter++) {
    xmlNode *node = writeConnectionNode(*connIDIter, &rootNode);
    xmlAddChild(rootNode, node);
  }

  framework->releaseFrameworkService("cca.BuilderService", "cca.ApplicationLoaderService");

  xmlSaveFormatFileEnc(fileName.c_str(), xmlDoc, "UTF-8", 1);
  // free the document
  xmlFreeDoc(xmlDoc);
  xmlDoc = 0;
}

xmlNode* ApplicationLoader::writeComponentNode(const sci::cca::ComponentID::pointer& cid,
                                               const sci::cca::TypeMap::pointer& properties,
                                               xmlNode** rootNode)
{
  xmlNode* componentNode = xmlNewChild(*rootNode, 0, BAD_CAST "component", 0);
  xmlNewProp(componentNode, BAD_CAST "name", BAD_CAST cid->getInstanceName().c_str());
  std::string b = properties->getString("builder name", "");
  if (! b.empty()) {
    xmlNewProp(componentNode, BAD_CAST "builder", BAD_CAST b.c_str());
  }
  return componentNode;
}


xmlNode* ApplicationLoader::writeConnectionNode(const sci::cca::ConnectionID::pointer& cid, xmlNode** rootNode)
{
  xmlNode* connectionNode = xmlNewChild(*rootNode, 0, BAD_CAST "connection", 0);
  sci::cca::ComponentID::pointer user = cid->getUser();
  sci::cca::ComponentID::pointer provider = cid->getProvider();

  xmlNewProp(connectionNode, BAD_CAST "user", BAD_CAST user->getInstanceName().c_str());
  xmlNewProp(connectionNode, BAD_CAST "usesport", BAD_CAST cid->getUserPortName().c_str());
  xmlNewProp(connectionNode, BAD_CAST "provider", BAD_CAST provider->getInstanceName().c_str());
  xmlNewProp(connectionNode, BAD_CAST "providesport", BAD_CAST cid->getProviderPortName().c_str());
  return connectionNode;
}

void ApplicationLoader::readComponentNode()
{
}


void ApplicationLoader::readConnectionNode()
{
}


}
