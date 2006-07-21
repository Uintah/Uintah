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

#include <SCIRun/ApplicationLoader.h>
#include <SCIRun/StandAlone/sr2_version.h>
#include <Core/Util/Environment.h>
#include <Core/XMLUtil/XMLUtil.h>

#include <libxml/catalog.h>
#include <fstream>
#include <iostream>

namespace SCIRun {

const std::string ApplicationLoader::APPLICATION_FILE_EXTENSION("app");

void ApplicationLoader::writeComponentNode()
{
}


void ApplicationLoader::writeConnectionNode()
{
}

void ApplicationLoader::readComponentNode()
{
}


void ApplicationLoader::readConnectionNode()
{
}

void ApplicationLoader::setFileName(const std::string& fn)
{
  // throw exception if file name is empty?
  if (! fn.empty()) {
    fileName = fn;
  }
}

// options: get builder as argument or use GUIService to get any and all builders?
// OR call ApplicationLoader from GUIService?
bool ApplicationLoader::loadNetworkFile()
{
  if (fileName.empty()) {
    return false;
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
    std::cerr << "ApplicationLoader.cc: Failed to allocate parser context" << std::endl;
    return false;
  }
  /* parse the file, activating the DTD validation option */
  doc = xmlCtxtReadFile(ctxt, fileName.c_str(), 0, (XML_PARSE_DTDATTR | XML_PARSE_DTDVALID | XML_PARSE_PEDANTIC));
  /* check if parsing suceeded */
  if (doc == 0) {
    std::cerr << "ApplicationLoader.cc: Failed to parse " << fileName << std::endl;
    return false;
  } else {
    /* check if validation suceeded */
    if (ctxt->valid == 0) {
      std::cerr << "ApplicationLoader.cc: Failed to validate " << fileName << std::endl;
      return false;
    }
  }

//   GuiInterface *gui = GuiInterface::getSingleton();
//   gui->eval("::netedit dontschedule");

  // parse the doc at network node.
  //process_network_node(doc->children);

  xmlFreeDoc(doc);
  /* free up the parser context */
  xmlFreeParserCtxt(ctxt);
// #ifndef _WIN32
//   // there is a problem on windows when using Uintah
//   // which is either caused or exploited by this
//   xmlCleanupParser();
// #endif

//   gui->eval("setGlobal NetworkChanged 0");
//   gui->eval("set netedit_savefile " + net_file_);
//   gui->eval("::netedit scheduleok");

  // first draw autoview.
//   autoview_pending_ = true;
//   return true;

//   if (filename.empty()) {
//     //displayMsg("Error: cannot load file with empty file name.");
//     // use GUIService to display error message
//     return;
//   }
//   std::ifstream is(filename.c_str());

//   int numMod = 0;
//   int numConn = 0;
//   std::string modName;
//   int modName_x;
//   int modName_y;
//   std::vector<sci::cca::ComponentID::pointer> cidTable;

//   is >> numMod >> numConn;
//   std::cout << "numMod=" << numMod << std::endl;
//   std::cout << "numConn=" << numConn << std::endl;

//   // If there's a error creating a component, stop trying to load
//   // the network file until there are improvements to the
//   // network file format.
//   try {
// //     sci::cca::ports::BuilderService::pointer builder = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
//     for (int i = 0; i < numMod; i++) {
//       is >> modName >> modName_x >> modName_y;

//       sci::cca::ComponentID::pointer cid;
//       TypeMap *tm = new TypeMap;
//       tm->putInt("x", modName_x);
//       tm->putInt("y", modName_y);

//       cid = builder->createInstance(modName,
//                                     modName, sci::cca::TypeMap::pointer(tm));

//       if (! cid.isNull()) {
//         if (modName != "SCIRun.Builder") {
//           cidTable.push_back(cid);
//         }
//       }
//     }

//     if (cidTable.size() < 2) {
//       // report error
//       unsetCursor();
//       return;
//     }

//     for (int i = 0; i < numConn; i++) {
//       int iu, ip;
//       std::string up, pp;
//       is >> iu >> up >> ip >> pp;
//       //networkCanvasView->addPendingConnection(cidTable[iu], up, cidTable[ip], pp);
//       sci::cca::ConnectionID::pointer connID =
//         builder->connect(cidTable[iu], up, cidTable[ip], pp);
//     }
//   }
//   catch(const sci::cca::CCAException::pointer &pe) {
//     //displayMsg(pe->getNote());
//   }
//   // is this still needed? - for SCIRun Dataflow maybe?
//   catch(const Exception &e) {
//     //displayMsg(e.message());
//   }
//   catch(...) {
//     //displayMsg("Caught unexpected exception while loading network.");
//   }

// //   services->releasePort("cca.BuilderService");
//   is.close();
  return false;
}

bool ApplicationLoader::saveNetworkFileAs(const std::string& fn)
{
  fileName = fn;
  return saveNetworkFile();
}

bool ApplicationLoader::saveNetworkFile()
{
  if (fileName.empty()) {
    return false;
  }
  xmlNodePtr root_node = 0; /* node pointers */
  xmlDtdPtr dtd = 0;        /* DTD pointer */

  LIBXML_TEST_VERSION;

  /*
   * Creates a new document, a node and set it as a root node
   */
  xmlDoc = xmlNewDoc(BAD_CAST "1.0");
  root_node = xmlNewNode(0, BAD_CAST "application");
  nodeStack.push(root_node);
  xmlDocSetRootElement(xmlDoc, root_node);

  /*
   * Creates a DTD declaration.
   */
  string dtdstr = string("network.dtd");

  dtd = xmlCreateIntSubset(xmlDoc, BAD_CAST "application",
			   BAD_CAST "-//SCIRun2/Application DTD",
			   BAD_CAST dtdstr.c_str());

  /*
   * xmlNewChild() creates a new node, which is "attached" as child node
   * of root_node node.
   */
  xmlNewProp(root_node, BAD_CAST "version", BAD_CAST SR2_VERSION);

//   QCanvasItemList tempQCL = miniCanvas->allItems();
//   setCursor(Qt::WaitCursor);
//   std::ofstream saveOutputFile(filename.c_str());

//   const ModuleMap *modules = networkCanvasView->getModules();
//   std::vector<Module*> saveModules;
//   std::vector<Connection*> saveConnections = networkCanvasView->getConnections();

//   saveOutputFile << modules->size() << std::endl;
//   saveOutputFile << saveConnections.size() << std::endl;

//   if (saveOutputFile.is_open()) {
//     for (ModuleMap::const_iterator iter = modules->begin(); iter != modules->end(); iter++) {
//       saveOutputFile << iter->second->componentID()->getInstanceName() << std::endl;
//       saveOutputFile << iter->second->x() << std::endl;
//       saveOutputFile << iter->second->y() << std::endl;
//       saveModules.push_back(iter->second);
//     }

//     // inefficient (O(n^2)), but will change with improvements to network file format
//     for (unsigned int k = 0; k < saveConnections.size(); k++) {
//       Module* um = saveConnections[k]->usesPort()->module();
//       Module* pm = saveConnections[k]->providesPort()->module();
//       unsigned int iu = 0;
//       unsigned int ip = 0;

//       for (unsigned int i = 0; i < saveModules.size();i++) {
//         if (saveModules[i] == um) {
//           iu = i;
//         }
//         if (saveModules[i] == pm) {
//           ip = i;
//         }
//       }
//       saveOutputFile << iu << " " <<
//         saveConnections[k]->usesPort()->name() << " " << ip << " " <<
//         saveConnections[k]->providesPort()->name() << std::endl;
//     }
//   }
//   saveOutputFile.close();
//   unsetCursor();
  xmlSaveFormatFileEnc(fileName.c_str(), xmlDoc, "UTF-8", 1);

  // free the document
  xmlFreeDoc(xmlDoc);
  xmlDoc = 0;

  return true;
}


}
