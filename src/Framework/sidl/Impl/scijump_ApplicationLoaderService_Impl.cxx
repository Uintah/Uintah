// 
// File:          scijump_ApplicationLoaderService_Impl.cxx
// Symbol:        scijump.ApplicationLoaderService-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.ApplicationLoaderService
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_ApplicationLoaderService_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
#endif
#ifndef included_gov_cca_ConnectionID_hxx
#include "gov_cca_ConnectionID.hxx"
#endif
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_core_FrameworkService_hxx
#include "sci_cca_core_FrameworkService.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._includes)
#include <iostream>
#include <Core/Util/Environment.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <scijump_CCAException.hxx>

#include <libxml/catalog.h>
#include <fstream>
#include <iostream>

using namespace scijump;
using namespace SCIRun;

#define MAX_COMPONENTS 500
#define MAX_CONNECTIONS 250
// DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::ApplicationLoaderService_impl::ApplicationLoaderService_impl() : 
  StubBase(reinterpret_cast< void*>(
  ::scijump::ApplicationLoaderService::_wrapObj(reinterpret_cast< void*>(
  this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._ctor2)
  // Insert-Code-Here {scijump.ApplicationLoaderService._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._ctor2)
}

// user defined constructor
void scijump::ApplicationLoaderService_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._ctor)
  // Insert-Code-Here {scijump.ApplicationLoaderService._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._ctor)
}

// user defined destructor
void scijump::ApplicationLoaderService_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._dtor)
  // Insert-Code-Here {scijump.ApplicationLoaderService._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._dtor)
}

// static class initializer
void scijump::ApplicationLoaderService_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._load)
  // Insert-Code-Here {scijump.ApplicationLoaderService._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._load)
}

// user defined static methods:
/**
 * Method:  create[]
 */
::sci::cca::core::FrameworkService
scijump::ApplicationLoaderService_impl::create_impl (
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService.create)
  scijump::ApplicationLoaderService als = scijump::ApplicationLoaderService::_create();
  als.initialize(framework);
  return als;
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService.create)
}


// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::ApplicationLoaderService_impl::initialize_impl (
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService.initialize)
  this->framework = framework;
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService.initialize)
}

/**
 * Method:  getFileName[]
 */
::std::string
scijump::ApplicationLoaderService_impl::getFileName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService.getFileName)
  return fileName;
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService.getFileName)
}

/**
 * Method:  setFileName[]
 */
void
scijump::ApplicationLoaderService_impl::setFileName_impl (
  /* in */const ::std::string& filename ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService.setFileName)
  // throw exception if file name is empty?
  if (!filename.empty()) {
    this->fileName = filename;
  }
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService.setFileName)
}

/**
 * Method:  loadFile[]
 */
void
scijump::ApplicationLoaderService_impl::loadFile_impl (
  /* out array<gov.cca.ComponentID> */::sidl::array< ::gov::cca::ComponentID>& 
    cidList,
  /* out array<gov.cca.ConnectionID> */::sidl::array< 
    ::gov::cca::ConnectionID>& connList ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService.loadFile)
  if (fileName.empty()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Empty file name");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  }

  LIBXML_TEST_VERSION;

  xmlParserCtxtPtr ctxt; /* the parser context */
  xmlDocPtr doc; /* the resulting document tree */

  string srcDir = string(sci_getenv("SCIRUN_SRCDIR")) + string("/Framework/XML/application.dtd");

  cidList = ::sidl::array< ::gov::cca::ComponentID>::create1d(MAX_COMPONENTS);
  connList = ::sidl::array< ::gov::cca::ConnectionID>::create1d(MAX_CONNECTIONS);
  int cid_ctr = 0;
  int conn_ctr = 0;

  xmlInitializeCatalog();
  xmlCatalogAdd(BAD_CAST "public", BAD_CAST "-//SCIJump/Application DTD", BAD_CAST srcDir.c_str());

  /* create a parser context */
  ctxt = xmlNewParserCtxt();
  if (ctxt == 0) {
    // throw exception
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Failed to allocate parser context");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  }
  /* parse the file, activating the DTD validation option */
  doc = xmlCtxtReadFile(ctxt, fileName.c_str(), 0, (XML_PARSE_DTDATTR | XML_PARSE_DTDVALID | XML_PARSE_PEDANTIC));
  /* check if parsing suceeded */
  if (doc == 0) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Failed to parse " + fileName);
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  } else {
    /* check if validation suceeded */
    if (ctxt->valid == 0) {
      xmlFreeDoc(doc);
      scijump::CCAException ex = scijump::CCAException::_create();
      ex.setNote("Failed to validate " + fileName);
      ex.add(__FILE__, __LINE__, "loadFile");
      throw ex;
    }
  }

  scijump::SCIJumpFramework sj = babel_cast<scijump::SCIJumpFramework>(framework);
  if(sj._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Cannot cast framework pointer");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  }
  gov::cca::Services mainServices = sj.getServices("app svc", "main", 0);
  mainServices.registerUsesPort("mainBuilder", "cca.BuilderService", mainServices.createTypeMap());
  ::gov::cca::Port bsp = mainServices.getPort("mainBuilder");
  if (bsp._is_nil()) {
    xmlFreeDoc(doc);
    /* free up the parser context */
    xmlFreeParserCtxt(ctxt);
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("BuilderService not available");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  }
  scijump::BuilderService bs = babel_cast<scijump::BuilderService>(bsp);
  if (bs._is_nil()) {
    xmlFreeDoc(doc);
    /* free up the parser context */
    xmlFreeParserCtxt(ctxt);
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("BuilderService not available");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  }

  // iterate over nodes in XML tree and create instances, connections
  // position info -> GUI Service???
  xmlNode* node = doc->children;
  for(; node != 0; node = node->next) {
    if(node->type == XML_ELEMENT_NODE &&
       std::string(to_char_ptr(node->name)) == std::string("application")) {

      xmlNode* libNode = node->children;
      for(;libNode != 0; libNode = libNode->next) {
        if(libNode->type == XML_ELEMENT_NODE &&
           std::string(to_char_ptr(libNode->name)) == std::string("component")) {

          cidList.set(cid_ctr,readComponentNode(bs,&libNode)); 
	  cid_ctr++;
	  if(cid_ctr >= MAX_COMPONENTS) {
	    scijump::CCAException ex = scijump::CCAException::_create();
	    ex.setNote("Max number of components exceeded");
	    ex.add(__FILE__, __LINE__, "loadFile");
	    throw ex;
	  }
        }

        if(libNode->type == XML_ELEMENT_NODE &&
           std::string(to_char_ptr(libNode->name)) == std::string("connection")) {

          connList.set(conn_ctr,readConnectionNode(bs,&libNode));
	  conn_ctr++;
	  if(conn_ctr >= MAX_CONNECTIONS) {
	    scijump::CCAException ex = scijump::CCAException::_create();
	    ex.setNote("Max number of connections exceeded");
	    ex.add(__FILE__, __LINE__, "loadFile");
	    throw ex;
	  }
        }
      }

    }
  }

  mainServices.releasePort("mainBuilder");
  sj.releaseServices(mainServices);
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService.loadFile)
}

/**
 * Method:  loadFile[As]
 */
void
scijump::ApplicationLoaderService_impl::loadFile_impl (
  /* in */const ::std::string& filename,
  /* out array<gov.cca.ComponentID> */::sidl::array< ::gov::cca::ComponentID>& 
    cidList,
  /* out array<gov.cca.ConnectionID> */::sidl::array< 
    ::gov::cca::ConnectionID>& connList ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService.loadFileAs)
  this->fileName = filename;
  loadFile(cidList,connList);
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService.loadFileAs)
}

/**
 * Method:  saveFile[]
 */
void
scijump::ApplicationLoaderService_impl::saveFile_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService.saveFile)
  if (fileName.empty()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Empty file name");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
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
			   BAD_CAST "-//SCIJump/Application DTD",
			   BAD_CAST dtdstr.c_str());
  /*
   * xmlNewChild() creates a new node, which is "attached" as child node
   * of rootNode node.
   */
  //xmlNewProp(rootNode, BAD_CAST "version", BAD_CAST SCIJUMP_VERSION);

  scijump::SCIJumpFramework sj = babel_cast<scijump::SCIJumpFramework>(framework);
  if(sj._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Cannot cast framework pointer");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  }

  gov::cca::Services mainServices = sj.getServices("app svc", "main", sj.createTypeMap());
  mainServices.registerUsesPort("mainBuilder", "cca.BuilderService", mainServices.createTypeMap());
  ::gov::cca::Port bsp = mainServices.getPort("mainBuilder");
  if (bsp._is_nil()) {
    xmlFreeDoc(xmlDoc);
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("BuilderService not available");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  }

  scijump::BuilderService bs = babel_cast<scijump::BuilderService>(bsp);
  if (bs._is_nil()) {
    xmlFreeDoc(xmlDoc);
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("BuilderService not available");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  }

  // get componentIDs, names, GUI positions -> GUI Service???
  ::sidl::array< ::gov::cca::ComponentID> cidArray = bs.getComponentIDs();
  ::sidl::array< ::gov::cca::ConnectionID> connidArray = bs.getConnectionIDs(cidArray);
  for(int i=0; i < cidArray.length(); i++) {
    gov::cca::TypeMap props = bs.getComponentProperties(cidArray.get(i));
    // Is this a framework service (an internal component)?
    // If so, do not save.
    // Builder information is not stored either, but it would be a good idea
    // to associate components with their Builders if a Builder was used.
    bool isInternalComponent = props.getBool("internal component", false);
    if (! isInternalComponent) {
      writeComponentNode(cidArray.get(i), props, &rootNode);
    }
  }

  for(int i=0; i < connidArray.length(); i++) {
    writeConnectionNode(connidArray.get(i), &rootNode);
  }

  mainServices.releasePort("mainBuilder");
  sj.releaseServices(mainServices);

  xmlSaveFormatFileEnc(fileName.c_str(), xmlDoc, "UTF-8", 1);
  // free the document
  xmlFreeDoc(xmlDoc);
  xmlDoc = 0;
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService.saveFile)
}

/**
 * Method:  saveFile[As]
 */
void
scijump::ApplicationLoaderService_impl::saveFile_impl (
  /* in */const ::std::string& filename ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService.saveFileAs)
  this->fileName = filename;
  saveFile();
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService.saveFileAs)
}


// DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._misc)
void 
scijump::ApplicationLoaderService_impl::writeComponentNode(gov::cca::ComponentID cid,
							   gov::cca::TypeMap& properties,
							   xmlNode** rootNode)
{
  sci::cca::core::ComponentInfo cinfo = babel_cast< sci::cca::core::ComponentInfo>(cid);
  if(cinfo._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Bad cast");
    ex.add(__FILE__, __LINE__, "loadFile");
    throw ex;
  } 
  xmlNode* componentNode = xmlNewChild(*rootNode, 0, BAD_CAST "component", 0);
  xmlNewProp(componentNode, BAD_CAST "name", BAD_CAST cinfo.getInstanceName().c_str());
  xmlNewProp(componentNode, BAD_CAST "classname", BAD_CAST cinfo.getClassName().c_str());
  std::string b = properties.getString("builder name", "");
  if (! b.empty()) {
    xmlNewProp(componentNode, BAD_CAST "builder", BAD_CAST b.c_str());
  }
  xmlAddChild(*rootNode, componentNode);
}

void 
scijump::ApplicationLoaderService_impl::writeConnectionNode(gov::cca::ConnectionID cid, 
							    xmlNode** rootNode)
{
  gov::cca::ComponentID user = cid.getUser();
  gov::cca::ComponentID provider = cid.getProvider();

  //not writing the GO and UI port connections 
  if(user.getInstanceName() == "SCIRun.GUIBuilder") return;

  xmlNode* connectionNode = xmlNewChild(*rootNode, 0, BAD_CAST "connection", 0);

  xmlNewProp(connectionNode, BAD_CAST "user", BAD_CAST user.getInstanceName().c_str());
  xmlNewProp(connectionNode, BAD_CAST "usesport", BAD_CAST cid.getUserPortName().c_str());
  xmlNewProp(connectionNode, BAD_CAST "provider", BAD_CAST provider.getInstanceName().c_str());
  xmlNewProp(connectionNode, BAD_CAST "providesport", BAD_CAST cid.getProviderPortName().c_str());
  xmlAddChild(*rootNode, connectionNode);
}


::gov::cca::ComponentID
scijump::ApplicationLoaderService_impl::readComponentNode(sci::cca::ports::BuilderService& bs, xmlNode** node)
{
  xmlAttrPtr nameAttr = get_attribute_by_name(*node, "name");
  xmlAttrPtr classAttr = get_attribute_by_name(*node, "classname");
  if((nameAttr != 0)&&(classAttr != 0)) {
    std::string component_name(to_char_ptr(nameAttr->children->content));
    std::string class_name(to_char_ptr(classAttr->children->content));
    std::cerr << "component_name = " << component_name << "\n";
    return bs.createInstance(component_name, class_name, 0);
  }
}


::gov::cca::ConnectionID
scijump::ApplicationLoaderService_impl::readConnectionNode(sci::cca::ports::BuilderService& bs, xmlNode** node)
{

  xmlAttrPtr userAttr = get_attribute_by_name(*node, "user");
  xmlAttrPtr usesportAttr = get_attribute_by_name(*node, "usesport");
  xmlAttrPtr providerAttr = get_attribute_by_name(*node, "provider");
  xmlAttrPtr providesportAttr = get_attribute_by_name(*node, "providesport");

  if((userAttr != 0)&&(usesportAttr != 0)&&
     (providerAttr != 0)&&(providesportAttr != 0)) {

    std::string user(to_char_ptr(userAttr->children->content));
    std::string usesport(to_char_ptr(usesportAttr->children->content));
    std::string provider(to_char_ptr(providerAttr->children->content));
    std::string providesport(to_char_ptr(providesportAttr->children->content));

    gov::cca::ComponentID user_cid = bs.getComponentID(user);
    gov::cca::ComponentID provider_cid = bs.getComponentID(provider);

    return bs.connect(user_cid,usesport,provider_cid,providesport);
  }
  else {
    //error
  }
}
// DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._misc)

