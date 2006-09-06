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


/* ComponentNode.cc
 * 
 * written by 
 *   Chris Moulding
 *   Sept 2000
 *   Copyright (c) 2000
 *   University of Utah
 */

#include <Dataflow/Network/ComponentNode.h>
#include <Dataflow/GuiInterface/GuiInterface.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <libxml/xmlreader.h>
#include <Core/Util/RWS.h>
#include <Core/Malloc/Allocator.h>

#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <sstream>
#include <fstream>

namespace SCIRun {
using std::map;
using std::cout;
using std::endl;
using std::ostringstream;

template <class PInfo>
void 
set_port_info(vector<PInfo*> &ports, xmlNodePtr snode)
{
  xmlNodePtr ipnode = snode->children;
  for (; ipnode != 0; ipnode = ipnode->next) {
    if (string(to_char_ptr(ipnode->name)) == string("port"))
    {
      PInfo *pinfo = scinew PInfo;
      xmlNodePtr pnode = ipnode->children;
      for (; pnode != 0; pnode = pnode->next) {
	if (string(to_char_ptr(pnode->name)) == string("name")) {
	  pinfo->name = string(to_char_ptr(pnode->children->content));
	}
	if (string(to_char_ptr(pnode->name)) == string("datatype")) 
	{
	  pinfo->datatype = string(to_char_ptr(pnode->children->content));
	}
      }
      ports.push_back(pinfo);
    }
  } 
}


bool
set_port_info(ModuleInfo &mi, const xmlNodePtr cnode)
{
  xmlNodePtr node = cnode->children;
  for (; node != 0; node = node->next) {
    if (string(to_char_ptr(node->name)) == string("io")) {
      xmlNodePtr ionode = node->children;
      for (; ionode != 0; ionode = ionode->next) {
	if (ionode && (string(to_char_ptr(ionode->name)) == 
		       string("inputs"))) 
	{
	  xmlAttrPtr lpd_att = get_attribute_by_name(ionode, 
						     "lastportdynamic");
	  if (lpd_att) {
	    mi.last_port_dynamic_ = false;
	    if (string(to_char_ptr(lpd_att->children->content)) == 
		string("yes")) 
	    {
	      mi.last_port_dynamic_ = true;
	    }
	  } else {
	    std::cerr << "Missing attribute lastportdynamic for module: " 
		      << mi.module_name_ << std::endl;
	    return false;
	  }
	  // set input port info.
	  set_port_info(mi.iports_, ionode);
	}
	else if (ionode && (string(to_char_ptr(ionode->name)) == 
			    string("outputs"))) 
	{
	  // set input port info.
	  set_port_info(mi.oports_, ionode);
	} 
      }
    } 
  }
  return false;
}


bool
set_description(ModuleInfo &mi, const xmlNodePtr cnode)
{
  xmlNodePtr onode = cnode->children;
  for (; onode != 0; onode = onode->next) {
    if (string(to_char_ptr(onode->name)) == string("overview")) {
      
      xmlNodePtr node = onode->children;
      for (; node != 0; node = node->next) {
	if (string(to_char_ptr(node->name)) == string("description")) 
	{
	  mi.help_description_ = get_serialized_children(node);
	  return true;
	} 
	else if (string(to_char_ptr(node->name)) == string("summary")) 
	{
	  mi.summary_ = get_serialized_children(node);
	} 
	else if (string(to_char_ptr(node->name)) == string("authors")) 
	{
	  xmlNodePtr anode = node->children;
	  for (; anode != 0; anode = anode->next) {
	    if (string(to_char_ptr(anode->name)) == string("author")) 
	    {
	      string s = string(to_char_ptr(anode->children->content));
	      mi.authors_.push_back(s);
	    }
	  }
	} 
      }
    }
  }
  return false;
}


bool
set_gui_info(ModuleInfo &mi, const xmlNodePtr cnode)
{
  mi.has_gui_node_ = false;
  xmlNodePtr onode = cnode->children;
  for (; onode != 0; onode = onode->next) {
    if (string(to_char_ptr(onode->name)) == string("gui")) {
      mi.has_gui_node_ = true;
      return true;
    }
  }
  return false;
}

void
write_component_file(const ModuleInfo &mi, const char* filename) 
{
  xmlDocPtr doc = 0;        /* document pointer */
  xmlNodePtr root_node = 0; /* node pointers */
  xmlDtdPtr dtd = 0;        /* DTD pointer */
  
  LIBXML_TEST_VERSION;
  
  /* 
   * Creates a new document, a node and set it as a root node
   */
  doc = xmlNewDoc(BAD_CAST "1.0");
  root_node = xmlNewNode(0, BAD_CAST "component");
  xmlDocSetRootElement(doc, root_node);
  
  /*
   * Creates a DTD declaration.
   */
  string fname(filename);
  string dtdstr;
  if (fname.find("Package") == string::npos) {
    dtdstr = string("component.dtd");
  } else {
    dtdstr = string("../../../../Dataflow/XML/component.dtd");
  }
  dtd = xmlCreateIntSubset(doc, BAD_CAST "component", 0, 
			   BAD_CAST dtdstr.c_str());
  
  /* 
   * xmlNewChild() creates a new node, which is "attached" as child node
   * of root_node node. 
   */
  xmlNewProp(root_node, BAD_CAST "name", BAD_CAST mi.module_name_.c_str());
  xmlNewProp(root_node, BAD_CAST "category", BAD_CAST mi.category_name_.c_str());
  const char* opt = mi.optional_ ? "true" : "false";
  xmlNewProp(root_node, BAD_CAST "optional", BAD_CAST opt);

  // overview
  xmlNodePtr node = xmlNewChild(root_node, 0, BAD_CAST "overview", 0); 
  // authors
  xmlNodePtr tmp = xmlNewChild(node, 0, BAD_CAST "authors", 0); 
  vector<string>::const_iterator aiter = mi.authors_.begin();
  while (aiter != mi.authors_.end()) {
    const string &a = *aiter++;
    xmlNewChild(tmp, 0, BAD_CAST "author", BAD_CAST a.c_str()); 
  }
  // summary
  xmlNewChild(node, 0, BAD_CAST "summary", BAD_CAST mi.summary_.c_str());
  // description
  tmp = xmlNewChild(node, 0, BAD_CAST "description", 0);


  // trim out the <p> </p> from the first and last line added from 
  // the temp xml file that the component wizard creates, and add 
  // the 'p' node instead.
  
  size_t s = mi.help_description_.find("<p>", 0);
  size_t e = mi.help_description_.find("</p>", 0);
  s = mi.help_description_.find("\n", s) + 1;
  e = mi.help_description_.rfind("\n", e) - 1;

  string hdtrimmed = mi.help_description_.substr(s, e - s);
  
  xmlNewChild(tmp, 0, BAD_CAST "p", 
	      BAD_CAST hdtrimmed.c_str());
  
 
  // io
  node = xmlNewChild(root_node, 0, BAD_CAST "io", 0); 

  tmp = xmlNewChild(node, 0, BAD_CAST "inputs", 0); 
  const char* lpd = mi.last_port_dynamic_ ? "yes" : "no";
  xmlNewProp(tmp, BAD_CAST "lastportdynamic", BAD_CAST lpd);
  vector<IPortInfo*>::const_iterator iter = mi.iports_.begin();
  while(iter != mi.iports_.end()) 
  {
    IPortInfo *p = *iter++;
    xmlNodePtr port = xmlNewChild(tmp, 0, BAD_CAST "port", 0); 
    xmlNewChild(port, 0, BAD_CAST "name", BAD_CAST p->name.c_str()); 
    xmlNewChild(port, 0, BAD_CAST "datatype", 
		BAD_CAST p->datatype.c_str()); 
  }

  if (mi.oports_.size()) {
    tmp = xmlNewChild(node, 0, BAD_CAST "outputs", 0); 
  
    vector<OPortInfo*>::const_iterator iter = mi.oports_.begin();
    while(iter != mi.oports_.end()) 
    {
      OPortInfo *p = *iter++;
      xmlNodePtr port = xmlNewChild(tmp, 0, BAD_CAST "port", 0); 
      xmlNewChild(port, 0, BAD_CAST "name", BAD_CAST p->name.c_str()); 
      xmlNewChild(port, 0, BAD_CAST "datatype", 
		  BAD_CAST p->datatype.c_str()); 
    }
  }

  // write the file
  xmlSaveFormatFileEnc(filename, doc, "UTF-8", 1);
  
  // free the document
  xmlFreeDoc(doc);
}

bool
read_component_file(ModuleInfo &mi, const char* filename) 
{
  /*
   * this initialize the library and check potential ABI mismatches
   * between the version it was compiled for and the actual shared
   * library used.
   */
  LIBXML_TEST_VERSION;
  
  xmlParserCtxtPtr ctxt; /* the parser context */
  xmlDocPtr doc; /* the resulting document tree */

  /* create a parser context */
  ctxt = xmlNewParserCtxt();
  if (ctxt == 0) {
    std::cerr << "ComponentNode.cc: Failed to allocate parser context" 
	      << std::endl;
    return false;
  }
  /* parse the file, activating the DTD validation option */
  doc = xmlCtxtReadFile(ctxt, filename, 0, (XML_PARSE_DTDATTR | 
					    XML_PARSE_DTDVALID | 
					    XML_PARSE_NOERROR));
  /* check if parsing suceeded */
  if (doc == 0 || ctxt->valid == 0) {

    xmlError* error = xmlCtxtGetLastError(ctxt);
    GuiInterface *gui = GuiInterface::getSingleton();
    string mtype = "Parse ";
    if (doc) {
      mtype = "Validation ";
    }
    ostringstream msg;
    msg << "createSciDialog -error -title \"Component XML " 
	<< mtype << "Error\" -message \"" 
	<< endl << mtype << "Failure for: " << endl << filename << endl
	<< endl << "Error Message: " << endl << error->message << endl << "\"";
    gui->eval(msg.str());
    return false;
  } 
  
  xmlNode* node = doc->children;
  for (; node != 0; node = node->next) {
    // skip all but the component node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("component")) 
    {
      //! set all the ModuleInfo
      xmlAttrPtr name_att = get_attribute_by_name(node, "name");
      xmlAttrPtr cat_att = get_attribute_by_name(node, "category");
      
      // set the component attributes.
      if (name_att == 0 || cat_att == 0) {
	std::cerr << "Attibutes missing from component node in : " 
		  << filename << std::endl;
	return false;
      }
      mi.module_name_ = string(to_char_ptr(name_att->children->content));
      mi.category_name_ = string(to_char_ptr(cat_att->children->content));
      xmlAttrPtr opt_att = get_attribute_by_name(node, "optional");
      mi.optional_ = false;
      if (opt_att != 0) {
	if (string(to_char_ptr(opt_att->children->content)) == 
	    string("true")) {
	  mi.optional_ = true;
	}
      } 
      //set the description string.
      set_description(mi, node);
      set_port_info(mi, node);
      set_gui_info(mi, node);
    }
  }

  xmlFreeDoc(doc);
  /* free up the parser context */
  xmlFreeParserCtxt(ctxt);  
#ifndef _WIN32
  // there is a problem on windows when using Uintah 
  // which is either caused or exploited by this
  xmlCleanupParser();
#endif
  return true;
}

} // End namespace SCIRun

